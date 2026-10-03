import { spawn } from "node:child_process";
import { isAbsolute } from "node:path";
import { createInterface } from "node:readline";
import { RequestError } from "./request-error";

// Native tokenizers avoids transformers.js's quadratic full-text token counting.
// Request text exists only in the process pipe and memory; nothing is written to disk.
const SCRIPT = `
import json, sys
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(sys.argv[1])
tokenizer.no_padding()
tokenizer.no_truncation()
print("ready", flush=True)
for line in sys.stdin:
    try:
        count = len(tokenizer.encode(json.loads(line), add_special_tokens=False).ids)
    except Exception:
        count = "invalid"
    print(count, flush=True)
`;
export interface TokenCounter {
  count(text: string): Promise<number>;
  ready(): boolean;
  close(): void;
}

export async function createTokenCounter(
  python: string,
  tokenizerPath: string,
  onUnavailable?: () => void
): Promise<TokenCounter> {
  if (!(isAbsolute(python) && isAbsolute(tokenizerPath))) {
    throw new Error(
      "Native tokenizer requires absolute local executable and artifact paths"
    );
  }
  const child = spawn(python, ["-u", "-c", SCRIPT, tokenizerPath], {
    stdio: ["pipe", "pipe", "ignore"],
    env: { ...process.env, TOKENIZERS_PARALLELISM: "false" },
  });
  const lines = createInterface({ input: child.stdout });
  let closed = false;
  let awaiting: { resolve(value: string): void; reject(): void } | undefined;
  const close = (): void => {
    if (closed) {
      return;
    }
    closed = true;
    awaiting?.reject();
    awaiting = undefined;
    lines.close();
    child.kill();
    onUnavailable?.();
  };
  child.once("error", close);
  child.once("exit", close);
  child.stdin.on("error", close);
  lines.on("line", (line: string) => {
    if (awaiting) {
      const waiting = awaiting;
      awaiting = undefined;
      waiting.resolve(line);
    } else {
      close();
    }
  });
  const receive = (): Promise<string> =>
    new Promise((resolve, reject) => {
      if (closed) {
        reject(new Error("Native tokenizer unavailable"));
        return;
      }
      const timeout = setTimeout(close, 10_000);
      awaiting = {
        resolve: (line) => {
          clearTimeout(timeout);
          resolve(line);
        },
        reject: () => {
          clearTimeout(timeout);
          reject(new Error("Native tokenizer unavailable"));
        },
      };
    });
  if ((await receive()) !== "ready") {
    close();
    throw new Error("Native tokenizer failed to initialize");
  }
  let previous: Promise<unknown> = Promise.resolve();
  return {
    ready: () => !closed,
    count(text) {
      const result = previous.then(async () => {
        const response = receive();
        child.stdin.write(`${JSON.stringify(text)}\n`);
        const line = await response;
        if (line === "invalid") {
          throw new RequestError(400, "invalid_input");
        }
        const value = Number(line);
        if (!Number.isSafeInteger(value) || value < 0) {
          close();
          throw new Error("Invalid tokenizer response");
        }
        return value;
      });
      previous = result.catch(() => undefined);
      return result;
    },
    close,
  };
}
