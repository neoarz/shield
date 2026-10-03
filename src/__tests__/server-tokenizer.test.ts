import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createTokenCounter } from "../server/tokenizer";

// The real tokenizer loop runs; only the native tokenizers package is replaced.
const probe = spawnSync(
  "python3",
  ["-c", "import sys; print(sys.executable)"],
  { encoding: "utf8" }
);
const python = probe.status === 0 ? probe.stdout.trim() : "";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe.skipIf(!python)("native tokenizer process", () => {
  it("refuses an input the tokenizer cannot read and keeps counting", async () => {
    vi.stubEnv(
      "PYTHONPATH",
      fileURLToPath(new URL("./fixtures", import.meta.url))
    );
    vi.stubEnv("PYTHONDONTWRITEBYTECODE", "1");
    const unavailable = vi.fn();
    const counter = await createTokenCounter(
      python,
      "/test/tokenizer.json",
      unavailable
    );
    try {
      await expect(counter.count("hello world")).resolves.toBe(2);
      await expect(counter.count("hello \ud800")).rejects.toMatchObject({
        status: 400,
        code: "invalid_input",
      });
      expect(counter.ready()).toBe(true);
      await expect(counter.count("one two three")).resolves.toBe(3);
      expect(unavailable).not.toHaveBeenCalled();
    } finally {
      counter.close();
    }
  });
});
