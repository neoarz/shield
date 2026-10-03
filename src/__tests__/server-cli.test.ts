import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const CLI = fileURLToPath(new URL("../server/cli.ts", import.meta.url));
const BUN_VERSION = fileURLToPath(
  new URL("./fixtures/bun-version.mjs", import.meta.url)
);
const bun = spawnSync("bun", ["--version"]).status === 0;

/** Runs the private server CLI with only these settings, until it exits. */
function start(env: Record<string, string>, preload: string[] = []) {
  return spawnSync("bun", [...preload, CLI], {
    env: {
      PATH: process.env.PATH,
      HOME: process.env.HOME,
      SHIELD_ARTIFACT_MANIFEST: "/missing/manifest.json",
      SHIELD_TOKENIZER_PYTHON: "/missing/python",
      SHIELD_PRIVATE_TOKEN: "private-test-token-with-at-least-32-characters",
      ...env,
    },
    encoding: "utf8",
    timeout: 20_000,
  });
}

describe.skipIf(!bun)("private server CLI", () => {
  it("names an invalid capacity setting, never its value, and exits", () => {
    for (const name of [
      "SHIELD_CONCURRENCY",
      "SHIELD_GPU_CONCURRENCY",
      "SHIELD_MAX_QUEUE",
      "SHIELD_QUEUE_TIMEOUT_MS",
      "SHIELD_REQUEST_TIMEOUT_MS",
      "SHIELD_UPLOAD_TIMEOUT_MS",
      "SHIELD_MAX_UPLOADS",
    ]) {
      const result = start({ [name]: "-7319" });
      expect(result.status).toBe(1);
      expect(result.stderr).toContain(name);
      expect(result.stderr).not.toContain("7319");
    }
  }, 30_000);

  it("refuses Bun releases older than 1.4.2", () => {
    for (const [version, refused] of [
      ["1.3.12", true],
      ["1.4.1", true],
      ["1.4.2", false],
      ["1.10.0", false],
    ] as const) {
      const result = start({ TEST_BUN_VERSION: version }, [
        "--preload",
        BUN_VERSION,
      ]);
      expect(result.status).toBe(1);
      expect(result.stderr.includes("Bun 1.4.2")).toBe(refused);
    }
  }, 30_000);
});
