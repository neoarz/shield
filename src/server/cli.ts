import { once } from "node:events";
import { assertCudaValidation } from "./cuda-validation";
import type { ShieldHttpServer } from "./http";
import {
  createLocalClassifier,
  createShieldServer,
  loadArtifactManifest,
  type ShieldServerOptions,
} from "./index";
import { InferenceGate } from "./inference-gate";
import { createModelProcess } from "./model-process";

/** Client disconnects cancel inference only from Bun 1.4.2 on. */
const MIN_BUN = [1, 4, 2];
/** The longest delay setTimeout keeps; a longer one fires at once. */
const MAX_TIMEOUT_MS = 2_147_483_647;

/** A startup error safe to print: it names what to fix, never a setting's value. */
class ConfigurationError extends Error {}

function integer(
  name: string,
  fallback: number,
  minimum: number,
  maximum: number
): number {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new ConfigurationError(`Invalid ${name}`);
  }
  return value;
}

function capacityOptions(): Omit<
  ShieldServerOptions,
  "classifier" | "bearerToken"
> {
  const size = (name: string, fallback: number, minimum = 1): number =>
    integer(name, fallback, minimum, Number.MAX_SAFE_INTEGER);
  const delay = (name: string, fallback: number): number =>
    integer(name, fallback, 1, MAX_TIMEOUT_MS);
  return {
    concurrency: size("SHIELD_CONCURRENCY", 1),
    gpuConcurrency: size("SHIELD_GPU_CONCURRENCY", 1),
    maxQueue: size("SHIELD_MAX_QUEUE", 8, 0),
    queueTimeoutMs: delay("SHIELD_QUEUE_TIMEOUT_MS", 5000),
    requestTimeoutMs: delay("SHIELD_REQUEST_TIMEOUT_MS", 24_000),
    uploadTimeoutMs: delay("SHIELD_UPLOAD_TIMEOUT_MS", 5000),
    maxUploads: size("SHIELD_MAX_UPLOADS", 16),
  };
}

function supportedBun(version: string): boolean {
  const parts = version.split(".").map((part) => Number.parseInt(part, 10));
  for (let i = 0; i < MIN_BUN.length; i++) {
    if (parts[i] !== MIN_BUN[i]) {
      return parts[i] > MIN_BUN[i];
    }
  }
  return true;
}

async function main(): Promise<void> {
  const manifestPath = process.env.SHIELD_ARTIFACT_MANIFEST;
  const bearerToken = process.env.SHIELD_PRIVATE_TOKEN;
  const pythonPath = process.env.SHIELD_TOKENIZER_PYTHON;
  if (!(manifestPath && pythonPath && bearerToken) || bearerToken.length < 32) {
    throw new ConfigurationError(
      "Missing SHIELD_ARTIFACT_MANIFEST, SHIELD_TOKENIZER_PYTHON, or a SHIELD_PRIVATE_TOKEN of at least 32 characters"
    );
  }
  const pool = process.env.SHIELD_POOL ?? "paid";
  const largeDevice = process.env.SHIELD_LARGE_DEVICE ?? "cpu";
  if (
    (pool !== "free" && pool !== "paid") ||
    (largeDevice !== "cpu" && largeDevice !== "cuda") ||
    (pool === "free" && largeDevice !== "cpu")
  ) {
    throw new ConfigurationError("Invalid SHIELD_POOL or SHIELD_LARGE_DEVICE");
  }
  if (largeDevice === "cuda" && process.env.NVIDIA_TF32_OVERRIDE !== "0") {
    throw new ConfigurationError(
      "CUDA serving requires full-precision matmul settings (NVIDIA_TF32_OVERRIDE=0)"
    );
  }
  const threads = integer("SHIELD_THREADS", 2, 1, 256);
  const port = integer("SHIELD_PORT", 8789, 1, 65_535);
  const healthPort = integer("SHIELD_HEALTH_PORT", 8790, 1, 65_535);
  if (healthPort === port) {
    throw new ConfigurationError(
      "The readiness listener requires its own SHIELD_HEALTH_PORT"
    );
  }
  const capacity = capacityOptions();
  const bun = process.versions.bun;
  if (bun !== undefined && !supportedBun(bun)) {
    throw new ConfigurationError(
      `Bun ${bun} does not cancel inference when a client disconnects; run Bun 1.4.2 or later`
    );
  }
  const manifest = await loadArtifactManifest(manifestPath, { pool });
  if (largeDevice === "cuda") {
    await assertCudaValidation(
      process.env.SHIELD_CUDA_VALIDATION_REPORT,
      manifest
    );
  }
  const cpuGates = {
    direct: new InferenceGate(),
    ensemble: new InferenceGate(),
  };
  const classifier = await createLocalClassifier(manifest, {
    threads,
    pythonPath,
    pool,
    largeDevice,
    warmup: true,
    createModel: (model, lane) =>
      createModelProcess({
        model,
        pythonPath,
        worker: new URL("./model-worker.ts", import.meta.url),
        gate: model.device === "cuda" ? undefined : cpuGates[lane],
      }),
  });
  let server: ShieldHttpServer | undefined;
  try {
    server = createShieldServer({ classifier, bearerToken, ...capacity });
    // The default binding cannot receive public traffic. Private network use is explicit.
    const bind = process.env.SHIELD_BIND ?? "127.0.0.1";
    server.listen(port, bind);
    server.readiness.listen(healthPort, bind);
    await Promise.all([
      once(server, "listening"),
      once(server.readiness, "listening"),
    ]);
  } catch (error) {
    // Running model workers would keep this process alive with nothing listening.
    server?.close();
    server?.readiness.close();
    await classifier.close?.();
    throw error;
  }
  let stopping = false;
  const stop = async (): Promise<void> => {
    if (stopping) {
      return;
    }
    stopping = true;
    await server.drain();
    await classifier.close?.();
    server.close();
    server.readiness.close();
  };
  const onStop = (): void => {
    stop().catch(() => {
      process.exitCode = 1;
    });
  };
  process.once("SIGTERM", onStop);
  process.once("SIGINT", onStop);
}

main().catch((error: unknown) => {
  process.stderr.write(
    error instanceof ConfigurationError
      ? `Shield private server failed to start: ${error.message}.\n`
      : "Shield private server failed to start; verify local artifacts and configuration.\n"
  );
  process.exitCode = 1;
});
