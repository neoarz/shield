// Reports TEST_BUN_VERSION as the running Bun version.
Object.defineProperty(process.versions, "bun", {
  value: process.env.TEST_BUN_VERSION,
});
