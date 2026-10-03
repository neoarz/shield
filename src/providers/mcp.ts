/**
 * Wraps an MCP client (`@modelcontextprotocol/sdk`): the tools a server
 * lists are checked for tool poisoning, what its tools, resources, and
 * prompts return is checked for injection before it reaches a model, and
 * with a tool policy, each tool call is checked against it first.
 */

import type { DetectResult } from "../detect";
import {
  InjectionDetectedError,
  OutputBlockedError,
  ToolPolicyError,
} from "../errors";
import { type ScanOutputOptions, scanOutputText } from "../output/scan";
import { atLeast } from "../output/util";
import type { ToolPolicy } from "../policy";
import {
  pinTools,
  type ScanToolsOptions,
  scanToolsAsync,
  type ToolDefinition,
  type ToolPins,
  type ToolScanResult,
} from "../tools";
import { createShield, jsonText, type ShieldProviderOptions } from "./guard";
import { decodeTextBlob, isRecord, withOverrides } from "./shared";

export interface ShieldMcpOptions
  extends Pick<
    ShieldProviderOptions,
    | "detect"
    | "scanToolResults"
    | "onDetection"
    | "onInjectionDetected"
    | "requireFullCoverage"
  > {
  /**
   * Options for checking the tools `listTools` returns with `scanTools`, or
   * `false` to leave them unchecked. Default: the detect options tool
   * results use.
   */
  scanTools?: ScanToolsOptions | false;
  /**
   * What happens to a tool `scanTools` flags: `"drop"` (default) leaves it
   * out of the list, `"throw"` throws `InjectionDetectedError` with source
   * `"tool"`, and `"warn"` keeps it.
   */
  onFlaggedTools?: "drop" | "throw" | "warn";
  /** Called with each flagged tool and its scan result, in every mode. */
  onToolFlagged?: (tool: ToolDefinition, result: ToolScanResult) => void;
  /**
   * Pins each tool's definition the first time `listTools` returns it
   * unflagged, and flags the tool as `changed_since_pinned` when a later
   * list returns a different definition (a "rug pull"). Pass an object to
   * keep pins across sessions: the client adds new tools to it, so save it
   * with `JSON.stringify` and pass it back next time. To accept a changed
   * tool, delete its entry. `false` turns pinning off. Default: pins kept
   * for the life of the wrapped client.
   */
  pins?: ToolPins | false;
  /**
   * Refuse `callTool` for a tool the latest `listTools` flagged, with
   * `InjectionDetectedError`, before the server is called. Dropping a tool
   * only hides it from the list; this stops a call to it by name. Default
   * `true`, except with `onFlaggedTools: "warn"`.
   */
  blockFlaggedToolCalls?: boolean;
  /**
   * Checks the arguments of every `callTool` with `scanOutputText()` before
   * the server is called, and throws `OutputBlockedError` for any high or
   * critical finding, such as a credential or a link that carries data out.
   * `false` turns it off. Default: the `scanOutputText()` defaults, secrets
   * and exfiltration links.
   */
  scanArguments?: ScanOutputOptions | false;
  /**
   * A tool policy from `createToolPolicy()`, for one session. The tools
   * `listTools` returns are declared to it, as this client's own, `callTool`
   * runs `policy.checkAsync()` before the server is called and throws
   * `ToolPolicyError` when the policy refuses the call, and each tool result,
   * or error the call failed with, is recorded with `policy.recordResult()`,
   * flagged when detection found an injection in it. A resource, a prompt,
   * or a task result read by its task ID, or its error, with an injection
   * is recorded with `policy.recordUntrusted()`. Default: none.
   */
  policy?: ToolPolicy;
}

/**
 * Text of tool call arguments for the output detectors, all of it: they
 * run in linear time, and a cut would let padding hide what follows it.
 */
function argumentText(args: unknown): string {
  if (typeof args === "string") {
    return args;
  }
  try {
    return JSON.stringify(args) ?? "";
  } catch {
    return "";
  }
}

const RISKS: DetectResult["risk"][] = [
  "none",
  "low",
  "medium",
  "high",
  "critical",
];

function isString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

/** Text of a resource's contents: its text, or a text blob decoded. */
function resourceText(resource: unknown): string {
  if (!isRecord(resource)) {
    return "";
  }
  return typeof resource.text === "string"
    ? resource.text
    : decodeTextBlob(resource.blob, resource.mimeType);
}

/** Text a model reads from a content block: text, an embedded resource, or a resource link's name, title, and description. */
function blockText(block: unknown): string {
  if (!isRecord(block)) {
    return "";
  }
  switch (block.type) {
    case "text":
      return typeof block.text === "string" ? block.text : "";
    case "resource":
      return resourceText(block.resource);
    case "resource_link":
      return [block.name, block.title, block.description]
        .filter(isString)
        .join("\n");
    default:
      return "";
  }
}

function joined(texts: string[]): string {
  return texts.filter(Boolean).join("\n");
}

const list = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

/**
 * Text of a tool call result: its content blocks, the keys and strings in
 * its structured content, and a legacy `toolResult`.
 */
function toolResultText(result: unknown): string {
  if (!isRecord(result)) {
    return "";
  }
  const texts = list(result.content).map(blockText);
  if (result.structuredContent !== undefined) {
    texts.push(jsonText(result.structuredContent));
  }
  if (result.toolResult !== undefined) {
    texts.push(jsonText(result.toolResult));
  }
  return joined(texts);
}

/** Text of an error a call failed with: its message, and the keys and strings in its data. */
function errorText(error: unknown): string {
  if (!isRecord(error)) {
    return typeof error === "string" ? error : "";
  }
  return joined([
    typeof error.message === "string" ? error.message : "",
    error.data === undefined ? "" : jsonText(error.data),
  ]);
}

function resourcesText(result: unknown): string {
  return isRecord(result)
    ? joined(list(result.contents).map(resourceText))
    : "";
}

/** Text of a prompt: its description and every message's content. */
function promptText(result: unknown): string {
  if (!isRecord(result)) {
    return "";
  }
  const texts = list(result.messages).map((message) =>
    isRecord(message) ? blockText(message.content) : ""
  );
  return joined([
    typeof result.description === "string" ? result.description : "",
    ...texts,
  ]);
}

const methodOf = (request: unknown): unknown =>
  isRecord(request) ? request.method : undefined;

const paramsOf = (request: unknown): unknown =>
  isRecord(request) ? request.params : undefined;

function isFlagged(scan: ToolScanResult): boolean {
  return scan.result.detected || scan.issues.length > 0;
}

/**
 * One error for every flagged tool: the highest risk found, or `"low"` when
 * the tools were flagged only for issues, and every category and issue.
 */
function flaggedToolsError(flagged: ToolScanResult[]): InjectionDetectedError {
  let risk = 1;
  const categories = new Set<string>();
  for (const scan of flagged) {
    if (scan.result.detected) {
      risk = Math.max(risk, RISKS.indexOf(scan.result.risk));
    }
    for (const match of scan.result.matches) {
      categories.add(match.category);
    }
    for (const issue of scan.issues) {
      categories.add(issue);
    }
  }
  return new InjectionDetectedError(RISKS[risk], [...categories], "tool");
}

function toolScanOptions(options: ShieldMcpOptions): ScanToolsOptions | null {
  if (options.scanTools === false) {
    return null;
  }
  if (options.scanTools) {
    return options.scanTools;
  }
  if (typeof options.scanToolResults === "object") {
    return options.scanToolResults;
  }
  return options.detect || {};
}

/** Tells apart the tools each wrapped client declares to a shared policy. */
let policySources = 0;

interface McpClient {
  listTools(...args: unknown[]): unknown;
  callTool(...args: unknown[]): unknown;
  readResource(...args: unknown[]): unknown;
  getPrompt(...args: unknown[]): unknown;
}

/** The SDK's `client.experimental.tasks`, which calls tools as tasks. */
interface McpTasks {
  callToolStream(...args: unknown[]): AsyncIterable<unknown>;
  requestStream(...args: unknown[]): AsyncIterable<unknown>;
  getTaskResult(...args: unknown[]): unknown;
}

/** The other ways the SDK's `Client` sends a request, which older SDKs lack. */
interface SdkClient extends McpClient {
  request?(...args: unknown[]): unknown;
  requestStream?(...args: unknown[]): AsyncIterable<unknown>;
  getTaskResult?(...args: unknown[]): unknown;
  experimental?: { tasks?: McpTasks };
}

/** How what a request returns, or the error it fails with, is checked. */
interface Check {
  /** Checks a result, and returns it or what to return in its place. */
  result(value: unknown): Promise<unknown>;
  /** Checks an error, and throws in its place when it blocks. */
  error(value: unknown): Promise<void>;
}

/**
 * Wraps an MCP `Client` so tool lists are checked for tool poisoning and
 * flagged tools are dropped, and tool results, resources, and prompts are
 * checked for injection like tool results in the other wrappers. With
 * `policy`, every tool call must pass a tool policy from
 * `createToolPolicy()` before the server is called. A tool call is checked
 * the same whichever way the SDK sends it: `callTool`, `request`, or
 * `experimental.tasks.callToolStream`, which throws while it is read.
 *
 * @example
 * ```ts
 * import { Client } from "@modelcontextprotocol/sdk/client/index.js";
 * import { shieldMcpClient } from "@zeroleaks/shield/mcp";
 *
 * const client = shieldMcpClient(new Client({ name: "agent", version: "1.0.0" }));
 * await client.connect(transport);
 * const { tools } = await client.listTools(); // flagged tools left out
 * ```
 */
export function shieldMcpClient<
  // Method syntax makes the parameter check bivariant, so the SDK's
  // `Client`, whose methods take specific param types, satisfies it.
  T extends McpClient,
>(client: T, options: ShieldMcpOptions = {}): T {
  const { input } = createShield(options);
  const scanOptions = toolScanOptions(options);
  const mode = options.onFlaggedTools ?? "drop";
  const pins: ToolPins | null =
    options.pins === false ? null : (options.pins ?? Object.create(null));
  const blockCalls =
    scanOptions !== null && (options.blockFlaggedToolCalls ?? mode !== "warn");
  const argumentOptions =
    options.scanArguments === false ? null : (options.scanArguments ?? {});
  /** Tools the latest list flagged, by name, with their scan results. */
  const flaggedByName = new Map<string, ToolScanResult>();
  const { policy } = options;
  policySources += 1;
  const policySource = `mcp:${policySources}`;
  /** The tools declared to the policy: every page of the latest list. */
  let declared: ToolDefinition[] = [];
  const blocking = (options.onDetection ?? "block") === "block";

  /** Declares the tools a list returned; a page fetched with a cursor adds to the pages before it. */
  const declare = (tools: unknown[], params: unknown): void => {
    if (!policy) {
      return;
    }
    const page = tools.filter(isRecord) as ToolDefinition[];
    const nextPage = isRecord(params) && typeof params.cursor === "string";
    declared = nextPage ? [...declared, ...page] : page;
    policy.declareTools(declared, policySource);
  };

  /** Adds the pins `pinTools` made for `tools` to `pins` itself, which the caller may have passed in. */
  const pinNew = (tools: ToolDefinition[]): void => {
    if (!pins) {
      return;
    }
    const added = pinTools(tools, pins);
    for (const name of Object.keys(added)) {
      if (Object.getOwnPropertyDescriptor(pins, name) === undefined) {
        Object.defineProperty(pins, name, {
          value: added[name],
          enumerable: true,
          writable: true,
          configurable: true,
        });
      }
    }
  };

  /** Detection's result for the text of a result, reported to `onInjectionDetected`; `undefined` when tool results aren't checked. */
  const inspect = async (
    result: unknown,
    textOf: (result: unknown) => string
  ): Promise<DetectResult | undefined> =>
    input.tool ? await input.inspect(textOf(result), "tool") : undefined;

  /**
   * Checks the text `textOf` finds in a result, or in the error a call failed
   * with, as a tool result: tells `record` whether detection flagged it, and
   * throws `InjectionDetectedError` when the detection blocks.
   */
  const review = async (
    value: unknown,
    textOf: (value: unknown) => string,
    record: (flagged: boolean) => void
  ): Promise<void> => {
    const detection = await inspect(value, textOf);
    record(Boolean(detection?.detected));
    if (detection?.detected && blocking) {
      throw new InjectionDetectedError(
        detection.risk,
        detection.matches.map((m) => m.category),
        "tool"
      );
    }
  };

  /** Records a resource or prompt with an injection as untrusted content from `source`. */
  const untrusted =
    (source: string) =>
    (flagged: boolean): void => {
      if (flagged) {
        policy?.recordUntrusted(source);
      }
    };

  /** Checks result and error text for injection, as `review` does. */
  const injectionCheck = (
    textOf: (value: unknown) => string,
    record: (flagged: boolean) => void
  ): Check => ({
    async result(value) {
      await review(value, textOf, record);
      return value;
    },
    error: (value) => review(value, errorText, record),
  });

  /** Checks a tool list for poisoned tools, and returns it with the flagged ones dropped. */
  const screenTools = async (
    result: unknown,
    params: unknown
  ): Promise<unknown> => {
    if (!(isRecord(result) && Array.isArray(result.tools))) {
      return result;
    }
    const tools: unknown[] = result.tools;
    if (!scanOptions) {
      declare(tools, params);
      return result;
    }
    const scans = (
      await scanToolsAsync(tools as ToolDefinition[], {
        ...scanOptions,
        pins: pins ?? undefined,
      })
    ).tools;
    const flagged = new Set<number>();
    const clean: ToolDefinition[] = [];
    for (const [i, scan] of scans.entries()) {
      if (isFlagged(scan)) {
        flagged.add(i);
        flaggedByName.set(scan.name, scan);
        options.onToolFlagged?.(tools[i] as ToolDefinition, scan);
      } else {
        flaggedByName.delete(scan.name);
        clean.push(tools[i] as ToolDefinition);
      }
    }
    pinNew(clean);
    declare(mode === "warn" ? tools : clean, params);
    if (flagged.size === 0 || mode === "warn") {
      return result;
    }
    if (mode === "throw") {
      throw flaggedToolsError(scans.filter((_, i) => flagged.has(i)));
    }
    return { ...result, tools: tools.filter((_, i) => !flagged.has(i)) };
  };

  /**
   * Checks a tool call before the server is called: the tool is not
   * flagged, its arguments carry no credential or exfiltration link, and
   * the policy allows it. Returns the check for its result.
   */
  const beginCall = async (params: unknown): Promise<Check> => {
    const call = isRecord(params) ? params : undefined;
    const name = typeof call?.name === "string" ? call.name : "";
    const flaggedScan = blockCalls ? flaggedByName.get(name) : undefined;
    if (flaggedScan) {
      throw flaggedToolsError([flaggedScan]);
    }
    if (argumentOptions && call && call.arguments !== undefined) {
      const scan = scanOutputText(
        argumentText(call.arguments),
        argumentOptions
      );
      const high = scan.findings.filter((f) => atLeast(f.severity, "high"));
      if (high.length > 0) {
        throw new OutputBlockedError(high);
      }
    }
    if (policy) {
      const decision = await policy.checkAsync({
        name,
        arguments: call?.arguments,
        source: policySource,
      });
      if (!decision.allowed) {
        throw new ToolPolicyError(decision);
      }
    }
    return injectionCheck(toolResultText, (flagged) =>
      policy?.recordResult(name, { flagged })
    );
  };

  /**
   * Runs what a request with MCP method `method` must pass before it is
   * sent, and returns the check for what it returns, or `undefined` for a
   * method that has none. Every way the client sends a request comes here,
   * so a tool call is checked the same whichever one sends it.
   */
  const begin = async (
    method: unknown,
    params: unknown
  ): Promise<Check | undefined> => {
    switch (method) {
      case "tools/list":
        return {
          result: (value) => screenTools(value, params),
          error: async () => undefined,
        };
      case "tools/call":
        return await beginCall(params);
      case "resources/read":
        return injectionCheck(resourcesText, untrusted("readResource"));
      case "prompts/get":
        return injectionCheck(promptText, untrusted("getPrompt"));
      case "tasks/result":
        return injectionCheck(toolResultText, untrusted("getTaskResult"));
      default:
        return;
    }
  };

  /**
   * Sends a request with MCP method `method` through `send`, and checks what
   * it returns or the error it fails with, which a server writes as freely
   * as a result. An error is rethrown as it is unless its injection blocks.
   */
  const sent = async (
    method: unknown,
    params: unknown,
    send: () => unknown
  ): Promise<unknown> => {
    const check = await begin(method, params);
    if (!check) {
      return await send();
    }
    let result: unknown;
    try {
      result = await send();
    } catch (error) {
      await check.error(error);
      throw error;
    }
    return await check.result(result);
  };

  /** Streams the messages of a request opened with `open`, checking its result or error as `sent` does. */
  async function* streamed(
    method: unknown,
    params: unknown,
    open: () => AsyncIterable<unknown>
  ): AsyncGenerator<unknown> {
    const check = await begin(method, params);
    for await (const message of open()) {
      if (check && isRecord(message) && message.type === "result") {
        const result = await check.result(message.result);
        yield result === message.result ? message : { ...message, result };
        continue;
      }
      if (check && isRecord(message) && message.type === "error") {
        await check.error(message.error);
      }
      yield message;
    }
  }

  const overrides: Record<string, unknown> = {
    listTools: (...args: unknown[]) =>
      sent("tools/list", args[0], () => client.listTools(...args)),
    callTool: (...args: unknown[]) =>
      sent("tools/call", args[0], () => client.callTool(...args)),
    readResource: (...args: unknown[]) =>
      sent("resources/read", args[0], () => client.readResource(...args)),
    getPrompt: (...args: unknown[]) =>
      sent("prompts/get", args[0], () => client.getPrompt(...args)),
  };
  const sdk: SdkClient = client;
  if (sdk.request) {
    const request = sdk.request.bind(client);
    overrides.request = (...args: unknown[]) =>
      sent(methodOf(args[0]), paramsOf(args[0]), () => request(...args));
  }
  if (sdk.requestStream) {
    const requestStream = sdk.requestStream.bind(client);
    overrides.requestStream = (...args: unknown[]) =>
      streamed(methodOf(args[0]), paramsOf(args[0]), () =>
        requestStream(...args)
      );
  }
  if (sdk.getTaskResult) {
    const getTaskResult = sdk.getTaskResult.bind(client);
    overrides.getTaskResult = (...args: unknown[]) =>
      sent("tasks/result", args[0], () => getTaskResult(...args));
  }
  const tasks = sdk.experimental?.tasks;
  if (sdk.experimental && tasks) {
    overrides.experimental = withOverrides(sdk.experimental, {
      tasks: withOverrides(tasks, {
        callToolStream: (...args: unknown[]) =>
          streamed("tools/call", args[0], () => tasks.callToolStream(...args)),
        requestStream: (...args: unknown[]) =>
          streamed(methodOf(args[0]), paramsOf(args[0]), () =>
            tasks.requestStream(...args)
          ),
        getTaskResult: (...args: unknown[]) =>
          sent("tasks/result", args[0], () => tasks.getTaskResult(...args)),
      }),
    });
  }
  return withOverrides(client, overrides);
}
