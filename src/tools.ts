import {
  type DetectOptions,
  type DetectResult,
  detect,
  slowDetection,
} from "./detect";
import { ShieldError } from "./errors";

/**
 * A tool definition in any of the common shapes: MCP (`inputSchema`),
 * OpenAI (`{ type: "function", function: { ... } }` or a Responses API
 * function tool), Anthropic (`input_schema`), or an AI SDK tool
 * (`parameters` or `inputSchema`).
 */
export type ToolDefinition = Record<string, unknown>;

export interface ToolScanResult {
  name: string;
  /** Detection over the tool's descriptions and schema text. */
  result: DetectResult;
  /**
   * Other problems with the definition itself. `truncated` means it is
   * longer than detection reads (`maxInputLength`), so the rest is unchecked.
   */
  issues: Array<
    | "duplicate_name"
    | "hidden_characters_in_name"
    | "oversized_description"
    | "changed_since_pinned"
    | "nested_too_deep"
    | "truncated"
  >;
}

/**
 * Tool definitions as pinned by `pinTools()`: each tool's name mapped to the
 * canonical JSON of what the model reads from it. Plain data, so it can be
 * saved with `JSON.stringify` and loaded back with `JSON.parse`.
 */
export type ToolPins = Record<string, string>;

export interface ScanToolsResult {
  /** True when any tool was flagged or has an issue. */
  flagged: boolean;
  tools: ToolScanResult[];
}

export interface ScanToolsOptions extends DetectOptions {
  /** Descriptions longer than this are reported as `oversized_description`. Default 4000. */
  maxDescriptionLength?: number;
  /**
   * Definitions pinned earlier with `pinTools()`. A tool whose pinned
   * definition differs from the one it has now is reported as
   * `changed_since_pinned`: a server can change a tool after it was
   * reviewed or approved (a "rug pull").
   */
  pins?: ToolPins;
}

const RE_INVISIBLE_IN_NAME =
  // biome-ignore lint/suspicious/noMisleadingCharacterClass: the class lists combining and format characters on purpose, to find or strip them.
  /[\u00ad\u034f\u061c\u115f\u1160\u180e\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]|\udb40[\udc00-\udc7f]/;
const RE_NON_ASCII = /[^\x20-\x7e]/;
/** How deep a definition is read; one nested deeper is flagged as `nested_too_deep`. */
const MAX_DEFINITION_DEPTH = 128;
/** What `canonical` writes in place of a value nested deeper than the definition is read. */
const TOO_DEEP = "[nested too deep]";
const RE_IDENTIFIER_SEPARATORS = /[_\-.]+/g;
const RE_CAMEL_BOUNDARY = /([a-z0-9])([A-Z])/g;

/** "readSshKey_first" -> "read Ssh Key first": models read names as words. */
export function identifierWords(name: string): string {
  return name
    .replace(RE_CAMEL_BOUNDARY, "$1 $2")
    .replace(RE_IDENTIFIER_SEPARATORS, " ");
}
/**
 * Keys of JSON Schema and of tool definitions. Every other key, such as a
 * parameter name or a vendor extension, is read as words, since the model
 * reads names too.
 */
const STRUCTURE_KEYS = new Set([
  "$anchor",
  "$comment",
  "$defs",
  "$dynamicAnchor",
  "$dynamicRef",
  "$id",
  "$ref",
  "$schema",
  "additionalItems",
  "additionalProperties",
  "allOf",
  "annotations",
  "anyOf",
  "const",
  "contains",
  "contentEncoding",
  "contentMediaType",
  "contentSchema",
  "default",
  "definitions",
  "dependencies",
  "dependentRequired",
  "dependentSchemas",
  "deprecated",
  "description",
  "destructiveHint",
  "else",
  "enum",
  "example",
  "examples",
  "exclusiveMaximum",
  "exclusiveMinimum",
  "execution",
  "format",
  "function",
  "icons",
  "idempotentHint",
  "if",
  "input_schema",
  "inputSchema",
  "items",
  "maxContains",
  "maximum",
  "maxItems",
  "maxLength",
  "maxProperties",
  "mimeType",
  "minContains",
  "minimum",
  "minItems",
  "minLength",
  "minProperties",
  "multipleOf",
  "name",
  "not",
  "nullable",
  "oneOf",
  "openWorldHint",
  "outputSchema",
  "parameters",
  "pattern",
  "patternProperties",
  "prefixItems",
  "properties",
  "propertyNames",
  "readOnly",
  "readOnlyHint",
  "required",
  "sizes",
  "src",
  "strict",
  "taskSupport",
  "then",
  "title",
  "type",
  "unevaluatedItems",
  "unevaluatedProperties",
  "uniqueItems",
  "writeOnly",
]);

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

/** The part of a definition that holds name, description, and schema. */
export function unwrap(tool: ToolDefinition): Record<string, unknown> {
  return asRecord(tool.function) ?? tool;
}

/** The input schema of an unwrapped definition, in whichever field its shape keeps it. */
export function schemaOf(def: Record<string, unknown>): unknown {
  return (
    def.inputSchema ??
    def.input_schema ??
    def.parameters ??
    asRecord(def.annotations)?.inputSchema
  );
}

/**
 * `value` with object keys sorted at every depth, so equal definitions
 * serialize equally. A value nested deeper than a definition is read
 * becomes `TOO_DEEP`.
 */
function canonical(value: unknown, depth = 0): unknown {
  if (value === null || typeof value !== "object") {
    return value;
  }
  if (depth > MAX_DEFINITION_DEPTH) {
    return TOO_DEEP;
  }
  if (Array.isArray(value)) {
    return value.map((item) => canonical(item, depth + 1));
  }
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(value).sort()) {
    Object.defineProperty(out, key, {
      value: canonical((value as Record<string, unknown>)[key], depth + 1),
      enumerable: true,
    });
  }
  return out;
}

/** What a tool's definition pins: everything the model reads from it, keys sorted. */
function fingerprint(def: Record<string, unknown>): string {
  return JSON.stringify(
    canonical({
      annotations: def.annotations ?? null,
      description: def.description ?? null,
      inputSchema: schemaOf(def) ?? null,
      outputSchema: def.outputSchema ?? null,
      title: def.title ?? null,
    })
  );
}

function hasPin(pins: ToolPins, name: string): boolean {
  return Object.getOwnPropertyDescriptor(pins, name) !== undefined;
}

/**
 * Pins tool definitions, so a later `scanTools(tools, { pins })` reports any
 * tool whose definition changed. Returns a copy of `pins` with every tool in
 * `tools` that was not pinned yet added; tools already pinned keep their pin,
 * so a changed tool stays reported until you remove its entry.
 */
export function pinTools(
  tools: ToolDefinition[],
  pins: ToolPins = {}
): ToolPins {
  const out: ToolPins = {};
  for (const name of Object.keys(pins)) {
    Object.defineProperty(out, name, {
      value: pins[name],
      enumerable: true,
      writable: true,
      configurable: true,
    });
  }
  for (const tool of tools) {
    const def = unwrap(tool);
    const name = String(def.name ?? "");
    if (!hasPin(out, name)) {
      Object.defineProperty(out, name, {
        value: fingerprint(def),
        enumerable: true,
        writable: true,
        configurable: true,
      });
    }
  }
  return out;
}

interface DefinitionText {
  text: string;
  /** Whether part of the definition is nested too deep to read. */
  tooDeep: boolean;
}

/**
 * The text a model reads from a tool definition: every string in it, at any
 * depth, and every key that isn't one of `STRUCTURE_KEYS`, as words. The
 * tool's name is left out; `scanTools` checks it for hidden characters.
 */
function definitionText(tool: ToolDefinition): DefinitionText {
  const out: string[] = [];
  let tooDeep = false;
  const seen = new Set<object>();
  const stack: Array<{ value: unknown; depth: number }> = [];
  const def = unwrap(tool);
  for (const key of Object.keys(def).reverse()) {
    if (key !== "name") {
      stack.push({ value: def[key], depth: 1 });
    }
  }
  for (let item = stack.pop(); item; item = stack.pop()) {
    const { value, depth } = item;
    if (typeof value === "string") {
      if (value) {
        out.push(value);
      }
      continue;
    }
    if (value === null || typeof value !== "object" || seen.has(value)) {
      continue;
    }
    if (depth > MAX_DEFINITION_DEPTH) {
      tooDeep = true;
      continue;
    }
    seen.add(value);
    const entries: [string | null, unknown][] = Array.isArray(value)
      ? value.map((child) => [null, child])
      : Object.entries(value);
    for (let i = entries.length - 1; i >= 0; i--) {
      const [key, child] = entries[i];
      stack.push({ value: child, depth: depth + 1 });
      if (key !== null && !STRUCTURE_KEYS.has(key)) {
        stack.push({ value: identifierWords(key), depth });
      }
    }
  }
  return { text: out.join("\n"), tooDeep };
}

/**
 * Checks tool descriptions and schemas locally, along with duplicate names,
 * hidden name characters, and changed definitions. For asynchronous detectors,
 * use scanToolsAsync.
 */
export function scanTools(
  tools: ToolDefinition[],
  options: ScanToolsOptions = {}
): ScanToolsResult {
  if (options.secondaryDetector || options.escalate) {
    throw new ShieldError(
      "Use scanToolsAsync for hosted or asynchronous detection.",
      "ASYNC_DETECTION_REQUIRES_AWAIT"
    );
  }
  const { maxDescriptionLength = 4000, pins, ...detectOptions } = options;
  const counts = new Map<string, number>();
  for (const tool of tools) {
    const name = String(unwrap(tool).name ?? "");
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }

  const results = tools.map((tool): ToolScanResult => {
    const def = unwrap(tool);
    const name = String(def.name ?? "");
    const description =
      typeof def.description === "string" ? def.description : "";
    const { text, tooDeep } = definitionText(tool);
    const result = detect(text, detectOptions);

    const issues: ToolScanResult["issues"] = [];
    if ((counts.get(name) ?? 0) > 1) {
      issues.push("duplicate_name");
    }
    if (RE_INVISIBLE_IN_NAME.test(name) || RE_NON_ASCII.test(name)) {
      issues.push("hidden_characters_in_name");
    }
    if (description.length > maxDescriptionLength) {
      issues.push("oversized_description");
    }
    if (tooDeep) {
      issues.push("nested_too_deep");
    }
    if (result.truncated) {
      issues.push("truncated");
    }
    if (pins && hasPin(pins, name) && pins[name] !== fingerprint(def)) {
      issues.push("changed_since_pinned");
    }
    return { name, result, issues };
  });

  return {
    flagged: results.some((r) => r.result.detected || r.issues.length > 0),
    tools: results,
  };
}

/** Checks tool definitions with hosted, model, or other asynchronous detectors. */
export async function scanToolsAsync(
  tools: ToolDefinition[],
  options: ScanToolsOptions = {}
): Promise<ScanToolsResult> {
  if (!(options.secondaryDetector || options.escalate)) {
    return scanTools(tools, options);
  }
  const scans = scanTools(tools, {
    ...options,
    secondaryDetector: undefined,
    escalate: undefined,
  });
  for (const [index, scan] of scans.tools.entries()) {
    const { text } = definitionText(tools[index]);
    if (text) {
      const pending = slowDetection(text, scan.result, options);
      if (pending) {
        scan.result = await pending;
      }
    }
  }
  return {
    flagged: scans.tools.some(
      (scan) => scan.result.detected || scan.issues.length > 0
    ),
    tools: scans.tools,
  };
}
