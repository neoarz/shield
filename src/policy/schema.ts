/**
 * A focused JSON Schema validator for tool call arguments. It knows `type`
 * (with OpenAPI's `nullable`), `required`, `properties`,
 * `patternProperties`, `additionalProperties`, `propertyNames`,
 * `minProperties`, `maxProperties`, `dependentRequired`,
 * `dependentSchemas`, `dependencies`, `prefixItems`, `items`,
 * `additionalItems`, `minItems`, `maxItems`, `uniqueItems`, `enum`,
 * `const`, `minLength`, `maxLength`, `pattern`, `minimum`, `maximum`,
 * `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`, `anyOf`, `oneOf`,
 * `allOf`, `not`, `if`, `then`, `else`, and `$ref` to a JSON pointer in the
 * same schema, such as `#/$defs/name`. Every other keyword, such as
 * `unevaluatedProperties` or `format`, is ignored.
 *
 * Schemas can come from an MCP server, so validation is bounded: `$ref` is
 * followed at most 32 deep, arguments are read at most 64 levels deep,
 * schemas inside schemas at most 512, one validation takes at most 100,000
 * steps, and patterns are run as `planPattern()` allows. Violations say
 * where and which keyword, never the value.
 */

import { PATTERN_WORK, type PatternPlan, planPattern } from "./pattern";

export interface SchemaViolation {
  /**
   * Where in the arguments: `$` is the arguments themselves, `$.to` a
   * property, `$.items[0]` an array item. A key the schema doesn't declare
   * comes from the arguments, so it is written as `*`.
   */
  path: string;
  /** The keyword that failed, such as `type` or `required`. */
  keyword: string;
  /** What is wrong, without the value. */
  message: string;
}

const MAX_STEPS = 100_000;
const MAX_DEPTH = 64;
/** How many schemas one check may be inside: properties, items, `$ref`, and combinators all count. */
const MAX_NESTING = 512;
const MAX_REF_DEPTH = 32;
const MAX_KEY_IN_PATH = 64;
/** Pattern work, as length ** degree, that counts as one step. */
const PATTERN_WORK_PER_STEP = 1024;
const RE_IDENTIFIER = /^[A-Za-z_$][\w$]*$/;
const RE_INDEX = /^(0|[1-9]\d*)$/;
const RE_POINTER_SLASH = /~1/g;
const RE_POINTER_TILDE = /~0/g;

type JsonRecord = Record<string, unknown>;

interface Run {
  root: unknown;
  /** Shared by every quiet run of one validation. `nesting` counts the schemas the check is inside now. */
  budget: { steps: number; exhausted: boolean; nesting: number };
  /** Violations found so far, or `null` in a quiet run, which only needs to know whether the value is valid. */
  out: SchemaViolation[] | null;
  max: number;
  /** Each schema's `patternProperties` as entries, read once per validation. */
  patterns: WeakMap<JsonRecord, [string, unknown][]>;
}

function asRecord(value: unknown): JsonRecord | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : undefined;
}

function hasOwn(record: object, key: string): boolean {
  return Object.getOwnPropertyDescriptor(record, key) !== undefined;
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function propertyPath(path: string, key: string): string {
  const shown =
    key.length > MAX_KEY_IN_PATH ? `${key.slice(0, MAX_KEY_IN_PATH)}...` : key;
  return RE_IDENTIFIER.test(shown)
    ? `${path}.${shown}`
    : `${path}[${JSON.stringify(shown)}]`;
}

/** Whether to stop: `max` violations were found, or the budget is spent. */
function full(run: Run): boolean {
  return run.out === null || run.budget.exhausted || run.out.length >= run.max;
}

function report(
  run: Run,
  path: string,
  keyword: string,
  message: string
): void {
  if (full(run) || !run.out) {
    return;
  }
  const seen = run.out.some((v) => v.path === path && v.keyword === keyword);
  if (!seen) {
    run.out.push({ path, keyword, message });
  }
}

/** Counts `steps` against the budget, and says whether it is spent. */
function spend(run: Run, steps = 1): boolean {
  run.budget.steps += steps;
  if (run.budget.steps > MAX_STEPS) {
    run.budget.exhausted = true;
  }
  return run.budget.exhausted;
}

/** Code points, which JSON Schema lengths count, rather than UTF-16 units. */
function codePoints(text: string): number {
  let count = 0;
  for (let i = 0; i < text.length; i++) {
    const unit = text.charCodeAt(i);
    if (unit >= 0xd8_00 && unit <= 0xdb_ff && i + 1 < text.length) {
      const next = text.charCodeAt(i + 1);
      if (next >= 0xdc_00 && next <= 0xdf_ff) {
        i += 1;
      }
    }
    count += 1;
  }
  return count;
}

/**
 * A string that JSON-equal values share, with object keys sorted, or
 * `undefined` for a value nested deeper than `MAX_DEPTH`.
 */
function jsonKey(value: unknown, depth = 0): string | undefined {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value) ?? "undefined";
  }
  if (depth > MAX_DEPTH) {
    return;
  }
  const parts: string[] = [];
  if (Array.isArray(value)) {
    for (const item of value) {
      const key = jsonKey(item, depth + 1);
      if (key === undefined) {
        return;
      }
      parts.push(key);
    }
    return `[${parts.join(",")}]`;
  }
  const record = value as JsonRecord;
  for (const name of Object.keys(record).sort()) {
    const key = jsonKey(record[name], depth + 1);
    if (key === undefined) {
      return;
    }
    parts.push(`${JSON.stringify(name)}:${key}`);
  }
  return `{${parts.join(",")}}`;
}

/** Whether no two of `items` are equal, or `undefined` when one is nested too deep to compare. */
function allDifferent(items: unknown[]): boolean | undefined {
  const keys = new Set<string>();
  for (const item of items) {
    const key = jsonKey(item);
    if (key === undefined) {
      return;
    }
    if (keys.has(key)) {
      return false;
    }
    keys.add(key);
  }
  return true;
}

function jsonEqual(a: unknown, b: unknown, depth = 0): boolean {
  if (a === b) {
    return true;
  }
  if (depth > MAX_DEPTH || typeof a !== "object" || typeof b !== "object") {
    return false;
  }
  if (a === null || b === null || Array.isArray(a) !== Array.isArray(b)) {
    return false;
  }
  if (Array.isArray(a)) {
    const other = b as unknown[];
    return (
      a.length === other.length &&
      a.every((item, i) => jsonEqual(item, other[i], depth + 1))
    );
  }
  const left = a as JsonRecord;
  const right = b as JsonRecord;
  const keys = Object.keys(left);
  return (
    keys.length === Object.keys(right).length &&
    keys.every(
      (key) => hasOwn(right, key) && jsonEqual(left[key], right[key], depth + 1)
    )
  );
}

/** Whether `value` has JSON type `type`, or `undefined` for a type name this validator doesn't know. */
function hasType(value: unknown, type: string): boolean | undefined {
  switch (type.toLowerCase()) {
    case "null":
      return value === null;
    case "boolean":
      return typeof value === "boolean";
    case "string":
      return typeof value === "string";
    case "number":
      return isNumber(value);
    case "integer":
      return isNumber(value) && Number.isInteger(value);
    case "array":
      return Array.isArray(value);
    case "object":
      return asRecord(value) !== undefined;
    default:
      return;
  }
}

/** The type names `value` doesn't match, or `null` when it matches or the keyword doesn't apply. */
function typeMismatch(schema: JsonRecord, value: unknown): string[] | null {
  let types: string[] = [];
  if (typeof schema.type === "string") {
    types = [schema.type];
  } else if (Array.isArray(schema.type)) {
    types = schema.type.filter((t): t is string => typeof t === "string");
  }
  let known = false;
  for (const type of types) {
    const match = hasType(value, type);
    if (match) {
      return null;
    }
    known ||= match === false;
  }
  if (!known || (value === null && schema.nullable === true)) {
    return null;
  }
  return types.map((t) => t.toLowerCase());
}

/** The schema a local JSON pointer (`#`, `#/$defs/name`) points to, or `undefined`. */
function resolveRef(root: unknown, ref: string): unknown {
  if (!ref.startsWith("#")) {
    return;
  }
  let pointer: string;
  try {
    pointer = decodeURIComponent(ref.slice(1));
  } catch {
    return;
  }
  if (pointer === "") {
    return root;
  }
  if (!pointer.startsWith("/")) {
    return;
  }
  let node: unknown = root;
  for (const raw of pointer.slice(1).split("/")) {
    const key = raw
      .replace(RE_POINTER_SLASH, "/")
      .replace(RE_POINTER_TILDE, "~");
    if (Array.isArray(node)) {
      node = RE_INDEX.test(key) ? node[Number(key)] : undefined;
    } else {
      const record = asRecord(node);
      node = record && hasOwn(record, key) ? record[key] : undefined;
    }
    if (node === undefined) {
      return;
    }
  }
  return node;
}

function checkString(
  run: Run,
  schema: JsonRecord,
  value: string,
  path: string
): boolean {
  let valid = true;
  const needsLength = isNumber(schema.minLength) || isNumber(schema.maxLength);
  const length = needsLength ? codePoints(value) : 0;
  if (isNumber(schema.minLength) && length < schema.minLength) {
    valid = false;
    report(
      run,
      path,
      "minLength",
      `must be at least ${schema.minLength} characters`
    );
  }
  if (isNumber(schema.maxLength) && length > schema.maxLength) {
    valid = false;
    report(
      run,
      path,
      "maxLength",
      `must be at most ${schema.maxLength} characters`
    );
  }
  if (typeof schema.pattern === "string") {
    valid = checkPattern(run, schema.pattern, value, path) && valid;
  }
  return valid;
}

/** Runs a planned pattern on `value`, no longer than its `maxLength`, counting the work against the budget. */
function runPattern(run: Run, plan: PatternPlan, value: string): boolean {
  const work = Math.min(value.length ** Math.max(plan.degree, 1), PATTERN_WORK);
  if (spend(run, Math.ceil(work / PATTERN_WORK_PER_STEP))) {
    return false;
  }
  return plan.regex.test(value);
}

function checkPattern(
  run: Run,
  pattern: string,
  value: string,
  path: string
): boolean {
  const plan = planPattern(pattern);
  if (!plan) {
    return true;
  }
  if (value.length > plan.maxLength) {
    report(
      run,
      path,
      "pattern",
      `is too long to check against its pattern (over ${plan.maxLength} characters)`
    );
    return false;
  }
  if (runPattern(run, plan, value)) {
    return true;
  }
  report(run, path, "pattern", "must match the pattern");
  return false;
}

/** The `minimum` and `exclusiveMinimum` that `value` is below, as keyword and message. */
function belowMinimum(schema: JsonRecord, value: number): [string, string][] {
  const out: [string, string][] = [];
  const { minimum, exclusiveMinimum } = schema;
  if (isNumber(minimum)) {
    const exclusive = exclusiveMinimum === true;
    if (exclusive ? value <= minimum : value < minimum) {
      out.push(["minimum", `must be ${exclusive ? ">" : ">="} ${minimum}`]);
    }
  }
  if (isNumber(exclusiveMinimum) && value <= exclusiveMinimum) {
    out.push(["exclusiveMinimum", `must be > ${exclusiveMinimum}`]);
  }
  return out;
}

/** The `maximum` and `exclusiveMaximum` that `value` is above, as keyword and message. */
function aboveMaximum(schema: JsonRecord, value: number): [string, string][] {
  const out: [string, string][] = [];
  const { maximum, exclusiveMaximum } = schema;
  if (isNumber(maximum)) {
    const exclusive = exclusiveMaximum === true;
    if (exclusive ? value >= maximum : value > maximum) {
      out.push(["maximum", `must be ${exclusive ? "<" : "<="} ${maximum}`]);
    }
  }
  if (isNumber(exclusiveMaximum) && value >= exclusiveMaximum) {
    out.push(["exclusiveMaximum", `must be < ${exclusiveMaximum}`]);
  }
  return out;
}

/**
 * Whether `value` is a multiple of `divisor`: exactly for integers, and
 * within rounding error otherwise, so 0.3 is a multiple of 0.1.
 */
function isMultiple(value: number, divisor: number): boolean {
  if (Number.isInteger(value) && Number.isInteger(divisor)) {
    return value % divisor === 0;
  }
  const quotient = value / divisor;
  const error = Math.abs(quotient - Math.round(quotient));
  return (
    Number.isFinite(quotient) &&
    error <= 4 * Number.EPSILON * Math.max(1, Math.abs(quotient))
  );
}

function checkNumber(
  run: Run,
  schema: JsonRecord,
  value: number,
  path: string
): boolean {
  const failed = [
    ...belowMinimum(schema, value),
    ...aboveMaximum(schema, value),
  ];
  const { multipleOf } = schema;
  if (
    isNumber(multipleOf) &&
    multipleOf > 0 &&
    !isMultiple(value, multipleOf)
  ) {
    failed.push(["multipleOf", `must be a multiple of ${multipleOf}`]);
  }
  for (const [keyword, message] of failed) {
    report(run, path, keyword, message);
  }
  return failed.length === 0;
}

/**
 * The schemas of an array's first items, from `prefixItems` or an `items`
 * array, and of the items after them, from `items` or `additionalItems`.
 */
function itemSchemas(schema: JsonRecord): { tuple: unknown[]; rest: unknown } {
  const { prefixItems, items, additionalItems } = schema;
  if (Array.isArray(prefixItems)) {
    return {
      tuple: prefixItems,
      rest: Array.isArray(items) ? undefined : items,
    };
  }
  if (Array.isArray(items)) {
    return { tuple: items, rest: additionalItems };
  }
  return { tuple: [], rest: items };
}

function checkArray(
  run: Run,
  schema: JsonRecord,
  value: unknown[],
  path: string,
  depth: number,
  refs: number
): boolean {
  let valid = true;
  if (isNumber(schema.minItems) && value.length < schema.minItems) {
    valid = false;
    report(
      run,
      path,
      "minItems",
      `must have at least ${schema.minItems} items`
    );
  }
  if (isNumber(schema.maxItems) && value.length > schema.maxItems) {
    valid = false;
    report(run, path, "maxItems", `must have at most ${schema.maxItems} items`);
  }
  if (schema.uniqueItems === true) {
    spend(run, value.length);
    const unique = allDifferent(value);
    if (unique !== true) {
      valid = false;
      report(
        run,
        path,
        "uniqueItems",
        unique === false
          ? "must not have equal items"
          : "has items nested too deep to compare"
      );
    }
  }
  const { tuple, rest } = itemSchemas(schema);
  for (const [i, item] of value.entries()) {
    const itemSchema = i < tuple.length ? tuple[i] : rest;
    if (itemSchema === undefined) {
      break;
    }
    valid =
      check(run, itemSchema, item, `${path}[${i}]`, depth + 1, refs) && valid;
    if (full(run) && !valid) {
      return false;
    }
  }
  return valid;
}

function checkRequired(
  run: Run,
  schema: JsonRecord,
  value: JsonRecord,
  path: string
): boolean {
  if (!Array.isArray(schema.required)) {
    return true;
  }
  let valid = true;
  for (const name of schema.required) {
    const present = hasOwn(value, name) && value[name] !== undefined;
    if (typeof name === "string" && !present) {
      valid = false;
      report(run, propertyPath(path, name), "required", "is required");
    }
  }
  return valid;
}

/**
 * Whether `key` matches `pattern`: `false` for a pattern that isn't run,
 * and `undefined` for a key too long to run it on.
 */
function keyMatches(
  run: Run,
  pattern: string,
  key: string
): boolean | undefined {
  const plan = planPattern(pattern);
  if (!plan) {
    return false;
  }
  if (key.length > plan.maxLength) {
    return;
  }
  return runPattern(run, plan, key);
}

/**
 * Checks one property against `properties` and every `patternProperties`
 * pattern its key matches, or against `additionalProperties` when there is
 * none. Only a key `properties` declares is shown in paths.
 */
function checkProperty(
  run: Run,
  schema: JsonRecord,
  value: JsonRecord,
  key: string,
  patterns: [string, unknown][],
  path: string,
  depth: number,
  refs: number
): boolean {
  const properties = asRecord(schema.properties) ?? {};
  let named = hasOwn(properties, key);
  const at = named ? propertyPath(path, key) : `${path}.*`;
  let valid = named
    ? check(run, properties[key], value[key], at, depth + 1, refs)
    : true;
  for (const [pattern, patternSchema] of patterns) {
    if (spend(run)) {
      return false;
    }
    const match = keyMatches(run, pattern, key);
    if (match === undefined) {
      report(
        run,
        `${path}.*`,
        "patternProperties",
        "has a key too long to check against its patterns"
      );
      return false;
    }
    if (match) {
      named = true;
      valid =
        check(run, patternSchema, value[key], at, depth + 1, refs) && valid;
    }
  }
  if (named) {
    return valid;
  }
  const additional = schema.additionalProperties;
  if (additional === false) {
    report(run, at, "additionalProperties", "is not allowed");
    return false;
  }
  if (additional === undefined || additional === true) {
    return true;
  }
  return check(run, additional, value[key], at, depth + 1, refs);
}

function patternEntries(run: Run, schema: JsonRecord): [string, unknown][] {
  let entries = run.patterns.get(schema);
  if (!entries) {
    entries = Object.entries(asRecord(schema.patternProperties) ?? {});
    run.patterns.set(schema, entries);
  }
  return entries;
}

function checkPropertyCount(
  run: Run,
  schema: JsonRecord,
  keys: string[],
  path: string
): boolean {
  let valid = true;
  if (isNumber(schema.minProperties) && keys.length < schema.minProperties) {
    valid = false;
    report(
      run,
      path,
      "minProperties",
      `must have at least ${schema.minProperties} properties`
    );
  }
  if (isNumber(schema.maxProperties) && keys.length > schema.maxProperties) {
    valid = false;
    report(
      run,
      path,
      "maxProperties",
      `must have at most ${schema.maxProperties} properties`
    );
  }
  return valid;
}

const DEPENDENCY_KEYWORDS = [
  "dependentRequired",
  "dependentSchemas",
  "dependencies",
] as const;

/**
 * Checks `dependentRequired`, `dependentSchemas`, and `dependencies`, which
 * holds either: what else the object must have once it has a property.
 */
function checkDependencies(
  run: Run,
  schema: JsonRecord,
  value: JsonRecord,
  keys: string[],
  path: string,
  depth: number,
  refs: number
): boolean {
  let valid = true;
  const present = new Set(keys);
  for (const keyword of DEPENDENCY_KEYWORDS) {
    const dependencies = asRecord(schema[keyword]) ?? {};
    for (const key of keys) {
      if (!hasOwn(dependencies, key)) {
        continue;
      }
      const dependency = dependencies[key];
      if (Array.isArray(dependency)) {
        if (spend(run, dependency.length)) {
          return false;
        }
        for (const name of dependency) {
          if (typeof name === "string" && !present.has(name)) {
            valid = false;
            report(run, propertyPath(path, name), keyword, "is required");
          }
        }
      } else if (keyword !== "dependentRequired") {
        valid = check(run, dependency, value, path, depth, refs) && valid;
      }
      if (full(run) && !valid) {
        return false;
      }
    }
  }
  return valid;
}

function checkObject(
  run: Run,
  schema: JsonRecord,
  value: JsonRecord,
  path: string,
  depth: number,
  refs: number
): boolean {
  const keys = Object.keys(value).filter((key) => value[key] !== undefined);
  const patterns = patternEntries(run, schema);
  let valid = checkRequired(run, schema, value, path);
  valid = checkPropertyCount(run, schema, keys, path) && valid;
  for (const key of keys) {
    valid =
      checkProperty(run, schema, value, key, patterns, path, depth, refs) &&
      valid;
    if (
      schema.propertyNames !== undefined &&
      !matches(run, schema.propertyNames, key, depth + 1, refs)
    ) {
      valid = false;
      report(
        run,
        `${path}.*`,
        "propertyNames",
        "has a key that doesn't match propertyNames"
      );
    }
    if (full(run) && !valid) {
      return false;
    }
  }
  return (
    checkDependencies(run, schema, value, keys, path, depth, refs) && valid
  );
}

/** Runs `check` without reporting, sharing the budget. */
function matches(
  run: Run,
  schema: unknown,
  value: unknown,
  depth: number,
  refs: number
): boolean {
  return check({ ...run, out: null }, schema, value, "$", depth, refs);
}

/** How many of `schemas` `value` matches, counting no further than `limit`. */
function countMatches(
  run: Run,
  schemas: unknown[],
  value: unknown,
  depth: number,
  refs: number,
  limit: number
): number {
  let count = 0;
  for (const schema of schemas) {
    if (count >= limit) {
      break;
    }
    if (matches(run, schema, value, depth, refs)) {
      count += 1;
    }
  }
  return count;
}

function checkAllOf(
  run: Run,
  schemas: unknown[],
  value: unknown,
  path: string,
  depth: number,
  refs: number
): boolean {
  let valid = true;
  for (const schema of schemas) {
    valid = check(run, schema, value, path, depth, refs) && valid;
    if (full(run) && !valid) {
      return false;
    }
  }
  return valid;
}

function checkCombinators(
  run: Run,
  schema: JsonRecord,
  value: unknown,
  path: string,
  depth: number,
  refs: number
): boolean {
  const { allOf, anyOf, oneOf, not } = schema;
  let valid = Array.isArray(allOf)
    ? checkAllOf(run, allOf, value, path, depth, refs)
    : true;
  if (
    Array.isArray(anyOf) &&
    countMatches(run, anyOf, value, depth, refs, 1) === 0
  ) {
    valid = false;
    report(run, path, "anyOf", "does not match any of the allowed schemas");
  }
  const oneOfMatches = Array.isArray(oneOf)
    ? countMatches(run, oneOf, value, depth, refs, 2)
    : 1;
  if (oneOfMatches !== 1) {
    valid = false;
    report(
      run,
      path,
      "oneOf",
      oneOfMatches === 0
        ? "does not match any of the allowed schemas"
        : "matches more than one of the schemas where exactly one must match"
    );
  }
  if (not !== undefined && matches(run, not, value, depth, refs)) {
    valid = false;
    report(run, path, "not", "matches a schema it must not match");
  }
  if (schema.if !== undefined) {
    const branch = matches(run, schema.if, value, depth, refs)
      ? schema.then
      : schema.else;
    if (branch !== undefined) {
      valid = check(run, branch, value, path, depth, refs) && valid;
    }
  }
  return valid;
}

function checkValue(
  run: Run,
  schema: JsonRecord,
  value: unknown,
  path: string,
  depth: number,
  refs: number
): boolean {
  let valid = true;
  const mismatch = typeMismatch(schema, value);
  if (mismatch) {
    report(run, path, "type", `must be ${mismatch.join(" or ")}`);
    return false;
  }
  if (Array.isArray(schema.enum)) {
    spend(run, schema.enum.length);
    if (!schema.enum.some((option) => jsonEqual(option, value))) {
      valid = false;
      report(run, path, "enum", "must be one of the allowed values");
    }
  }
  if (schema.const !== undefined && !jsonEqual(schema.const, value)) {
    valid = false;
    report(run, path, "const", "must be the allowed value");
  }
  if (typeof value === "string") {
    valid = checkString(run, schema, value, path) && valid;
  } else if (isNumber(value)) {
    valid = checkNumber(run, schema, value, path) && valid;
  } else if (Array.isArray(value)) {
    valid = checkArray(run, schema, value, path, depth, refs) && valid;
  } else if (asRecord(value)) {
    valid =
      checkObject(run, schema, value as JsonRecord, path, depth, refs) && valid;
  }
  return valid;
}

function check(
  run: Run,
  schema: unknown,
  value: unknown,
  path: string,
  depth: number,
  refs: number
): boolean {
  if (run.budget.exhausted || spend(run)) {
    return false;
  }
  if (schema === false) {
    report(run, path, "false", "is not allowed");
    return false;
  }
  const record = asRecord(schema);
  if (!record) {
    return true;
  }
  if (depth > MAX_DEPTH) {
    report(run, path, "depth", `is nested deeper than ${MAX_DEPTH} levels`);
    return false;
  }
  if (run.budget.nesting >= MAX_NESTING) {
    run.budget.exhausted = true;
    return false;
  }
  run.budget.nesting += 1;
  const valid = checkRecord(run, record, value, path, depth, refs);
  run.budget.nesting -= 1;
  return valid;
}

function checkRecord(
  run: Run,
  record: JsonRecord,
  value: unknown,
  path: string,
  depth: number,
  refs: number
): boolean {
  let valid = true;
  if (typeof record.$ref === "string") {
    if (refs >= MAX_REF_DEPTH) {
      report(run, path, "$ref", `follows $ref more than ${MAX_REF_DEPTH} deep`);
      return false;
    }
    const target = resolveRef(run.root, record.$ref);
    if (target !== undefined) {
      valid = check(run, target, value, path, depth, refs + 1);
      if (full(run) && !valid) {
        return false;
      }
    }
  }
  valid = checkValue(run, record, value, path, depth, refs) && valid;
  if (full(run) && !valid) {
    return false;
  }
  return checkCombinators(run, record, value, path, depth, refs) && valid;
}

/**
 * Validates `value` against a JSON schema and returns up to `max`
 * violations, none when it is valid. When validation runs out of steps, or
 * finds schemas nested more than 512 deep, the only violation is
 * `{ path: "$", keyword: "budget" }`.
 */
export function validateSchema(
  schema: unknown,
  value: unknown,
  max = 5
): SchemaViolation[] {
  const run: Run = {
    root: schema,
    budget: { steps: 0, exhausted: false, nesting: 0 },
    out: [],
    max,
    patterns: new WeakMap(),
  };
  const valid = check(run, schema, value, "$", 0, 0);
  if (run.budget.exhausted) {
    return [
      { path: "$", keyword: "budget", message: "is too complex to validate" },
    ];
  }
  if (!valid && run.out?.length === 0) {
    return [
      { path: "$", keyword: "schema", message: "does not match the schema" },
    ];
  }
  return run.out ?? [];
}
