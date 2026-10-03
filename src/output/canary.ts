import { ShieldError } from "../errors";
import type { OutputFinding } from "./types";
import { compareFindings, lastItem } from "./util";

export interface CanaryOptions {
  /** Letters and digits, upper-cased into the token. Default "zl". */
  prefix?: string;
  /** Random bytes, rendered as hex. 4 to 32; default 8. */
  bytes?: number;
}

export type CanaryMatchKind =
  | "verbatim"
  | "obfuscated"
  | "reversed"
  | "base64"
  | "hex"
  | "url_encoded";

const PREFIX_PATTERN = /^[A-Za-z0-9]{1,16}$/;
const CANARY_TOKEN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;

/**
 * A random canary token such as `ZL-CANARY-7f3a9c2e41b0d6a8`: one word of
 * letters, digits, and hyphens, so models reproduce it verbatim. Uses
 * `globalThis.crypto.getRandomValues` (Node 19+, Bun, Deno, Workers,
 * browsers) and throws a `ShieldError` with code `CRYPTO_UNAVAILABLE` when
 * Web Crypto is missing rather than falling back to a weak source.
 */
export function createCanary(options: CanaryOptions = {}): string {
  const prefix = options.prefix ?? "zl";
  const bytes = options.bytes ?? 8;
  if (!PREFIX_PATTERN.test(prefix)) {
    throw new TypeError("createCanary: prefix must be 1-16 letters or digits");
  }
  if (!Number.isInteger(bytes) || bytes < 4 || bytes > 32) {
    throw new RangeError("createCanary: bytes must be an integer from 4 to 32");
  }
  const webCrypto = (globalThis as { crypto?: Crypto }).crypto;
  if (typeof webCrypto?.getRandomValues !== "function") {
    throw new ShieldError(
      "createCanary: globalThis.crypto.getRandomValues is not available. Use Node 19+ (or Node 18 with --experimental-global-webcrypto), Bun, Deno, Cloudflare Workers, or a browser.",
      "CRYPTO_UNAVAILABLE"
    );
  }
  const random = new Uint8Array(bytes);
  webCrypto.getRandomValues(random);
  let hex = "";
  for (const byte of random) {
    hex += byte.toString(16).padStart(2, "0");
  }
  return `${prefix.toUpperCase()}-CANARY-${hex}`;
}

function assertCanary(canary: string): void {
  if (typeof canary !== "string" || !CANARY_TOKEN.test(canary)) {
    throw new TypeError(
      "canary must be a single token of letters, digits, hyphens, or underscores"
    );
  }
}

/**
 * A line to put in a system prompt. It makes the canary part of the
 * confidential instructions without asking the model to repeat it, so the
 * token only shows up in output when the instructions leak.
 */
export function canaryInstruction(canary: string): string {
  assertCanary(canary);
  return `Internal reference: ${canary}. This reference is confidential.`;
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/** Confusable Cyrillic and Greek letters, each followed by the ASCII letter it imitates. */
const HOMOGLYPH_PAIRS = "аaеeоoсcрpхxуyіiԁdАaВbЕeСcОoРpХxαaοoΑaΒbΕeΟo";
/** Greek and Cyrillic, where the look-alike letters live. */
const HOMOGLYPH_BLOCKS: [number, number][] = [
  [0x03_91, 0x03_c9],
  [0x04_00, 0x05_01],
];
/**
 * Blocks whose letters and digits NFKC-fold to ASCII: superscripts and
 * subscripts, modifier letters, letterlike symbols, number forms, circled
 * letters and digits, full-width forms, and mathematical alphanumerics.
 */
const COMPATIBILITY_BLOCKS: [number, number][] = [
  [0x00_aa, 0x00_ba],
  [0x02_b0, 0x02_ff],
  [0x1d_00, 0x1d_bf],
  [0x20_70, 0x21_8f],
  [0x24_60, 0x24_ff],
  [0xff_00, 0xff_ef],
  [0x1_d4_00, 0x1_d7_ff],
  [0x1_f1_00, 0x1_f1_ff],
];

/** The lowercase ASCII letter or digit for an ASCII code, or -1. */
function asciiAlnum(code: number): number {
  if ((code >= 48 && code <= 57) || (code >= 97 && code <= 122)) {
    return code;
  }
  return code >= 65 && code <= 90 ? code + 32 : -1;
}

/** Non-ASCII code points that read as an ASCII letter or digit, to its lowercase code. */
const FOLDS = new Map<number, number>();
for (const [from, to] of COMPATIBILITY_BLOCKS) {
  for (let code = from; code <= to; code++) {
    const folded = String.fromCodePoint(code).normalize("NFKC");
    const ascii = folded.length === 1 ? asciiAlnum(folded.charCodeAt(0)) : -1;
    if (ascii !== -1) {
      FOLDS.set(code, ascii);
    }
  }
}
for (let i = 0; i < HOMOGLYPH_PAIRS.length; i += 2) {
  FOLDS.set(
    HOMOGLYPH_PAIRS.charCodeAt(i),
    HOMOGLYPH_PAIRS.charCodeAt(i + 1)
  );
}

/** Maps a code point to a lowercase ASCII letter or digit, or -1 to drop it. */
function foldCodePoint(code: number): number {
  return code < 128 ? asciiAlnum(code) : (FOLDS.get(code) ?? -1);
}

function isHexDigit(code: number): boolean {
  return (
    (code >= 48 && code <= 57) ||
    (code >= 97 && code <= 102) ||
    (code >= 65 && code <= 70)
  );
}

/**
 * A normalized copy of the text (built with native string operations) plus
 * a lazily computed map from each view character back to its offset.
 */
interface View {
  chars: string;
  /** Offset in the original text of view character i. */
  offsetOf: (index: number) => number;
}

type Keep = (text: string, i: number) => number;

/**
 * Returns 1 to keep text[i], 0 to drop it, 2 to drop it and the next
 * character, 3 to keep it together with the next (a surrogate pair).
 */
function keepAlnum(text: string, i: number): number {
  const code = text.codePointAt(i) ?? 0;
  const kept = foldCodePoint(code) !== -1;
  if (code > 0xff_ff) {
    return kept ? 3 : 2;
  }
  return kept ? 1 : 0;
}

function keepHex(text: string, i: number): number {
  const code = text.charCodeAt(i);
  const next = text.charCodeAt(i + 1);
  if (code === 48 && (next === 120 || next === 88)) {
    return 2;
  }
  return isHexDigit(code) ? 1 : 0;
}

function lazyView(text: string, chars: string, keep: Keep): View {
  let positions: Int32Array | null = null;
  const build = (): Int32Array => {
    const out = new Int32Array(chars.length + 1);
    let size = 0;
    let i = 0;
    while (i < text.length && size < chars.length) {
      const verdict = keep(text, i);
      if (verdict === 1 || verdict === 3) {
        out[size] = i;
        size++;
      }
      i += verdict >= 2 ? 2 : 1;
    }
    out[size] = text.length;
    return out;
  };
  return {
    chars,
    offsetOf: (index) => {
      positions ??= build();
      return positions[index];
    },
  };
}

const FOLDABLE_RANGES = [...HOMOGLYPH_BLOCKS, ...COMPATIBILITY_BLOCKS]
  .map(([from, to]) => `\\u{${from.toString(16)}}-\\u{${to.toString(16)}}`)
  .join("");
const NOT_ALNUM_CANDIDATE = new RegExp(
  `[^A-Za-z0-9${FOLDABLE_RANGES}]+`,
  "gu"
);
const NON_ASCII = /[\u0080-￿]/;
const NON_ASCII_GLOBAL = /[\u0080-\u{10ffff}]/gu;
const NOT_HEX = /0[xX]|[^0-9A-Fa-f]+/g;

function foldString(char: string): string {
  const folded = foldCodePoint(char.codePointAt(0) ?? 0);
  return folded === -1 ? "" : String.fromCharCode(folded);
}

/**
 * Letters and digits only, lowercased, with full-width, circled, math, and
 * other compatibility forms (NFKC) and look-alike letters folded to ASCII.
 */
function alnumView(text: string): View {
  let chars = text.replace(NOT_ALNUM_CANDIDATE, "");
  if (NON_ASCII.test(chars)) {
    chars = chars.replace(NON_ASCII_GLOBAL, foldString);
  }
  return lazyView(text, chars.toLowerCase(), keepAlnum);
}

/** Hex digits only ("0x" prefixes dropped), lowercased. */
function hexView(text: string): View {
  return lazyView(text, text.replace(NOT_HEX, "").toLowerCase(), keepHex);
}

function reverse(value: string): string {
  return value.split("").reverse().join("");
}

function toHex(value: string): string {
  let out = "";
  for (let i = 0; i < value.length; i++) {
    out += value.charCodeAt(i).toString(16).padStart(2, "0");
  }
  return out;
}

const NON_ALNUM = /[^a-z0-9]/g;
const SEGMENT_SPLIT = /[-_]/;
const BASE64_SPECIAL = /[+/]/;
const PLUS = /\+/g;
const SLASH = /\//g;
const NUL = String.fromCharCode(0);

interface Needle {
  value: string;
  kind: CanaryMatchKind;
  confidence: number;
  /** When the matched text is exactly this string, report it as verbatim. */
  exact?: string;
}

interface Base64Needle {
  core: string;
  /** Base64 characters before the core that still encode part of the target. */
  lead: number;
}

interface CanaryNeedles {
  canary: string;
  /** Cheap prefilters: the normalized views are only built when these match. */
  alnumProbe: RegExp;
  hexProbe: RegExp;
  percentProbe: RegExp;
  lower: string;
  /** The random tail on its own (e.g. the hex of `ZL-CANARY-<hex>`), when long enough. */
  random: string | null;
  alnum: Needle[];
  hex: Needle[];
  base64: Base64Needle[];
  preview: string;
}

function randomPart(canary: string): string | null {
  const segments = canary.split(SEGMENT_SPLIT);
  const tail = segments.length > 1 ? (lastItem(segments) ?? "") : "";
  return tail.length >= 8 ? tail.toLowerCase() : null;
}

/**
 * Base64 of `value` at each of the three byte alignments it can have inside
 * a longer blob, trimmed to the characters that depend only on `value`, in
 * standard and URL-safe alphabets. Finding one of these cores finds the
 * value without decoding every base64 run in the text.
 */
function base64Cores(value: string): Base64Needle[] {
  const needles: Base64Needle[] = [];
  for (let offset = 0; offset < 3; offset++) {
    const encoded = btoa(NUL.repeat(offset) + value);
    const lead = Math.ceil((8 * offset) / 6);
    const end = Math.floor((8 * (offset + value.length)) / 6);
    const core = encoded.slice(lead, end);
    needles.push({ core, lead });
    if (BASE64_SPECIAL.test(core)) {
      needles.push({ core: core.replace(PLUS, "-").replace(SLASH, "_"), lead });
    }
  }
  return needles;
}

const PROBE_LENGTH = 4;
const REGEX_SPECIAL = /[\\^$.*+?()[\]{}|-]/g;

/** `codes` (sorted) as character class ranges, such as `\u{2460}-\u{2473}`. */
function classRanges(codes: number[]): string {
  const ranges: string[] = [];
  for (let i = 0; i < codes.length; i++) {
    const from = codes[i];
    while (i + 1 < codes.length && codes[i + 1] === codes[i] + 1) {
      i++;
    }
    const to = codes[i];
    const start = `\\u{${from.toString(16)}}`;
    ranges.push(from === to ? start : `${start}-\\u{${to.toString(16)}}`);
  }
  return ranges.join("");
}

/**
 * Any run of characters that don't fold to a letter or digit. A probe
 * letter can never also be read as a separator, so each run is scanned once.
 */
const SEPARATORS = `[^A-Za-z0-9${classRanges(
  [...FOLDS.keys()].sort((a, b) => a - b)
)}]*`;

/** Every character that folds to `code` (ASCII case, compatibility forms, look-alikes), as a class. */
function variantsOf(code: number): string {
  const variants = new Set([String.fromCharCode(code)]);
  if (code >= 97 && code <= 122) {
    variants.add(String.fromCharCode(code - 32));
  }
  for (const [from, to] of FOLDS) {
    if (to === code) {
      variants.add(String.fromCodePoint(from));
    }
  }
  const members = Array.from(variants)
    .map((v) => v.replace(REGEX_SPECIAL, "\\$&"))
    .join("");
  return `[${members}]`;
}

/**
 * Matches the first few characters of any alnum needle, each written as
 * any of its variants, with separators between them.
 */
function alnumProbe(values: string[]): RegExp {
  const alternatives = values.map((value) =>
    Array.from(value.slice(0, PROBE_LENGTH), (c) =>
      variantsOf(c.charCodeAt(0))
    ).join(SEPARATORS)
  );
  return new RegExp(alternatives.join("|"), "u");
}

/** Matches the first characters of each target, each written plainly or as a %XX escape. */
function percentProbe(values: string[]): RegExp {
  const alternatives = values.map((value) =>
    Array.from(value.slice(0, PROBE_LENGTH), (c) => {
      const variants = new Set([c, c.toLowerCase(), c.toUpperCase()]);
      const escapes = Array.from(variants, (v) => {
        const hex = v.charCodeAt(0).toString(16).padStart(2, "0");
        return `%${variantsOf(hex.charCodeAt(0))}${variantsOf(hex.charCodeAt(1))}`;
      });
      const plain = Array.from(variants, (v) =>
        v.replace(REGEX_SPECIAL, "\\$&")
      );
      return `(?:[${plain.join("")}]|${escapes.join("|")})`;
    }).join("")
  );
  return new RegExp(alternatives.join("|"), "u");
}

/** `[aA]` for a letter, the digit itself otherwise. */
function asciiCaseless(code: number): string {
  const char = String.fromCharCode(code);
  const upper = char.toUpperCase();
  return upper === char ? char : `[${char}${upper}]`;
}

/**
 * Matches the first bytes of any hex needle, allowing separators and
 * 0x / \\x / % escapes between bytes. Only ASCII hex digits count, as in
 * the hex view, so a digit is never also a separator.
 */
function hexProbe(values: string[]): RegExp {
  const alternatives = values.map((value) => {
    const bytes: string[] = [];
    for (let i = 0; i + 1 < Math.min(value.length, PROBE_LENGTH * 2); i += 2) {
      bytes.push(
        asciiCaseless(value.charCodeAt(i)) +
          asciiCaseless(value.charCodeAt(i + 1))
      );
    }
    return bytes.join("(?:[^0-9A-Fa-f]|0[xX])*");
  });
  return new RegExp(alternatives.join("|"), "u");
}

function buildNeedles(canary: string): CanaryNeedles {
  const lower = canary.toLowerCase();
  const core = lower.replace(NON_ALNUM, "");
  if (core.length < 6) {
    throw new RangeError(
      "canary is too short to detect reliably (min 6 letters or digits)"
    );
  }
  const random = randomPart(canary);
  const randomOriginal = random
    ? canary.slice(canary.length - random.length)
    : "";
  const alnum: Needle[] = [
    { value: core, kind: "obfuscated", confidence: 0.95 },
    { value: reverse(core), kind: "reversed", confidence: 0.9 },
  ];
  const hex: Needle[] = Array.from(
    new Set([canary, lower, canary.toUpperCase()]),
    (target) => ({ value: toHex(target), kind: "hex", confidence: 0.95 })
  );
  const base64Targets = new Set([canary, lower, canary.toUpperCase()]);
  if (random && random !== core) {
    alnum.push(
      {
        value: random,
        kind: "obfuscated",
        confidence: 0.9,
        exact: randomOriginal,
      },
      { value: reverse(random), kind: "reversed", confidence: 0.85 }
    );
    for (const target of new Set([random, random.toUpperCase()])) {
      hex.push({ value: toHex(target), kind: "hex", confidence: 0.9 });
    }
    base64Targets.add(random);
  }
  const visible = random
    ? canary.slice(0, canary.length - random.length + 4)
    : canary.slice(0, 4);
  return {
    canary,
    alnumProbe: alnumProbe(alnum.map((n) => n.value)),
    percentProbe: percentProbe(random ? [lower, random] : [lower]),
    hexProbe: hexProbe(hex.map((n) => n.value)),
    lower,
    random,
    alnum,
    hex,
    base64: Array.from(base64Targets).flatMap(base64Cores),
    preview: `${visible}…`,
  };
}

const NEEDLE_CACHE = new Map<string, CanaryNeedles>();
const NEEDLE_CACHE_SIZE = 32;

function needlesFor(canary: string): CanaryNeedles {
  const cached = NEEDLE_CACHE.get(canary);
  if (cached) {
    return cached;
  }
  const needles = buildNeedles(canary);
  if (NEEDLE_CACHE.size >= NEEDLE_CACHE_SIZE) {
    const oldest = NEEDLE_CACHE.keys().next().value;
    if (oldest !== undefined) {
      NEEDLE_CACHE.delete(oldest);
    }
  }
  NEEDLE_CACHE.set(canary, needles);
  return needles;
}

interface Collector {
  findings: OutputFinding[];
  preview: string;
  /** One flag per text offset already claimed by a finding. */
  covered: Uint8Array;
}

function overlapsExisting(
  collector: Collector,
  start: number,
  end: number
): boolean {
  for (let i = start; i < end; i++) {
    if (collector.covered[i] === 1) {
      return true;
    }
  }
  return false;
}

function add(
  collector: Collector,
  start: number,
  end: number,
  kind: CanaryMatchKind,
  confidence: number
): void {
  if (overlapsExisting(collector, start, end)) {
    return;
  }
  collector.covered.fill(1, start, end);
  collector.findings.push({
    type: "canary",
    kind,
    start,
    end,
    severity: "critical",
    confidence,
    preview: `${collector.preview} (${kind})`,
  });
}

function findVerbatim(
  text: string,
  needles: CanaryNeedles,
  collector: Collector
): void {
  const needle = needles.canary;
  let index = text.indexOf(needle);
  while (index !== -1) {
    add(collector, index, index + needle.length, "verbatim", 1);
    index = text.indexOf(needle, index + needle.length);
  }
}

const PERCENT_ESCAPE = /%[0-9A-Fa-f]{2}/y;

/** Percent-decodes %XX escapes (one byte each), keeping each char's start offset. */
function percentView(text: string): { chars: string; positions: Int32Array } {
  const positions = new Int32Array(text.length + 1);
  let chars = "";
  let size = 0;
  let i = 0;
  while (i < text.length) {
    PERCENT_ESCAPE.lastIndex = i;
    positions[size] = i;
    size++;
    if (text.charCodeAt(i) === 37 && PERCENT_ESCAPE.test(text)) {
      chars += String.fromCharCode(
        Number.parseInt(text.slice(i + 1, i + 3), 16)
      );
      i += 3;
    } else {
      chars += text.charAt(i);
      i++;
    }
  }
  positions[size] = text.length;
  return { chars: chars.toLowerCase(), positions };
}

function findPercentEncoded(
  text: string,
  needles: CanaryNeedles,
  collector: Collector
): void {
  if (!(text.includes("%") && needles.percentProbe.test(text))) {
    return;
  }
  const view = percentView(text);
  const targets = needles.random
    ? [needles.lower, needles.random]
    : [needles.lower];
  for (const needle of targets) {
    let index = view.chars.indexOf(needle);
    while (index !== -1) {
      const start = view.positions[index];
      const end = view.positions[index + needle.length];
      if (text.slice(start, end).includes("%")) {
        add(collector, start, end, "url_encoded", 0.95);
      }
      index = view.chars.indexOf(needle, index + needle.length);
    }
  }
}

const HEX_PREFIXES = ["0x", "0X", "\\x", "%"];

/** Widens a hex match to include the escape prefix of its first byte. */
function hexStart(text: string, start: number): number {
  for (const prefix of HEX_PREFIXES) {
    if (text.startsWith(prefix, start - prefix.length)) {
      return start - prefix.length;
    }
  }
  return start;
}

function addViewMatch(
  text: string,
  view: View,
  needle: Needle,
  index: number,
  collector: Collector
): void {
  const length = needle.value.length;
  const first = view.offsetOf(index);
  const last = view.offsetOf(index + length - 1);
  const end = last + ((text.codePointAt(last) ?? 0) > 0xff_ff ? 2 : 1);
  const start = needle.kind === "hex" ? hexStart(text, first) : first;
  // Separators between characters are fine; whole words in between are not.
  if (end - start > length * 8) {
    return;
  }
  const matched = text.slice(start, end);
  if (needle.exact !== undefined && matched === needle.exact) {
    add(collector, start, end, "verbatim", 0.95);
    return;
  }
  const encoded = needle.kind === "hex" && matched.includes("%");
  add(
    collector,
    start,
    end,
    encoded ? "url_encoded" : needle.kind,
    needle.confidence
  );
}

function searchView(
  text: string,
  view: View,
  needles: Needle[],
  collector: Collector
): void {
  for (const needle of needles) {
    let index = view.chars.indexOf(needle.value);
    while (index !== -1) {
      addViewMatch(text, view, needle, index, collector);
      index = view.chars.indexOf(needle.value, index + needle.value.length);
    }
  }
}

const LINE_BREAKS = /\r?\n/g;
/** True when some line ends with 16+ base64 characters and the next line continues them (MIME-style wrapping). */
function hasWrappedBase64(text: string): boolean {
  let newline = text.indexOf("\n");
  let checked = 0;
  while (newline !== -1 && checked < 10_000) {
    checked++;
    const lineEnd = text.charCodeAt(newline - 1) === 13 ? newline - 1 : newline;
    let run = 0;
    while (
      run < 16 &&
      lineEnd - run > 0 &&
      isBase64Char(text.charCodeAt(lineEnd - run - 1))
    ) {
      run++;
    }
    if (run === 16 && isBase64Char(text.charCodeAt(newline + 1))) {
      return true;
    }
    newline = text.indexOf("\n", newline + 1);
  }
  return false;
}

function isBase64Char(code: number): boolean {
  return (
    (code >= 48 && code <= 57) ||
    (code >= 65 && code <= 90) ||
    (code >= 97 && code <= 122) ||
    code === 43 ||
    code === 47 ||
    code === 45 ||
    code === 95 ||
    code === 61
  );
}

/** Widens a base64 core match to the whole 4-character groups that encode the canary. */
function base64Span(
  text: string,
  at: number,
  needle: Base64Needle
): [number, number] {
  let start = at;
  while (
    start > 0 &&
    at - start < needle.lead &&
    isBase64Char(text.charCodeAt(start - 1))
  ) {
    start--;
  }
  let end = at + needle.core.length;
  const limit = end + 3;
  while (
    end < text.length &&
    end < limit &&
    isBase64Char(text.charCodeAt(end))
  ) {
    end++;
  }
  return [start, end];
}

function searchBase64(
  text: string,
  needles: Base64Needle[],
  report: (at: number, needle: Base64Needle) => void
): void {
  for (const needle of needles) {
    let at = text.indexOf(needle.core);
    while (at !== -1) {
      report(at, needle);
      at = text.indexOf(needle.core, at + needle.core.length);
    }
  }
}

/**
 * Finds base64 of the canary, alone or inside a longer blob, by searching for
 * precomputed cores. Line-wrapped (MIME-style) base64 is searched with line
 * breaks removed.
 */
function findBase64(
  text: string,
  needles: CanaryNeedles,
  collector: Collector
): void {
  searchBase64(text, needles.base64, (at, needle) => {
    const [start, end] = base64Span(text, at, needle);
    add(collector, start, end, "base64", 0.95);
  });
  if (!hasWrappedBase64(text)) {
    return;
  }
  const joined = lazyView(text, text.replace(LINE_BREAKS, ""), (t, i) => {
    const code = t.charCodeAt(i);
    return code === 10 || code === 13 ? 0 : 1;
  });
  searchBase64(joined.chars, needles.base64, (at, needle) => {
    const start = joined.offsetOf(at);
    const end = joined.offsetOf(at + needle.core.length - 1) + 1;
    if (text.slice(start, end).includes("\n")) {
      add(collector, start, end, "base64", 0.9);
    }
  });
}

/**
 * Finds a canary in model output, including when the model transformed it:
 * changed case, spaced or hyphen-stripped letters, zero-width or other
 * characters inserted, full-width or look-alike letters, reversed, hex,
 * URL-encoded, or base64 (alone or inside a longer blob). The random tail of
 * a `createCanary` token counts on its own, so dropping the prefix does not
 * hide it.
 */
export function findCanary(text: string, canary: string): OutputFinding[] {
  assertCanary(canary);
  const needles = needlesFor(canary);
  if (typeof text !== "string" || text.length < 6) {
    return [];
  }
  const collector: Collector = {
    findings: [],
    preview: needles.preview,
    covered: new Uint8Array(text.length),
  };
  findVerbatim(text, needles, collector);
  findPercentEncoded(text, needles, collector);
  if (needles.alnumProbe.test(text)) {
    searchView(text, alnumView(text), needles.alnum, collector);
  }
  if (needles.hexProbe.test(text)) {
    searchView(text, hexView(text), needles.hex, collector);
  }
  findBase64(text, needles, collector);
  return collector.findings.sort(compareFindings);
}
