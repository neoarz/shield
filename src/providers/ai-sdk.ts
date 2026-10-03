import { ShieldError } from "../errors";
import { harden } from "../harden";
import {
  createShield,
  type InputGuard,
  jsonText,
  type OutputGuard,
  requestSignal,
  type Shield,
  type ShieldProviderOptions,
} from "./guard";
import {
  type ChunkedSanitizer,
  type ChunkResult,
  chunkString,
  createChunkedSanitizer,
} from "./utils";

export interface ShieldAISdkOptions extends ShieldProviderOptions {}

interface MessagePart {
  type: string;
  text?: string;
}
interface Message {
  role: string;
  content: string | MessagePart[];
}
/** AI SDK 6 also accepts system messages, alone or in an array, as `system`. */
interface SystemMessage {
  role: "system";
  content: string;
}
interface AISdkParams {
  system?: string | SystemMessage | Array<SystemMessage | MessagePart>;
  /** AI SDK 5 and later also accept an array of messages here. */
  prompt?: string | Array<MessagePart | Message>;
  messages?: Message[];
  [key: string]: unknown;
}

function extractMessageText(content: string | MessagePart[]): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .filter(
      (p): p is MessagePart & { text: string } =>
        p.type === "text" && typeof p.text === "string"
    )
    .map((p) => p.text)
    .join(" ");
}

function isMessage(item: MessagePart | Message): item is Message {
  return "role" in item;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/**
 * A system prompt hardened, or with `harden: false`, with the canary
 * planted in it.
 */
function hardenText(text: string, shield: Shield): string {
  if (shield.harden) {
    return harden(text, shield.harden);
  }
  const instruction = shield.plant(text);
  return instruction ? `${text}\n\n${instruction}` : text;
}

function hardenSystemMessage(
  message: SystemMessage,
  shield: Shield
): SystemMessage {
  return message.content
    ? { ...message, content: hardenText(message.content, shield) }
    : message;
}

/**
 * Hardens `system` in the shape it came in. Each system message is hardened
 * on its own, as the language model middleware does, and an array of text
 * parts becomes a single hardened text part. With `harden: false`, the
 * canary is planted instead, in one more text part after an array of them.
 */
function hardenSystem(
  system: NonNullable<AISdkParams["system"]>,
  shield: Shield
): AISdkParams["system"] {
  if (typeof system === "string") {
    return hardenText(system, shield);
  }
  if (!Array.isArray(system)) {
    return hardenSystemMessage(system, shield);
  }
  const parts = system.filter((item): item is MessagePart => !isMessage(item));
  if (parts.length < system.length) {
    return system.map((item) =>
      isMessage(item) ? hardenSystemMessage(item, shield) : item
    );
  }
  const text = extractMessageText(parts);
  if (shield.harden) {
    return text ? [{ type: "text", text: harden(text, shield.harden) }] : system;
  }
  const instruction = shield.plant(text);
  return instruction
    ? [...system, { type: "text", text: instruction }]
    : system;
}

/** Every message, whether in `messages` or in `prompt`. */
function paramMessages(params: AISdkParams): Message[] {
  const promptItems = Array.isArray(params.prompt) ? params.prompt : [];
  return [...(params.messages ?? []), ...promptItems.filter(isMessage)];
}

/** `prompt` and every user message, whether in `messages` or in `prompt`. */
function extractUserInputs(params: AISdkParams): string[] {
  const { prompt } = params;
  const promptItems = Array.isArray(prompt) ? prompt : [];
  return [
    typeof prompt === "string"
      ? prompt
      : extractMessageText(
          promptItems.filter((p): p is MessagePart => !isMessage(p))
        ),
    ...paramMessages(params)
      .filter((m) => m.role === "user")
      .map((m) => extractMessageText(m.content)),
  ];
}

/**
 * Text of a `tool-result` part. AI SDK 5 and later carry it in `output`,
 * typed as text, JSON, an error, or content parts; AI SDK 4 in `result`.
 */
function toolResultText(part: Record<string, unknown>): string {
  const { output } = part;
  if (isRecord(output) && typeof output.type === "string") {
    const { type, value } = output;
    if (type === "text" || type === "error-text") {
      return typeof value === "string" ? value : "";
    }
    if (type === "json" || type === "error-json") {
      return jsonText(value);
    }
    if (type === "content" && Array.isArray(value)) {
      return value
        .filter(
          (p): p is { type: string; text: string } =>
            isRecord(p) && p.type === "text" && typeof p.text === "string"
        )
        .map((p) => p.text)
        .join("\n");
    }
    return "";
  }
  if ("result" in part) {
    return typeof part.result === "string"
      ? part.result
      : jsonText(part.result);
  }
  return "";
}

/** Text of every tool result in `messages`, in any role. */
function toolResultTexts(messages: Array<{ content: unknown }>): string[] {
  const texts: string[] = [];
  for (const message of messages) {
    if (!Array.isArray(message.content)) {
      continue;
    }
    for (const part of message.content) {
      if (isRecord(part) && part.type === "tool-result") {
        texts.push(toolResultText(part));
      }
    }
  }
  return texts;
}

function checkParams(params: AISdkParams, input: InputGuard): void {
  for (const text of extractUserInputs(params)) {
    input.checkSync(text, "user");
  }
  if (input.tool) {
    for (const text of toolResultTexts(paramMessages(params))) {
      input.checkSync(text, "tool");
    }
  }
}

async function checkParamsAsync(
  params: AISdkParams,
  input: InputGuard
): Promise<void> {
  const signal = requestSignal(params);
  for (const text of extractUserInputs(params)) {
    await input.check(text, "user", signal);
  }
  if (input.tool) {
    for (const text of toolResultTexts(paramMessages(params))) {
      await input.check(text, "tool", signal);
    }
  }
}

function hasAsyncDetection(options: ShieldAISdkOptions): boolean {
  return [options.detect, options.scanToolResults].some(
    (value) =>
      typeof value === "object" &&
      Boolean(value.secondaryDetector || value.escalate)
  );
}

export function shieldMiddleware(options: ShieldAISdkOptions = {}) {
  const shield = createShield(options);
  return {
    wrapParams<P extends AISdkParams>(params: P): P {
      if (hasAsyncDetection(options)) {
        throw new ShieldError(
          "Use wrapParamsAsync for hosted or asynchronous detection.",
          "ASYNC_DETECTION_REQUIRES_AWAIT"
        );
      }
      checkParams(params, shield.input);

      if (!params.system) {
        return { ...params };
      }
      return {
        ...params,
        system: hardenSystem(params.system, shield),
      };
    },

    async wrapParamsAsync<P extends AISdkParams>(params: P): Promise<P> {
      await checkParamsAsync(params, shield.input);
      return params.system
        ? { ...params, system: hardenSystem(params.system, shield) }
        : { ...params };
    },

    /** Redacts prompt leaks and output findings from `text`. */
    sanitizeOutput(text: string, systemPrompt?: string): string {
      return shield.output.text(text, systemPrompt ?? options.systemPrompt);
    },
  };
}

interface PromptMessage {
  role: string;
  content: unknown;
}

/** The part of the call options every AI SDK middleware version receives. */
interface LanguageModelCallOptions {
  prompt: PromptMessage[];
}

interface ContentPart {
  type: string;
  text?: string;
  /** Tool call arguments on AI SDK 5 and later. */
  input?: unknown;
}

interface LanguageModelGenerateResult {
  /** AI SDK 5 and later. */
  content?: ContentPart[];
  /** AI SDK 4. */
  text?: string;
  /** AI SDK 4. */
  toolCalls?: Array<{ args?: unknown }>;
  /** AI SDK 5 and later. Its `body` is the provider's raw response. */
  response?: object;
  /** AI SDK 4. Its `body` is the provider's raw response. */
  rawResponse?: object;
}

/**
 * AI SDK 5 and later send `text-delta` parts with an `id` and `delta`, and
 * tool call arguments as `tool-input-delta` parts before the `tool-call`.
 * AI SDK 4 sends `text-delta` parts with `textDelta` and no id, and
 * `tool-call-delta` parts with `argsTextDelta`.
 */
interface LanguageModelStreamPart {
  type: string;
  id?: string;
  delta?: string;
  textDelta?: string;
  argsTextDelta?: string;
  toolCallId?: string;
  toolCallType?: string;
  toolName?: string;
  /** Tool call arguments on AI SDK 5 and later. */
  input?: unknown;
  /** Tool call arguments on AI SDK 4. */
  args?: unknown;
  providerMetadata?: unknown;
  error?: unknown;
}

interface LanguageModelStreamResult {
  stream: ReadableStream<LanguageModelStreamPart>;
}

/** Assignable to `LanguageModelMiddleware` from AI SDK 4, 5, and 6. */
export interface ShieldLanguageModelMiddleware {
  readonly specificationVersion: "v3";
  transformParams: <P extends LanguageModelCallOptions>(options: {
    params: P;
  }) => Promise<P>;
  wrapGenerate: <R extends LanguageModelGenerateResult>(options: {
    doGenerate: () => PromiseLike<R>;
    params: LanguageModelCallOptions;
  }) => Promise<R>;
  wrapStream: <R extends LanguageModelStreamResult>(options: {
    doStream: () => PromiseLike<R>;
    params: LanguageModelCallOptions;
  }) => Promise<R>;
}

/** Extract system prompt from AI SDK internal prompt format. */
function extractSystemFromPrompt(prompt: PromptMessage[]): string {
  return prompt
    .filter(
      (m): m is { role: string; content: string } =>
        m.role === "system" && typeof m.content === "string"
    )
    .map((m) => m.content)
    .join("\n");
}

/** Extract user text from AI SDK internal prompt format. */
function extractUserTextFromPrompt(prompt: PromptMessage[]): string[] {
  return prompt
    .filter((m) => m.role === "user")
    .flatMap((m) => {
      const c = m.content;
      if (Array.isArray(c)) {
        return c
          .filter(
            (p): p is { type: string; text: string } =>
              p &&
              typeof p === "object" &&
              p.type === "text" &&
              typeof (p as { text?: unknown }).text === "string"
          )
          .map((p) => p.text);
      }
      return [];
    });
}

function isTextPart(part: ContentPart): part is ContentPart & { text: string } {
  return part.type === "text" && typeof part.text === "string";
}

/**
 * Guards the text and tool call arguments of a `doGenerate` result, the text
 * parts as one text. If any of it was redacted, the provider's raw response
 * body is dropped too, since it still holds the original.
 */
function guardGenerateResult<R extends LanguageModelGenerateResult>(
  result: R,
  systemPrompt: string,
  output: OutputGuard
): R {
  let redacted = false;
  const guardText = (text: string): string => {
    const safe = output.text(text, systemPrompt);
    redacted ||= safe !== text;
    return safe;
  };
  const guardTexts = (texts: string[]): string[] => {
    const safe = output.texts(texts, systemPrompt);
    redacted ||= safe.some((text, i) => text !== texts[i]);
    return safe;
  };
  const guardArgs = (args: unknown): unknown => {
    if (typeof args === "string") {
      return guardText(args);
    }
    if (!isRecord(args)) {
      return args;
    }
    const safe = output.value(args, systemPrompt);
    redacted ||= safe !== args;
    return safe;
  };

  let guarded = result;
  if (result.content) {
    const texts = guardTexts(
      result.content.filter(isTextPart).map((part) => part.text)
    );
    let next = 0;
    guarded = {
      ...result,
      content: result.content.map((part) => {
        if (isTextPart(part)) {
          return { ...part, text: texts[next++] };
        }
        if (part.type === "tool-call" && "input" in part) {
          const input = guardArgs(part.input);
          return input === part.input ? part : { ...part, input };
        }
        return part;
      }),
    };
  } else if (typeof result.text === "string") {
    guarded = { ...result, text: guardText(result.text) };
  }
  if (Array.isArray(result.toolCalls)) {
    guarded = {
      ...guarded,
      toolCalls: result.toolCalls.map((call) => {
        const args = guardArgs(call.args);
        return args === call.args ? call : { ...call, args };
      }),
    };
  }
  if (!redacted) {
    return guarded;
  }
  const { response, rawResponse } = guarded;
  return {
    ...guarded,
    ...(response && { response: { ...response, body: undefined } }),
    ...(rawResponse && { rawResponse: { ...rawResponse, body: undefined } }),
  };
}

type DeltaField = "delta" | "textDelta" | "argsTextDelta";

/** Which block a streamed delta belongs to, and how to emit more of it. */
interface DeltaBlock {
  key: string;
  field: DeltaField;
  /** The part to put text in when the block is flushed. */
  template: LanguageModelStreamPart;
}

const textKey = (id: string | undefined): string => `text:${id ?? ""}`;
const toolKey = (id: string | undefined): string => `tool:${id ?? ""}`;

function deltaBlock(part: LanguageModelStreamPart): DeltaBlock | undefined {
  switch (part.type) {
    case "text-delta":
      return part.id === undefined
        ? {
            key: textKey(undefined),
            field: "textDelta",
            template: { type: "text-delta" },
          }
        : {
            key: textKey(part.id),
            field: "delta",
            template: { type: "text-delta", id: part.id },
          };
    case "tool-input-delta":
      return {
        key: toolKey(part.id),
        field: "delta",
        template: { type: "tool-input-delta", id: part.id },
      };
    case "tool-call-delta":
      return {
        key: toolKey(part.toolCallId),
        field: "argsTextDelta",
        template: {
          type: "tool-call-delta",
          toolCallType: part.toolCallType,
          toolCallId: part.toolCallId,
          toolName: part.toolName,
        },
      };
    default:
      return;
  }
}

/** The block whose deltas end at `part`, other than at `finish`. */
function endedBlock(part: LanguageModelStreamPart): string | undefined {
  switch (part.type) {
    case "text-end":
      return textKey(part.id);
    case "tool-input-end":
      return toolKey(part.id);
    // AI SDK 4 sends no end part for tool call deltas.
    case "tool-call":
      return toolKey(part.toolCallId);
    default:
      return;
  }
}

/** Where a `tool-call` part holds its arguments as a string. */
function argumentField(
  part: LanguageModelStreamPart
): "input" | "args" | undefined {
  if (typeof part.input === "string") {
    return "input";
  }
  if (typeof part.args === "string") {
    return "args";
  }
}

const isText = (block: DeltaBlock): boolean =>
  block.template.type === "text-delta";

/**
 * Guards the text and tool call arguments of a model stream and passes every
 * other part through, except `raw` parts, which carry the provider's chunks
 * before sanitization. Each tool call's argument deltas are held until they
 * end, then re-emitted in 64-character deltas ahead of their end part:
 * `tool-input-end`, the `tool-call` on AI SDK 4, or `finish`.
 *
 * Text blocks, which citations and some providers split a reply into, are
 * guarded as one text. With an infinite `chunkSize` (buffer mode), every
 * part from the first text delta on is held until `finish`, and each block's
 * text is re-emitted the same way ahead of its `text-end`. With a finite
 * one, each block's text goes out as each chunk fills and the rest ahead of
 * its end, and each block is scanned with the end of the one before it.
 *
 * With `throwOnLeak` or `blockOnOutputFindings`, a leak or blocked finding
 * ends the stream with an `error` part in place of the text. Every AI SDK
 * version reports an `error` part through `onError` and `fullStream` and
 * still settles its result promises, which erroring the stream itself does
 * not do on AI SDK 4 and 5.
 */
function guardStreamParts(
  systemPrompt: string,
  chunkSize: number,
  output: OutputGuard,
  options: ShieldAISdkOptions
): TransformStream<LanguageModelStreamPart, LanguageModelStreamPart> {
  type Controller = TransformStreamDefaultController<LanguageModelStreamPart>;
  const blocks = new Map<
    string,
    { sanitizer: ChunkedSanitizer; block: DeltaBlock }
  >();
  const overlap = output.overlap(chunkSize);
  const buffered = chunkSize === Number.POSITIVE_INFINITY;
  /** The sanitizer of the last text block that ended, for the next one. */
  let lastText: ChunkedSanitizer | undefined;
  /** In buffer mode, the parts from the first text delta on. */
  let held: LanguageModelStreamPart[] | undefined;

  /** Always returns false, since the stream has ended. */
  const stop = (controller: Controller, error: ShieldError): false => {
    controller.enqueue({ type: "error", error });
    controller.terminate();
    return false;
  };

  /** Returns false if a leak or blocked finding ended the stream. */
  const enqueueText = (
    controller: Controller,
    part: LanguageModelStreamPart,
    field: DeltaField,
    results: ChunkResult[]
  ): boolean => {
    for (const result of results) {
      const verdict = output.report(result);
      if (verdict.leak && options.throwOnLeak) {
        return stop(controller, verdict.leak);
      }
      if (verdict.block) {
        return stop(controller, verdict.block);
      }
      for (const text of chunkString(result.sanitized)) {
        controller.enqueue({ ...part, [field]: text });
      }
    }
    return true;
  };

  const flushBlock = (controller: Controller, key: string): boolean => {
    const entry = blocks.get(key);
    if (!entry) {
      return true;
    }
    blocks.delete(key);
    const last = entry.sanitizer.flush();
    if (isText(entry.block)) {
      lastText = entry.sanitizer;
    }
    const { template, field } = entry.block;
    return !last || enqueueText(controller, template, field, [last]);
  };

  const flushBlocks = (controller: Controller): boolean => {
    for (const key of [...blocks.keys()]) {
      if (!flushBlock(controller, key)) {
        return false;
      }
    }
    return true;
  };

  /** Returns false if a leak or blocked finding ended the stream. */
  const pushDelta = (
    controller: Controller,
    part: LanguageModelStreamPart,
    block: DeltaBlock
  ): boolean => {
    let entry = blocks.get(block.key);
    if (!entry) {
      const carried = isText(block) ? lastText : undefined;
      if (carried) {
        lastText = undefined;
      }
      entry = {
        sanitizer:
          carried ??
          createChunkedSanitizer(
            systemPrompt,
            output.scanWindow,
            chunkSize,
            overlap
          ),
        block,
      };
      blocks.set(block.key, entry);
    }
    const results = entry.sanitizer.push(String(part[block.field] ?? ""));
    if (!enqueueText(controller, part, block.field, results)) {
      return false;
    }
    // Keep provider metadata that arrives on a delta whose text is still
    // held back.
    if (part.providerMetadata && !results.some((r) => r.sanitized)) {
      controller.enqueue({ ...part, [block.field]: "" });
    }
    return true;
  };

  /** The tool call with its arguments guarded, or undefined if that ended the stream. */
  const guardToolCall = (
    controller: Controller,
    part: LanguageModelStreamPart
  ): LanguageModelStreamPart | undefined => {
    const field = argumentField(part);
    if (!field) {
      return part;
    }
    const text = part[field] as string;
    try {
      const safe = output.text(text, systemPrompt);
      return safe === text ? part : { ...part, [field]: safe };
    } catch (error) {
      if (error instanceof ShieldError) {
        stop(controller, error);
        return;
      }
      throw error;
    }
  };

  /** Flushes the blocks `part` ends. Returns false if that ended the stream. */
  const flushEnded = (
    controller: Controller,
    part: LanguageModelStreamPart
  ): boolean => {
    if (part.type === "finish") {
      return flushBlocks(controller);
    }
    const key = endedBlock(part);
    return key === undefined || flushBlock(controller, key);
  };

  /** Returns false if a leak or blocked finding ended the stream. */
  const handle = (
    controller: Controller,
    part: LanguageModelStreamPart
  ): boolean => {
    const block = deltaBlock(part);
    if (block) {
      return pushDelta(controller, part, block);
    }
    if (!flushEnded(controller, part)) {
      return false;
    }
    const next =
      part.type === "tool-call" ? guardToolCall(controller, part) : part;
    if (next) {
      controller.enqueue(next);
    }
    return next !== undefined;
  };

  /**
   * Buffer mode: guards the text of the held parts as one text, then sends
   * them on, each block's text ahead of the part that ends it. Returns false
   * if a leak or blocked finding ended the stream.
   */
  const release = (controller: Controller): boolean => {
    const parts = held ?? [];
    held = undefined;
    const texts = new Map<string, { block: DeltaBlock; text: string }>();
    for (const part of parts) {
      const block = deltaBlock(part);
      if (block && isText(block)) {
        const text = texts.get(block.key)?.text ?? "";
        texts.set(block.key, {
          block,
          text: text + String(part[block.field] ?? ""),
        });
      }
    }
    let safe: string[];
    try {
      safe = output.texts(
        [...texts.values()].map((entry) => entry.text),
        systemPrompt
      );
    } catch (error) {
      if (error instanceof ShieldError) {
        return stop(controller, error);
      }
      throw error;
    }
    const pending = new Map(
      [...texts].map(([key, { block }], i) => [key, { block, text: safe[i] }])
    );
    /** Sends the text of the blocks `keys` name, or of every block left. */
    const emit = (keys = [...pending.keys()]): void => {
      for (const key of keys) {
        const entry = pending.get(key);
        if (!entry) {
          continue;
        }
        pending.delete(key);
        const { template, field } = entry.block;
        for (const text of chunkString(entry.text)) {
          controller.enqueue({ ...template, [field]: text });
        }
      }
    };
    for (const part of parts) {
      const block = deltaBlock(part);
      if (block && isText(block)) {
        // Keep provider metadata that arrives on a text delta.
        if (part.providerMetadata) {
          controller.enqueue({ ...part, [block.field]: "" });
        }
        continue;
      }
      const ended = endedBlock(part);
      if (part.type === "finish") {
        emit();
      } else if (ended !== undefined) {
        emit([ended]);
      }
      if (!handle(controller, part)) {
        return false;
      }
    }
    emit();
    return true;
  };

  return new TransformStream({
    transform(part, controller) {
      if (part.type === "raw") {
        return;
      }
      if (held) {
        held.push(part);
        if (part.type === "finish") {
          release(controller);
        }
        return;
      }
      if (buffered && part.type === "text-delta") {
        held = [part];
        return;
      }
      handle(controller, part);
    },
    flush(controller) {
      if (!held || release(controller)) {
        flushBlocks(controller);
      }
    },
  });
}

/**
 * AI SDK language model middleware. Pass it to `wrapLanguageModel` for
 * automatic hardening, injection detection on user input and tool results,
 * and output guarding in `generateText` and `streamText`, with no manual
 * `sanitizeOutput` call. Works with AI SDK 4, 5, and 6.
 *
 * @example
 * ```ts
 * import { wrapLanguageModel, generateText } from "ai";
 * import { openai } from "@ai-sdk/openai";
 * import { shieldLanguageModelMiddleware } from "@zeroleaks/shield/ai-sdk";
 *
 * const model = wrapLanguageModel({
 *   model: openai("gpt-5.5"),
 *   middleware: shieldLanguageModelMiddleware(),
 * });
 *
 * const result = await generateText({ model, system: "You are helpful.", prompt: "Hi" });
 * // result.text is sanitized against the system prompt
 * ```
 */
export function shieldLanguageModelMiddleware(
  options: ShieldAISdkOptions = {}
): ShieldLanguageModelMiddleware {
  const shield = createShield(options);
  const { output } = shield;
  // The AI SDK hands the params object transformParams returns to
  // wrapGenerate and wrapStream, so key the pre-hardening prompt on it.
  // Params from anywhere else fall back to `options.systemPrompt` or the
  // system messages they carry.
  const systemPrompts = new WeakMap<LanguageModelCallOptions, string>();
  const systemPromptFor = (params: LanguageModelCallOptions): string =>
    systemPrompts.get(params) ??
    options.systemPrompt ??
    extractSystemFromPrompt(params.prompt);

  return {
    specificationVersion: "v3",

    transformParams: async ({ params }) => {
      const signal = requestSignal(params);
      for (const text of extractUserTextFromPrompt(params.prompt)) {
        await shield.input.check(text, "user", signal);
      }
      if (shield.input.tool) {
        for (const text of toolResultTexts(params.prompt)) {
          await shield.input.check(text, "tool", signal);
        }
      }

      const prompt = params.prompt.map((msg) =>
        msg.role === "system" && typeof msg.content === "string"
          ? { ...msg, content: hardenText(msg.content, shield) }
          : msg
      );
      const transformed = { ...params, prompt };
      systemPrompts.set(
        transformed,
        options.systemPrompt ?? extractSystemFromPrompt(params.prompt)
      );
      return transformed;
    },

    wrapGenerate: async ({ doGenerate, params }) => {
      const result = await doGenerate();
      const systemText = systemPromptFor(params);
      if (!output.active(systemText)) {
        return result;
      }
      return guardGenerateResult(result, systemText, output);
    },

    wrapStream: async ({ doStream, params }) => {
      const result = await doStream();
      const systemText = systemPromptFor(params);
      const mode = options.streamingSanitize ?? "buffer";
      if (mode === "passthrough" || !output.active(systemText)) {
        return result;
      }

      const chunkSize =
        mode === "chunked"
          ? (options.streamingChunkSize ?? 8192)
          : Number.POSITIVE_INFINITY;
      return {
        ...result,
        stream: result.stream.pipeThrough(
          guardStreamParts(systemText, chunkSize, output, options)
        ),
      };
    },
  };
}
