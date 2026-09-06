import { ReactAdapterDiagnostics } from "./diagnostics"
import type {
  JSONSchema7,
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3FinishReason,
  LanguageModelV3GenerateResult,
  LanguageModelV3Middleware,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  LanguageModelV3Usage,
} from "@ai-sdk/provider"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"
import { ReactAdapterConfig, type Mode } from "./config"
import { downconvert, type Epoch } from "./downconvert"
import { parse } from "./parse"

export * as ReactAdapter from "./index"
export type { Epoch, Mode }

/**
 * Per-request runtime state supplied by opencode for epoch mode (and a
 * per-session layout override). Everything is optional: without it the
 * adapter is fully stateless and epoch mode degrades to single-message mode.
 */
export type Runtime = {
  /** Persisted epoch for this session, if any. */
  epoch?: Epoch
  /** Per-session layout override (e.g. set by /react <mode>); wins over config. */
  mode?: Mode
  /** Persist a freshly seeded epoch. Awaited before the request goes out. */
  onSeed?: (epoch: Epoch) => Promise<void>
}

/**
 * ReAct adapter (phase 3 assembly): AI SDK middleware that down-converts
 * native tool calling into a ReAct text protocol for a degraded
 * OpenAI-compatible provider (user/assistant roles only, no tools, unreliable
 * finish_reason) and up-converts the model's text back into synthetic
 * tool-call stream parts, so opencode's core never knows the difference.
 *
 * v1 buffers the full upstream response and emits synthetic parts at the end —
 * correctness over streaming UX.
 * TODO: incremental parsing — emit text deltas as they arrive until a label
 * line is detected, then hold back; requires a streaming variant of parse().
 */

/** Distinguishable error for the laundered-recitation case (empty response with finish_reason "stop"). */
export class EmptyResponseError extends Error {
  constructor(attempts: number) {
    super(
      `react-adapter: provider returned an empty response ${attempts} time(s); ` +
        `this is typically a recitation abort laundered into finish_reason "stop". ` +
        `Retry the request or reduce verbatim canonical content in the context.`,
    )
    this.name = "ReactAdapterEmptyResponseError"
  }
}

const NUDGE =
  "[Your previous response was empty. Reply now using the required format: Thought, then Action and Action Input, or Final Answer.]"

const EMPTY_USAGE: LanguageModelV3Usage = {
  inputTokens: { total: undefined, noCache: undefined, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: undefined, text: undefined, reasoning: undefined },
}

/** Failed empty turns are never persisted; retries reuse the same clean prompt plus this nudge. */
function nudged(params: LanguageModelV3CallOptions): LanguageModelV3CallOptions {
  const prompt = params.prompt.map((message, i) => {
    if (i !== params.prompt.length - 1 || message.role !== "user") return message
    return { ...message, content: [...message.content, { type: "text" as const, text: `\n\n${NUDGE}` }] }
  })
  return { ...params, prompt }
}

type Buffered = {
  text: string
  parts: LanguageModelV3StreamPart[]
  finishReason: LanguageModelV3FinishReason
  usage: LanguageModelV3Usage
  errored: boolean
}

async function drain(result: LanguageModelV3StreamResult): Promise<Buffered> {
  const parts: LanguageModelV3StreamPart[] = []
  const reader = result.stream.getReader()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    parts.push(value)
  }
  const finish = parts.find((p) => p.type === "finish")
  return {
    text: parts.flatMap((p) => (p.type === "text-delta" ? [p.delta] : [])).join(""),
    parts,
    finishReason: finish?.finishReason ?? { unified: "stop", raw: undefined },
    usage: finish?.usage ?? EMPTY_USAGE,
    errored: parts.some((p) => p.type === "error"),
  }
}

function synthesize(
  buffered: Buffered,
  schemas: Record<string, JSONSchema7 | undefined>,
): LanguageModelV3StreamPart[] {
  const parsed = parse(buffered.text, schemas)
  const passthrough = buffered.parts.filter(
    (p) =>
      p.type === "stream-start" ||
      p.type === "response-metadata" ||
      p.type === "reasoning-start" ||
      p.type === "reasoning-delta" ||
      p.type === "reasoning-end",
  )
  const out: LanguageModelV3StreamPart[] = [...passthrough]

  const id = crypto.randomUUID()
  const emitText = (text: string) => {
    out.push(
      { type: "text-start", id: `react-text-${id}` },
      { type: "text-delta", id: `react-text-${id}`, delta: text },
      { type: "text-end", id: `react-text-${id}` },
    )
  }

  switch (parsed.kind) {
    case "action": {
      const visible = [parsed.leading, parsed.thought].filter((x) => x).join("\n\n")
      if (visible) emitText(visible)
      out.push(
        { type: "tool-input-start", id, toolName: parsed.toolName },
        { type: "tool-input-delta", id, delta: parsed.input },
        { type: "tool-input-end", id },
        { type: "tool-call", toolCallId: id, toolName: parsed.toolName, input: parsed.input },
        {
          type: "finish",
          usage: buffered.usage,
          finishReason: { unified: "tool-calls", raw: buffered.finishReason.raw },
        },
      )
      break
    }
    case "final": {
      emitText(parsed.answer)
      out.push({
        type: "finish",
        usage: buffered.usage,
        finishReason: { unified: "stop", raw: buffered.finishReason.raw },
      })
      break
    }
    case "text": {
      emitText(parsed.text)
      out.push({ type: "finish", usage: buffered.usage, finishReason: buffered.finishReason })
      break
    }
  }
  return out
}

function replay(parts: LanguageModelV3StreamPart[]): ReadableStream<LanguageModelV3StreamPart> {
  return new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(part)
      controller.close()
    },
  })
}

export function middleware(
  providerOptions: Record<string, unknown> | undefined,
  runtime?: Runtime,
): LanguageModelV3Middleware[] {
  const resolved = ReactAdapterConfig.resolve(providerOptions)
  if (!resolved) return []
  const cfg = runtime?.mode ? { ...resolved, mode: runtime.mode } : resolved

  // transformParams strips `tools` from the outgoing call, but the parser
  // still needs the schemas for bare-string input recovery. The middleware
  // array is constructed per request and transformParams always runs before
  // wrapStream/wrapGenerate for the same call, so a closure is safe here.
  let schemas: Record<string, JSONSchema7 | undefined> = {}
  let originalParams: any
  let downconvertedParams: any

  return [
    {
      specificationVersion: "v3",
      async transformParams(args) {
        schemas = Object.fromEntries(
          (args.params.tools ?? []).flatMap((t) => (t.type === "function" ? [[t.name, t.inputSchema]] : [])),
        )
        let seeding: Promise<void> | undefined
        const downconverted = downconvert(args.params, cfg, {
          epoch: runtime?.epoch,
          onSeed: (epoch) => {
            seeding = runtime?.onSeed?.(epoch)
          },
        })
        await seeding

        originalParams = args.params
        downconvertedParams = downconverted

        if (process.env.OPENCODE_LOG_LEVEL === "DEBUG") {
          const logPath = path.join(os.homedir(), ".local/share/opencode/log/opencode.log")
          let logOutput = "\n=== REACT ADAPTER OUTBOUND PROMPT ===\n"
          for (const msg of downconverted.prompt) {
            logOutput += `\n[Role: ${msg.role}]\n`
            if (Array.isArray(msg.content)) {
              for (const part of msg.content) {
                if (part.type === "text") logOutput += part.text + "\n"
              }
            } else if (typeof msg.content === "string") {
              logOutput += msg.content + "\n"
            }
          }
          logOutput += "\n=====================================\n\n"
          try {
            fs.appendFileSync(logPath, logOutput)
          } catch (e) {
            // Ignore logging errors
          }
        }

        return downconverted
      },
      async wrapStream({ doStream, params, model }) {
        ReactAdapterDiagnostics.logContext(originalParams, downconvertedParams, model)
        const { result, buffered } = await withEmptyRetry(cfg.retries, () => doStream(), params, model)
        if (!buffered) {
          // Retries exhausted: distinguishable error, never a silent stop.
          return {
            stream: replay([
              { type: "stream-start", warnings: [] },
              { type: "error", error: new EmptyResponseError(cfg.retries + 1) },
              {
                type: "finish",
                usage: EMPTY_USAGE,
                finishReason: { unified: "error", raw: "react-adapter/empty-response" },
              },
            ]),
          }
        }
        // Upstream errors pass through untouched.
        if (buffered.errored) return { ...result, stream: replay(buffered.parts) }
        return { ...result, stream: replay(synthesize(buffered, schemas)) }
      },
      async wrapGenerate({ doGenerate, params, model }) {
        ReactAdapterDiagnostics.logContext(originalParams, downconvertedParams, model)
        let attempt = 0
        let generated = await doGenerate()
        while (isEmptyGenerate(generated) && attempt < cfg.retries) {
          attempt++
          generated = await model.doGenerate(nudged(params))
        }
        if (isEmptyGenerate(generated)) throw new EmptyResponseError(attempt + 1)

        const text = generated.content.flatMap((c) => (c.type === "text" ? [c.text] : [])).join("")
        const parsed = parse(text, schemas)
        const result: LanguageModelV3GenerateResult = { ...generated }
        switch (parsed.kind) {
          case "action": {
            const visible = [parsed.leading, parsed.thought].filter((x) => x).join("\n\n")
            result.content = [
              ...(visible ? [{ type: "text" as const, text: visible }] : []),
              {
                type: "tool-call" as const,
                toolCallId: crypto.randomUUID(),
                toolName: parsed.toolName,
                input: parsed.input,
              },
            ]
            result.finishReason = { unified: "tool-calls", raw: generated.finishReason.raw }
            break
          }
          case "final": {
            result.content = [{ type: "text", text: parsed.answer }]
            result.finishReason = { unified: "stop", raw: generated.finishReason.raw }
            break
          }
          case "text":
            break
        }
        return result
      },
    },
  ]
}

function isEmptyGenerate(result: LanguageModelV3GenerateResult) {
  return !result.content.some((c) => (c.type === "text" && c.text.trim() !== "") || c.type === "tool-call")
}

async function withEmptyRetry(
  retries: number,
  first: () => PromiseLike<LanguageModelV3StreamResult>,
  params: LanguageModelV3CallOptions,
  model: LanguageModelV3,
) {
  let result = await first()
  let buffered = await drain(result)
  let attempt = 0
  while (buffered.text.trim() === "" && !buffered.errored && attempt < retries) {
    attempt++
    result = await model.doStream(nudged(params))
    buffered = await drain(result)
  }
  if (buffered.text.trim() === "" && !buffered.errored) return { result, buffered: undefined }
  return { result, buffered }
}
