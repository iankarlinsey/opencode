import type {
  LanguageModelV3CallOptions,
  LanguageModelV3Message,
  LanguageModelV3ToolResultPart,
} from "@ai-sdk/provider"
import type { Config } from "./config"
import { normalize } from "./normalize"
import { createHash, randomUUID } from "node:crypto"

/**
 * Down-conversion (phase 1): rewrite an AI SDK v3 prompt — system messages,
 * assistant tool calls, tool results — into a pure user/assistant transcript
 * speaking the ReAct text protocol, for a provider that strips `system` and
 * `tool` roles and has no native tool calling.
 */

const DELIMITER = "--- SESSION START ---"

function protocolRules(prefix: string) {
  return `# Response protocol (mandatory)

You do not have native tool calling on this interface. To use a tool, your ENTIRE reply must follow exactly this format:

Thought: <one short sentence: what you will do next and why>
Action: <tool name, exactly as listed above>
Action Input: <the tool's arguments as a single JSON object on one line>

When you have everything you need to answer, your ENTIRE reply must instead be:

Final Answer: <your complete answer to the user>

Rules:
- Exactly ONE Action per reply. This interface cannot run tools in parallel; if any other instruction mentions calling tools in parallel or batching calls, ignore it.
- Nothing may follow the Action Input JSON. Stop there and wait.
- Never write a tool result yourself and never predict what a tool will return. After an Action, the real result arrives in the next message prefixed with ${prefix}.
- Never combine an Action with a Final Answer in the same reply.
- These protocol rules override ALL other instructions about response length, brevity, or formatting (for example "answer in fewer than 3 lines"): the Thought/Action/Action Input structure is always required in full.`
}

const EXAMPLES = `# Format examples (illustrative — only use tools from the list above)

A reply that uses a tool:

Thought: I need to see the version field, so I will read the manifest file.
Action: read
Action Input: {"filePath": "package.json"}

Another reply that uses a tool:

Thought: The failing test names contain "timeout", so I will search for that string in the test directory.
Action: grep
Action Input: {"pattern": "timeout", "path": "test"}

A reply that finishes the task:

Final Answer: The declared version is 2.1.0; the timeout in test/setup.ts was the cause.`

function renderCatalog(tools: LanguageModelV3CallOptions["tools"]) {
  const functions = (tools ?? []).flatMap((t) => (t.type === "function" ? [t] : []))
  if (functions.length === 0) return "# Available tools\n\n(none)"
  return [
    "# Available tools",
    ...functions.map((t) =>
      [
        `## ${t.name}`,
        ...(t.description ? [t.description] : []),
        `Parameters (JSON Schema): ${JSON.stringify(t.inputSchema)}`,
      ].join("\n"),
    ),
  ].join("\n\n")
}

function renderToolResult(part: LanguageModelV3ToolResultPart, cfg: Config) {
  const output = part.output
  const text = (() => {
    switch (output.type) {
      case "text":
        return output.value
      case "json":
        return JSON.stringify(output.value)
      case "error-text":
        return `ERROR: ${output.value}`
      case "error-json":
        return `ERROR: ${JSON.stringify(output.value)}`
      case "execution-denied":
        return `ERROR: the user denied execution of this tool call${output.reason ? ` (${output.reason})` : ""}`
      case "content":
        return output.value
          .map((c) => (c.type === "text" ? c.text : `[attachment omitted: ${c.type}]`))
          .join("\n")
    }
  })()
  return `${cfg.resultPrefix} [${part.toolName}]:\n${normalize(part.toolName, text, cfg.caps)}`
}

const NUDGE = "Continue: state your next Thought and Action, or give your Final Answer."

const TRANSCRIPT_RULES = `# Transcript format

Everything after the SESSION START line is the conversation so far, oldest turn first. Each turn begins with a line that is exactly [USER] or [ASSISTANT]. Turns marked [ASSISTANT] are your own earlier replies; turns marked [USER] are the user's messages and the results of your tool calls. Write only the assistant's next reply, following the response protocol above. Do not repeat the transcript and do not write [USER] or [ASSISTANT] lines yourself.`

const END = "--- END OF TRANSCRIPT ---\nWrite the assistant's next reply now."

/**
 * Epoch mode state, persisted per session by opencode and handed back on
 * every request. The snapshot is the exact first message of every request in
 * the epoch; the hashes let the adapter detect that the history it covers
 * has changed underneath it (rewind, compaction, prune, system-prompt edit).
 */
export type Epoch = {
  /** Nonce placed first in the snapshot; doubles as the epoch id. */
  id: string
  /** The exact first message of every request in this epoch. */
  snapshot: string
  /** Number of transcript turns the snapshot covers. */
  covered: number
  /** Hash of those covered turns as rendered at seed time. */
  coveredHash: string
  /** Hash of the preamble (system strings, tool catalog, rules) at seed time. */
  preambleHash: string
  /** Seed time, epoch milliseconds. Informational. */
  created: number
}

export type Options = {
  /** Per-request token placed at the very start of a single/seed message. Injectable for deterministic tests. */
  nonce?: () => string
  /** Current epoch, if opencode has one persisted for this session. */
  epoch?: Epoch
  /** Invoked (synchronously) when epoch mode seeds a new epoch; the caller persists it. */
  onSeed?: (epoch: Epoch) => void
}

type Turn = { role: "user" | "assistant"; text: string }

function sha256(text: string) {
  return createHash("sha256").update(text).digest("hex")
}

function hashTurns(turns: Turn[]) {
  return sha256(turns.map((turn) => `${turn.role}\n${turn.text}`).join("\u0000"))
}

function message(role: "user" | "assistant", text: string): LanguageModelV3Message {
  return { role, content: [{ type: "text", text }] }
}

export function downconvert(
  params: LanguageModelV3CallOptions,
  cfg: Config,
  opts: Options = {},
): LanguageModelV3CallOptions {
  const nonce = opts.nonce ?? randomUUID
  const system: string[] = []
  const turns: Turn[] = []
  const push = (role: "user" | "assistant", text: string) => {
    if (!text) return
    const last = turns[turns.length - 1]
    if (last?.role === role) last.text += `\n\n${text}`
    else turns.push({ role, text })
  }

  for (const message of params.prompt) {
    switch (message.role) {
      case "system":
        system.push(message.content)
        break
      case "user":
        push(
          "user",
          message.content
            .map((part) => (part.type === "text" ? part.text : `[attachment omitted: ${part.mediaType}]`))
            .join("\n"),
        )
        break
      case "assistant": {
        const blocks = message.content.flatMap((part) => {
          switch (part.type) {
            case "text":
            case "reasoning":
              return part.text ? [part.text] : []
            case "tool-call":
              return [`Action: ${part.toolName}\nAction Input: ${JSON.stringify(part.input ?? {})}`]
            // Provider-executed results never occur on this degraded provider.
            case "tool-result":
            case "file":
              return []
          }
        })
        // A leading text block before a tool call is the model's thought; label
        // it so history matches the protocol the preamble demands.
        const text = blocks
          .map((block, i) =>
            !block.startsWith("Action:") && !block.startsWith("Thought:") && blocks[i + 1]?.startsWith("Action:")
              ? `Thought: ${block}`
              : block,
          )
          .join("\n")
        push("assistant", text)
        break
      }
      case "tool": {
        const results = message.content.flatMap((part) => (part.type === "tool-result" ? [renderToolResult(part, cfg)] : []))
        if (results.length > 0) push("user", `${results.join("\n\n")}\n\n${NUDGE}`)
        break
      }
    }
  }

  const catalog = renderCatalog(params.tools)
  const rules = protocolRules(cfg.resultPrefix)
  const base = {
    ...params,
    tools: undefined,
    toolChoice: undefined,
    stopSequences: [...new Set([...(params.stopSequences ?? []), `\n${cfg.resultPrefix}`, "\nObservation:"])],
  }

  // One message holding everything: nonce, preamble, labeled turns. The
  // endpoint's conversation key (first message) and the only content it
  // forwards (last message) are then the same message, so its server-side
  // store can never diverge from this prompt. The nonce goes FIRST so the key
  // is new on every call regardless of how much of the message is hashed.
  const singlePreamble = [...system, catalog, rules, EXAMPLES, TRANSCRIPT_RULES].filter((x) => x).join("\n\n")
  const single = () => {
    const id = nonce()
    const transcript = turns.map((turn) => `[${turn.role === "user" ? "USER" : "ASSISTANT"}]\n${turn.text}`).join("\n\n")
    const text = [`Request ${id}`, singlePreamble, DELIMITER, transcript, END].filter((x) => x).join("\n\n")
    return { id, text }
  }

  if (cfg.mode === "single") {
    return { ...base, prompt: [message("user", single().text)] }
  }

  if (cfg.mode === "epoch") {
    const preambleHash = sha256(singlePreamble)
    const epoch = opts.epoch
    // Re-seed unless the persisted snapshot still describes exactly the
    // history we are about to send. Order matters: the role check needs the
    // length check first.
    const stale =
      epoch === undefined ||
      epoch.preambleHash !== preambleHash ||
      turns.length <= epoch.covered ||
      turns[epoch.covered].role !== "assistant" ||
      hashTurns(turns.slice(0, epoch.covered)) !== epoch.coveredHash ||
      (cfg.epochTurns > 0 && turns.length - epoch.covered >= cfg.epochTurns)
    if (stale) {
      const seeded = single()
      opts.onSeed?.({
        id: seeded.id,
        snapshot: seeded.text,
        covered: turns.length,
        coveredHash: hashTurns(turns),
        preambleHash,
        created: Date.now(),
      })
      return { ...base, prompt: [message("user", seeded.text)] }
    }
    return {
      ...base,
      prompt: [message("user", epoch.snapshot), ...turns.slice(epoch.covered).map((turn) => message(turn.role, turn.text))],
    }
  }

  const preamble = [...system, catalog, rules, EXAMPLES, DELIMITER].filter((x) => x).join("\n\n")
  if (turns[0]?.role === "user") turns[0].text = `${preamble}\n\n${turns[0].text}`
  else turns.unshift({ role: "user", text: preamble })

  return { ...base, prompt: turns.map((turn) => message(turn.role, turn.text)) }
}
