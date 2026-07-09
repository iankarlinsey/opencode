import type {
  LanguageModelV3CallOptions,
  LanguageModelV3Message,
  LanguageModelV3ToolResultPart,
} from "@ai-sdk/provider"
import type { Config } from "./config"
import { normalize } from "./normalize"

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
          .map((c) => (c.type === "text" ? c.text : `[attachment omitted: ${"mediaType" in c ? c.mediaType : c.url}]`))
          .join("\n")
    }
  })()
  return `${cfg.resultPrefix} [${part.toolName}]:\n${normalize(part.toolName, text, cfg.caps)}`
}

const NUDGE = "Continue: state your next Thought and Action, or give your Final Answer."

export function downconvert(params: LanguageModelV3CallOptions, cfg: Config): LanguageModelV3CallOptions {
  const system: string[] = []
  const turns: { role: "user" | "assistant"; text: string }[] = []
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

  const preamble = [...system, renderCatalog(params.tools), protocolRules(cfg.resultPrefix), EXAMPLES, DELIMITER]
    .filter((x) => x)
    .join("\n\n")
  if (turns[0]?.role === "user") turns[0].text = `${preamble}\n\n${turns[0].text}`
  else turns.unshift({ role: "user", text: preamble })

  return {
    ...params,
    prompt: turns.map(
      (turn): LanguageModelV3Message => ({ role: turn.role, content: [{ type: "text", text: turn.text }] }),
    ),
    tools: undefined,
    toolChoice: undefined,
    stopSequences: [...new Set([...(params.stopSequences ?? []), `\n${cfg.resultPrefix}`, "\nObservation:"])],
  }
}
