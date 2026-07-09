import type { JSONSchema7 } from "@ai-sdk/provider"

/**
 * Up-conversion parser (phase 3): a small line-oriented state machine over the
 * model's text reply, recognizing `Thought:`, `Action:`, `Action Input:` and
 * `Final Answer:` labels at line starts. A single regex cannot handle the
 * observed failure modes (fenced replies, multi-line JSON, competing labels).
 */

export type Parsed =
  | { kind: "action"; leading?: string; thought?: string; toolName: string; input: string; warnings: string[] }
  | { kind: "final"; answer: string; warnings: string[] }
  | { kind: "text"; text: string; warnings: string[] }

const LABEL = /^\s{0,3}(thought|action\s+input|action|final\s+answer)\s*:\s*(.*)$/i

function kindOf(label: string): "thought" | "input" | "action" | "final" {
  const lower = label.toLowerCase().replace(/\s+/g, " ")
  if (lower === "thought") return "thought"
  if (lower === "action input") return "input"
  if (lower === "action") return "action"
  return "final"
}

/** Strip a markdown fence wrapping the ENTIRE reply (```...``` around the whole block). */
function unfence(text: string) {
  const trimmed = text.trim()
  const m = trimmed.match(/^```[a-zA-Z]*\n([\s\S]*?)\n?```$/)
  return m ? m[1] : text
}

function tryJSON(text: string): string | undefined {
  try {
    const value = JSON.parse(text)
    if (value !== null && typeof value === "object") return JSON.stringify(value)
  } catch {}
  return undefined
}

/**
 * Action Input tolerance ladder: strict JSON, then JSON inside a ``` fence,
 * then the outermost {...} slice, then a bare string wrapped as the tool's
 * single required string parameter. Anything else is returned raw so
 * streamText's repairToolCall can route it to the "invalid" tool.
 */
function resolveInput(raw: string, schema: JSONSchema7 | undefined, warnings: string[]): string {
  const trimmed = raw.trim()
  if (trimmed === "") return "{}"

  const direct = tryJSON(trimmed)
  if (direct) return direct

  const fence = trimmed.match(/```[a-zA-Z]*\n?([\s\S]*?)\n?```/)
  if (fence) {
    const fenced = tryJSON(fence[1].trim())
    if (fenced) return fenced
  }

  const first = trimmed.indexOf("{")
  const last = trimmed.lastIndexOf("}")
  if (first >= 0 && last > first) {
    const slice = tryJSON(trimmed.slice(first, last + 1))
    if (slice) {
      warnings.push("extracted JSON object embedded in non-JSON action input")
      return slice
    }
  }

  // Input that looks like JSON but failed every parse attempt is malformed
  // JSON, not a bare string — let repairToolCall handle it rather than
  // wrapping garbage as e.g. a shell command.
  const looksLikeJSON = trimmed.startsWith("{") || trimmed.startsWith("[")
  const properties = schema?.properties
  const required = schema?.required ?? []
  if (!looksLikeJSON && properties && required.length === 1) {
    const only = properties[required[0]]
    if (typeof only === "object" && only.type === "string") {
      warnings.push(`wrapped bare-string action input as {"${required[0]}": ...}`)
      const unquoted = trimmed.match(/^"(.*)"$/s)?.[1] ?? trimmed
      return JSON.stringify({ [required[0]]: unquoted })
    }
  }

  warnings.push("action input is not parseable JSON; passing through raw for tool-call repair")
  return trimmed
}

export function parse(raw: string, schemas?: Record<string, JSONSchema7 | undefined>): Parsed {
  const warnings: string[] = []
  const lines = unfence(raw).split("\n")
  const labels = lines.flatMap((line, index) => {
    const m = line.match(LABEL)
    return m ? [{ index, kind: kindOf(m[1]), inline: m[2] }] : []
  })

  const textUntilNextLabel = (from: { index: number; inline: string }) => {
    const next = labels.find((l) => l.index > from.index)
    const rest = lines.slice(from.index + 1, next?.index ?? lines.length)
    return [from.inline, ...rest].join("\n").trim()
  }

  const action = labels.find((l) => l.kind === "action")
  if (action) {
    if (labels.some((l) => l.kind === "action" && l !== action)) warnings.push("multiple Action labels; taking the first")
    if (labels.some((l) => l.kind === "final")) warnings.push("both Action and Final Answer present; taking the Action")

    // Tool name: inline text, or the next line when the model wrapped early
    // (never a line that is itself a label or the start of the input JSON).
    const nextLabel = labels.find((l) => l.index > action.index)
    const fallback = nextLabel?.index !== action.index + 1 ? (lines[action.index + 1] ?? "").trim() : ""
    const nameSource = action.inline.trim() || (fallback.startsWith("{") ? "" : fallback)
    const toolName = nameSource.replace(/[`*]/g, "").split(/\s/)[0] ?? ""

    const input = labels.find((l) => l.kind === "input" && l.index >= action.index)
    if (!input) warnings.push("Action without Action Input; using empty object")

    const thought = labels.find((l) => l.kind === "thought" && l.index < action.index)
    const leading = lines
      .slice(0, labels[0]?.index ?? 0)
      .join("\n")
      .trim()

    return {
      kind: "action",
      leading: leading || undefined,
      thought: thought ? textUntilNextLabel(thought) || undefined : undefined,
      toolName,
      input: input ? resolveInput(textUntilNextLabel(input), schemas?.[toolName], warnings) : "{}",
      warnings,
    }
  }

  const final = labels.find((l) => l.kind === "final")
  if (final) {
    // Everything from the label to the end is the answer, verbatim — including
    // any stray labels the model emitted afterwards.
    const answer = [final.inline, ...lines.slice(final.index + 1)].join("\n").trim()
    // Preceding prose and Thought content (label stripped) stays visible.
    const before = lines
      .slice(0, final.index)
      .map((l) => l.match(LABEL)?.[2] ?? l)
      .join("\n")
      .trim()
    return { kind: "final", answer: before ? `${before}\n\n${answer}` : answer, warnings }
  }

  return { kind: "text", text: raw.trim(), warnings }
}
