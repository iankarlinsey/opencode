import { ProviderTransform } from "@/provider/transform"

/**
 * Tool output normalization (recitation mitigation).
 *
 * The provider's underlying Gemini model aborts with an empty response when the
 * context contains verbatim canonical-format text (raw `ls -la` columns being
 * the confirmed case). Before any tool result is injected into the transcript
 * we (1) rewrite listing-like output into a novel shape, (2) optionally prefix
 * lines with `N| ` as a cheap general de-canonicalizer for other rigid formats
 * (git status, stack traces), and (3) truncate on line boundaries with an
 * informative marker.
 */

export type Strategy = {
  /** Maximum output size in characters; truncation happens on line boundaries. */
  cap: number
  /** Rewrite `ls -l`-style listings into an indented name list. */
  decanonicalizeListings?: boolean
  /** Prefix every line with `N| ` when the output looks like a rigid canonical format. */
  lineNumbers?: boolean
}

const DEFAULT_STRATEGY: Strategy = { cap: 4_000, decanonicalizeListings: true, lineNumbers: true }

const STRATEGIES: Record<string, Strategy> = {
  bash: { cap: 4_000, decanonicalizeListings: true, lineNumbers: true },
  read: { cap: 16_000 },
  list: { cap: 4_000, decanonicalizeListings: true },
  grep: { cap: 4_000 },
  glob: { cap: 4_000 },
}

// `-rw-r--r--  1 user group  1234 Jan  5 10:32 name` (a trailing @/+/. after
// the permission string covers macOS xattr/ACL markers)
const LS_LINE = /^([bcdlps-])[rwxsStT-]{9}[@+.]?\s+\d+\s+\S+(\s+\S+)?\s+(\d+)\s+\S+\s+\S+\s+\S+\s+(.+)$/

function looksLikeListing(lines: string[]) {
  if (/^total \d+$/.test(lines[0] ?? "")) return true
  const matching = lines.filter((l) => LS_LINE.test(l)).length
  return lines.length > 0 && matching >= Math.max(2, Math.ceil(lines.length / 2))
}

function decanonicalizeListing(lines: string[]) {
  return lines.flatMap((line) => {
    if (/^total \d+$/.test(line) || line === "") return []
    const m = line.match(LS_LINE)
    if (!m) return [line]
    const kind = m[1] === "d" ? "directory" : m[1] === "l" ? "symlink" : "file"
    if (kind === "directory") return [`  ${m[4]}/ (directory)`]
    return [`  ${m[4]} (${kind}, ${m[3]} bytes)`]
  })
}

// Rigid formats that are likely verbatim training data: `git status` porcelain
// and long-form headers, and Java/Node stack traces.
const RIGID_LINE = /^(\s+at\s+\S+ \(.+\)$|[ MADRCU?!]{2} \S|On branch \S+|Your branch is|Changes (not staged|to be committed)|Untracked files:)/

function looksRigid(lines: string[]) {
  const matching = lines.filter((l) => RIGID_LINE.test(l)).length
  return matching >= Math.max(2, Math.ceil(lines.length / 3))
}

export function normalize(toolName: string, output: string, overrides?: Record<string, number>): string {
  const base = STRATEGIES[toolName] ?? DEFAULT_STRATEGY
  const cap = overrides?.[toolName] ?? overrides?.["default"] ?? base.cap
  let lines = ProviderTransform.sanitizeSurrogates(output).split("\n")

  if (base.decanonicalizeListings && looksLikeListing(lines)) {
    lines = decanonicalizeListing(lines)
  } else if (base.lineNumbers && looksRigid(lines)) {
    lines = lines.map((l, i) => `${i + 1}| ${l}`)
  }

  const total = lines.length
  const kept: string[] = []
  let size = 0
  for (const line of lines) {
    // A single line larger than the whole cap is cut on a code-point boundary
    // (Array.from splits by code point, never inside a surrogate pair).
    const chunk = line.length > cap ? Array.from(line).slice(0, cap).join("") : line
    if (size + chunk.length + 1 > cap && kept.length > 0) break
    kept.push(chunk)
    size += chunk.length + 1
  }
  if (kept.length === total && kept[total - 1] === lines[total - 1]) return kept.join("\n")
  return (
    kept.join("\n") +
    `\n[truncated: showing ${kept.length} of ${total} lines — request again with offset/limit for more]`
  )
}
