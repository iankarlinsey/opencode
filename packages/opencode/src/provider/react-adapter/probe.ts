#!/usr/bin/env bun
/**
 * Standalone endpoint probe for the ReAct adapter (Phase 0).
 *
 * Deliberately self-contained (no imports from the adapter) so this single
 * file can be copied onto a locked-down machine and run there.
 *
 * Usage:
 *   REACT_PROBE_BASE_URL=https://host/v1 \
 *   REACT_PROBE_API_KEY=... \
 *   REACT_PROBE_MODEL=gemini-3.1-pro-preview \
 *   bun run packages/opencode/src/provider/react-adapter/probe.ts [--write]
 *
 * Prints a markdown report to stdout. With --write, updates the Results
 * section of PROBES.md next to this file.
 */

const baseURL = process.env["REACT_PROBE_BASE_URL"]
const apiKey = process.env["REACT_PROBE_API_KEY"]
const model = process.env["REACT_PROBE_MODEL"] ?? "gemini-3.1-pro-preview"

if (!baseURL || !apiKey) {
  console.error("Set REACT_PROBE_BASE_URL and REACT_PROBE_API_KEY (and optionally REACT_PROBE_MODEL)")
  process.exit(1)
}

type Message = { role: string; content: string }

type ProbeResponse = {
  status: number
  finishReason?: string
  content?: string
  error?: string
  raw: unknown
}

async function chat(messages: Message[], extra?: Record<string, unknown>): Promise<ProbeResponse> {
  const res = await fetch(`${baseURL!.replace(/\/$/, "")}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      temperature: 0,
      ...extra,
    }),
  })
  const body: any = await res.json().catch(() => undefined)
  if (!res.ok) return { status: res.status, error: JSON.stringify(body ?? "(no body)"), raw: body }
  const choice = body?.choices?.[0]
  return {
    status: res.status,
    finishReason: choice?.finish_reason,
    content: choice?.message?.content ?? "",
    raw: body,
  }
}

function show(r: ProbeResponse) {
  if (r.error) return `HTTP ${r.status}: ${r.error.slice(0, 400)}`
  const content = r.content ?? ""
  const preview = content.length > 300 ? content.slice(0, 300) + "…" : content
  return `HTTP ${r.status}, finish_reason=${JSON.stringify(r.finishReason)}, content(${content.length} chars)=${JSON.stringify(preview)}`
}

// Canonical `ls -la` output — this is the exact shape that trips the
// recitation filter (verified empirically). Kept verbatim on purpose.
const RAW_LS = [
  "total 48",
  "drwxr-xr-x  6 user user 4096 Jan  5 10:32 .",
  "drwxr-xr-x 14 user user 4096 Jan  5 10:30 ..",
  "-rw-r--r--  1 user user  220 Jan  5 10:30 .bash_logout",
  "-rw-r--r--  1 user user 3771 Jan  5 10:30 .bashrc",
  "drwxr-xr-x  3 user user 4096 Jan  5 10:31 .config",
  "-rw-r--r--  1 user user  807 Jan  5 10:30 .profile",
  "drwxr-xr-x  2 user user 4096 Jan  5 10:32 docs",
  "-rw-r--r--  1 user user 1204 Jan  5 10:32 package.json",
  "drwxr-xr-x  4 user user 4096 Jan  5 10:32 src",
].join("\n")

// Mirrors the ls de-canonicalizer in ./normalize.ts (kept inline so this
// script stays copy-paste standalone). If you change one, change both.
function decanonicalizeLs(text: string): string {
  const lines = text.split("\n")
  const out: string[] = []
  for (const line of lines) {
    if (/^total \d+$/.test(line)) continue
    const m = line.match(/^([bcdlps-])[rwxsStT-]{9}[@+.]?\s+\d+\s+\S+\s+\S+\s+(\d+)\s+\S+\s+\S+\s+\S+\s+(.+)$/)
    if (!m) {
      out.push(line)
      continue
    }
    const kind = m[1] === "d" ? "directory" : m[1] === "l" ? "symlink" : "file"
    out.push(kind === "directory" ? `  ${m[3]}/ (directory)` : `  ${m[3]} (${kind}, ${m[2]} bytes)`)
  }
  return out.join("\n")
}

type Verdict = { id: string; title: string; verdict: string; detail: string }
const results: Verdict[] = []

// Probe 1: system role — stripped silently or hard error?
async function probeSystemRole() {
  const r = await chat([
    { role: "system", content: "You must begin every reply with the exact token SYS-MARKER-7Q." },
    { role: "user", content: "Say hello." },
  ])
  let verdict: string
  if (r.error) verdict = `HARD ERROR (${r.status}) — system role rejected; adapter must never emit it`
  else if ((r.content ?? "").includes("SYS-MARKER-7Q")) verdict = "DELIVERED — system role reaches the model"
  else verdict = "STRIPPED SILENTLY — 200 OK but the system instruction had no effect"
  results.push({ id: "1", title: "system role handling", verdict, detail: show(r) })
}

// Probe 2: consecutive user messages — delivered, merged, or error?
async function probeConsecutiveUser() {
  const r = await chat([
    { role: "user", content: "Remember this code word and nothing else: ZEBRA-42." },
    { role: "user", content: "What is the code word? Reply with the code word only." },
  ])
  let verdict: string
  if (r.error) verdict = `HARD ERROR (${r.status}) — strict alternation enforced; adapter must merge same-role messages`
  else if ((r.content ?? "").includes("ZEBRA-42")) verdict = "DELIVERED — both user messages reached the model"
  else verdict = "DEGRADED — 200 OK but earlier user message apparently lost; adapter must merge same-role messages"
  results.push({ id: "2", title: "consecutive user messages", verdict, detail: show(r) })
}

// Probe 3: response that would BEGIN with a stop sequence.
async function probeStopAtStart() {
  const r = await chat(
    [
      {
        role: "user",
        content: 'Reply with exactly the following two lines and nothing else:\nSTOPWORD-XY: alpha\nsecond line',
      },
    ],
    { stop: ["STOPWORD-XY:"] },
  )
  let verdict: string
  if (r.error) verdict = `HARD ERROR (${r.status})`
  else if ((r.content ?? "") === "")
    verdict =
      'EMPTY + finish_reason=' +
      JSON.stringify(r.finishReason) +
      " — a reply beginning with a stop sequence is indistinguishable from the recitation-empty case; adapter must treat both as retryable"
  else verdict = "NON-EMPTY — stop-at-start returns partial/other content; see detail"
  results.push({ id: "3", title: "stop sequence at response start", verdict, detail: show(r) })
}

// Probe 4: recitation repro — raw ls -la vs de-canonicalized listing.
async function probeRecitation() {
  const ask = "Summarize the directory listing above in one sentence."
  const raw = await chat([{ role: "user", content: `Here is a directory listing:\n\n${RAW_LS}\n\n${ask}` }])
  const normalized = await chat([
    { role: "user", content: `Here is a directory listing:\n\n${decanonicalizeLs(RAW_LS)}\n\n${ask}` },
  ])
  const rawEmpty = !raw.error && (raw.content ?? "") === ""
  const normEmpty = !normalized.error && (normalized.content ?? "") === ""
  let verdict: string
  if (rawEmpty && !normEmpty) verdict = "REPRODUCED + MITIGATED — raw ls output → empty; normalized output → answer"
  else if (rawEmpty && normEmpty) verdict = "REPRODUCED, NOT MITIGATED — normalization did not help; needs a different strategy"
  else if (!rawEmpty) verdict = "NOT REPRODUCED — raw ls output answered fine this run (flaky filter? rerun a few times)"
  else verdict = "UNCLEAR"
  results.push({
    id: "4",
    title: "recitation: raw vs normalized ls output",
    verdict,
    detail: `raw: ${show(raw)}\n  normalized: ${show(normalized)}`,
  })
}

await probeSystemRole()
await probeConsecutiveUser()
await probeStopAtStart()
await probeRecitation()

const report = [
  `## Probe run ${new Date().toISOString()} — model \`${model}\``,
  "",
  ...results.flatMap((r) => [`### Probe ${r.id}: ${r.title}`, "", `**Verdict:** ${r.verdict}`, "", `Detail: ${r.detail}`, ""]),
].join("\n")

console.log(report)

if (process.argv.includes("--write")) {
  const path = new URL("./PROBES.md", import.meta.url).pathname
  const existing = await Bun.file(path).text()
  const marker = "<!-- probe-results -->"
  const idx = existing.indexOf(marker)
  const head = idx >= 0 ? existing.slice(0, idx + marker.length) : existing + "\n" + marker
  await Bun.write(path, head + "\n\n" + report + "\n")
  console.error(`\nWrote results to ${path}`)
}
