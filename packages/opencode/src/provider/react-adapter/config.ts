/**
 * Activation and tuning for the ReAct adapter.
 *
 * The provider is declared as a normal `@ai-sdk/openai-compatible` provider in
 * opencode.json; the adapter activates via `"reactMode": true` inside that
 * provider's `options` block (alongside `baseURL` and `apiKey`). Optional
 * tuning lives in an `"react": { ... }` object in the same options block —
 * every field has a default so an empty object works:
 *
 * ```json
 * {
 *   "provider": {
 *     "corp": {
 *       "npm": "@ai-sdk/openai-compatible",
 *       "options": {
 *         "baseURL": "https://host/v1",
 *         "apiKey": "{env:CORP_KEY}",
 *         "reactMode": true,
 *         "react": { "resultPrefix": "TOOL_RESULT", "retries": 2, "caps": { "read": 16000 } }
 *       }
 *     }
 *   }
 * }
 * ```
 *
 * `"mode"` in the react block selects the transcript layout. Background: the
 * corporate STARK shim keeps a server-side conversation keyed on the content
 * of the FIRST message and forwards only its stored history plus the LAST
 * message of each request, ignoring the rest of the body. Rewind therefore
 * never reaches the model and the store has a hidden retention wall.
 *
 * - `"alternating"` (default): plain user/assistant messages, preamble in the
 *   first user message. Fastest on a well-behaved endpoint; broken on STARK
 *   after a rewind or ~100 messages.
 * - `"single"`: every request is ONE user message (nonce, preamble, labeled
 *   turns). First and last message are the same, so the shim's store can never
 *   diverge. Always correct, but the whole transcript is prefilled on every
 *   call. `"singleMessage": true` is an alias.
 * - `"epoch"`: alternating messages whose first message is a fixed SNAPSHOT of
 *   the transcript (nonce first), created by a single-message "re-seed" request.
 *   The shim caches the epoch; only the re-seed pays full prefill. The adapter
 *   re-seeds whenever the snapshot no longer matches the current history
 *   (rewind, compaction, prune, system-prompt change), when `epochTurns` new
 *   turns have accumulated, or when the persisted epoch is cleared (manual
 *   /reseed). Needs opencode to persist the epoch per session; without that
 *   plumbing every call re-seeds, i.e. behaves like `"single"`.
 */
export * as ReactAdapterConfig from "./config"

export type Mode = "alternating" | "single" | "epoch"

export const MODES: readonly Mode[] = ["alternating", "single", "epoch"]

export type Config = {
  /** Prefix for injected tool results. Deliberately NOT "Observation" — that label is high-frequency ReAct training data and participates in echo/recitation failures. */
  resultPrefix: string
  /** Retries for empty responses (the laundered recitation case) before surfacing an error. */
  retries: number
  /** Per-tool truncation cap overrides in characters; the "default" key overrides the fallback cap. */
  caps: Record<string, number>
  /** Transcript layout (see module doc). Default "alternating"; `singleMessage: true` is an alias for "single". */
  mode: Mode
  /** Epoch mode: re-seed once this many transcript turns have accumulated since the last seed; 0 disables the periodic re-seed. Default 40. */
  epochTurns: number
}

export function isMode(value: unknown): value is Mode {
  return typeof value === "string" && (MODES as readonly string[]).includes(value)
}

export function resolve(options: Record<string, unknown> | undefined): Config | undefined {
  if (options?.["reactMode"] !== true) return undefined
  const react = (
    typeof options["react"] === "object" && options["react"] !== null ? options["react"] : {}
  ) as Record<string, unknown>
  return {
    resultPrefix: typeof react["resultPrefix"] === "string" ? react["resultPrefix"] : "TOOL_RESULT",
    retries: typeof react["retries"] === "number" ? react["retries"] : 2,
    caps:
      typeof react["caps"] === "object" && react["caps"] !== null ? (react["caps"] as Record<string, number>) : {},
    mode: isMode(react["mode"]) ? react["mode"] : react["singleMessage"] === true ? "single" : "alternating",
    epochTurns:
      typeof react["epochTurns"] === "number" && react["epochTurns"] >= 0 ? Math.floor(react["epochTurns"]) : 40,
  }
}
