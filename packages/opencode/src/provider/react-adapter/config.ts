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
 * `"singleMessage": true` in the react block changes the transcript layout:
 * instead of alternating user/assistant messages, every request carries the
 * whole conversation (nonce, preamble, labeled turns) inside ONE user
 * message. Use it against endpoints that keep a server-side conversation
 * keyed on the first message and forward only the last message of each
 * request (observed with the corporate STARK shim): with a single message,
 * "first" and "last" are the same message, so the endpoint's store can never
 * diverge from opencode's history — rewind works and there is no hidden
 * retention limit. Cost: no server-side reuse, one large message per call.
 */
export * as ReactAdapterConfig from "./config"

export type Config = {
  /** Prefix for injected tool results. Deliberately NOT "Observation" — that label is high-frequency ReAct training data and participates in echo/recitation failures. */
  resultPrefix: string
  /** Retries for empty responses (the laundered recitation case) before surfacing an error. */
  retries: number
  /** Per-tool truncation cap overrides in characters; the "default" key overrides the fallback cap. */
  caps: Record<string, number>
  /** Send the entire transcript as one user message per request (see module doc). Default false. */
  singleMessage: boolean
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
    singleMessage: react["singleMessage"] === true,
  }
}
