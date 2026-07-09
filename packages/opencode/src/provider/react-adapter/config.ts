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
 */
export * as ReactAdapterConfig from "./config"

export type Config = {
  /** Prefix for injected tool results. Deliberately NOT "Observation" — that label is high-frequency ReAct training data and participates in echo/recitation failures. */
  resultPrefix: string
  /** Retries for empty responses (the laundered recitation case) before surfacing an error. */
  retries: number
  /** Per-tool truncation cap overrides in characters; the "default" key overrides the fallback cap. */
  caps: Record<string, number>
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
  }
}
