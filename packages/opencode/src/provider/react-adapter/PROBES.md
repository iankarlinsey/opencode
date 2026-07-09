# Endpoint probes (Phase 0)

The corporate endpoint is an OpenAI-compatible chat-completions shim in front of
Gemini (`gemini-3.1-pro-preview` via a Vertex AI enterprise shim). The adapter's
design rests on four behavioral questions that `probe.ts` answers empirically.

Run:

```sh
REACT_PROBE_BASE_URL=https://<host>/v1 \
REACT_PROBE_API_KEY=<key> \
REACT_PROBE_MODEL=gemini-3.1-pro-preview \
bun run packages/opencode/src/provider/react-adapter/probe.ts --write
```

`--write` appends the run's results below the marker at the bottom of this file.
`probe.ts` is intentionally self-contained (no imports) so it can be copied
alone onto the machine that has network access to the endpoint.

## What each probe decides

| # | Question | Adapter decision it gates |
|---|----------|---------------------------|
| 1 | Is a `system` message rejected (4xx) or silently stripped? | None — the adapter never emits `system` either way. The result only tells us how loudly a regression would fail. |
| 2 | Are two consecutive `user` messages delivered, merged, or an error? | None — the adapter always merges consecutive same-role messages, which is safe under every observed outcome. Confirms the merge is *required* (error/degraded) or merely *harmless* (delivered). |
| 3 | Does a reply that would begin with a stop sequence come back as empty content + `finish_reason:"stop"`? | If yes, an empty response is ambiguous between "recitation abort" and "stopped at token zero" — both are handled by the same retry-with-nudge path, so the adapter treats every empty response as recoverable. |
| 4 | Does raw `ls -la` output in context reproduce the empty-response recitation failure, and does the de-canonicalized format fix it? | Validates the Phase 2 normalization strategy. If NOT mitigated, escalate: line-number prefixing for all bash output, or stronger rewriting. |

## Defensive posture taken regardless of probe outcomes

These are baked into the adapter because they are safe under every possible
probe result:

- Only `user` and `assistant` roles are ever emitted.
- Consecutive same-role messages are merged with `\n\n`.
- Empty responses are never treated as completion; they trigger up to N retries
  (default 2) with a format nudge appended to the last user message, then a
  distinguishable `react-adapter` error — never a silent stop.
- Tool results are normalized (truncated on line boundaries, listing-like
  output de-canonicalized) before injection.
- The tool-result prefix is `TOOL_RESULT`, not `Observation:`, to stay away
  from high-frequency ReAct training data; `\nObservation:` is still included
  as a stop sequence so a model that regresses into classic ReAct cannot
  fabricate observations.

## Results

Results from actual runs are appended below by `probe.ts --write`.
Status: **PENDING — not yet run against the endpoint** (requires network access
to the corporate shim).

<!-- probe-results -->
