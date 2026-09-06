import { describe, expect, test } from "bun:test"
import type { LanguageModelV3CallOptions } from "@ai-sdk/provider"
import { parse } from "@/provider/react-adapter/parse"
import { normalize } from "@/provider/react-adapter/normalize"
import { downconvert, type Epoch } from "@/provider/react-adapter/downconvert"
import { ReactAdapterConfig } from "@/provider/react-adapter/config"
import { ReactAdapter } from "@/provider/react-adapter"

const SCHEMAS = {
  bash: {
    type: "object" as const,
    properties: { command: { type: "string" as const } },
    required: ["command"],
  },
  read: {
    type: "object" as const,
    properties: { filePath: { type: "string" as const }, offset: { type: "number" as const } },
    required: ["filePath"],
  },
}

describe("react-adapter parse", () => {
  test("valid block", () => {
    const parsed = parse("Thought: list files first.\nAction: bash\nAction Input: {\"command\": \"ls\"}", SCHEMAS)
    expect(parsed).toMatchObject({ kind: "action", thought: "list files first.", toolName: "bash" })
    expect(JSON.parse((parsed as any).input)).toEqual({ command: "ls" })
  })

  test("missing Thought", () => {
    const parsed = parse('Action: read\nAction Input: {"filePath": "a.ts"}', SCHEMAS)
    expect(parsed).toMatchObject({ kind: "action", toolName: "read", thought: undefined })
  })

  test("fenced JSON input", () => {
    const parsed = parse('Action: read\nAction Input:\n```json\n{"filePath": "a.ts",\n "offset": 5}\n```', SCHEMAS)
    expect(parsed.kind).toBe("action")
    expect(JSON.parse((parsed as any).input)).toEqual({ filePath: "a.ts", offset: 5 })
  })

  test("bare-string input wrapped into single required string param", () => {
    const parsed = parse("Action: bash\nAction Input: ls -la src", SCHEMAS)
    expect(parsed.kind).toBe("action")
    expect(JSON.parse((parsed as any).input)).toEqual({ command: "ls -la src" })
    expect((parsed as any).warnings.join()).toContain("bare-string")
  })

  test("bare-string input with multiple required params passes through raw", () => {
    const schemas = {
      edit: {
        type: "object" as const,
        properties: { path: { type: "string" as const }, text: { type: "string" as const } },
        required: ["path", "text"],
      },
    }
    const parsed = parse("Action: edit\nAction Input: not json at all", schemas)
    expect(parsed.kind).toBe("action")
    expect((parsed as any).input).toBe("not json at all")
  })

  test("malformed JSON passes through raw for repair, never becomes a bare string", () => {
    const parsed = parse('Action: bash\nAction Input: {"command": "echo hi",}', SCHEMAS)
    expect(parsed.kind).toBe("action")
    // trailing comma is invalid JSON; it must reach repairToolCall raw instead
    // of being wrapped as a shell command
    expect((parsed as any).input).toBe('{"command": "echo hi",}')
    expect((parsed as any).warnings.join()).toContain("not parseable JSON")
  })

  test("multiple Actions takes the first and warns", () => {
    const parsed = parse(
      'Action: bash\nAction Input: {"command": "ls"}\nAction: read\nAction Input: {"filePath": "x"}',
      SCHEMAS,
    )
    expect(parsed).toMatchObject({ kind: "action", toolName: "bash" })
    expect((parsed as any).warnings.join()).toContain("multiple Action")
  })

  test("Final Answer", () => {
    const parsed = parse("Thought: done now.\nFinal Answer: it is 42.\nSecond line.")
    expect(parsed.kind).toBe("final")
    expect((parsed as any).answer).toBe("done now.\n\nit is 42.\nSecond line.")
  })

  test("Action wins over Final Answer and warns", () => {
    const parsed = parse('Action: bash\nAction Input: {"command": "ls"}\nFinal Answer: done', SCHEMAS)
    expect(parsed).toMatchObject({ kind: "action", toolName: "bash" })
    expect((parsed as any).warnings.join()).toContain("both Action and Final Answer")
  })

  test("empty string", () => {
    expect(parse("")).toMatchObject({ kind: "text", text: "" })
  })

  test("plain prose with no labels", () => {
    expect(parse("Just a normal reply.")).toMatchObject({ kind: "text", text: "Just a normal reply." })
  })

  test("text then Action keeps leading text", () => {
    const parsed = parse('Let me check.\nAction: bash\nAction Input: {"command": "ls"}', SCHEMAS)
    expect(parsed).toMatchObject({ kind: "action", toolName: "bash", leading: "Let me check." })
  })

  test("Action embedded mid-prose is not a label", () => {
    const parsed = parse("I think the right Action: bash here would be to wait.")
    expect(parsed.kind).toBe("text")
  })

  test("whole reply wrapped in a markdown fence", () => {
    const parsed = parse('```\nThought: ok.\nAction: bash\nAction Input: {"command": "ls"}\n```', SCHEMAS)
    expect(parsed).toMatchObject({ kind: "action", toolName: "bash" })
  })

  test("Action without Action Input yields empty object and warns", () => {
    const parsed = parse("Action: bash", SCHEMAS)
    expect(parsed).toMatchObject({ kind: "action", toolName: "bash", input: "{}" })
    expect((parsed as any).warnings.join()).toContain("without Action Input")
  })
})

describe("react-adapter normalize", () => {
  const LS = [
    "total 48",
    "drwxr-xr-x  6 user user 4096 Jan  5 10:32 .",
    "drwxr-xr-x 14 user user 4096 Jan  5 10:30 ..",
    "-rw-r--r--  1 user user 3771 Jan  5 10:30 .bashrc",
    "lrwxrwxrwx  1 user user   11 Jan  5 10:31 latest -> ./releases",
    "drwxr-xr-x  4 user user 4096 Jan  5 10:32 src",
  ].join("\n")

  test("de-canonicalizes ls -la output", () => {
    const out = normalize("bash", LS)
    expect(out).not.toContain("total 48")
    expect(out).not.toContain("drwxr-xr-x")
    expect(out).toContain(".bashrc (file, 3771 bytes)")
    expect(out).toContain("src/ (directory)")
    expect(out).toContain("latest -> ./releases (symlink, 11 bytes)")
  })

  test("truncates on line boundaries with informative marker", () => {
    const input = Array.from({ length: 500 }, (_, i) => `unique project line ${i}`).join("\n")
    const out = normalize("bash", input)
    expect(out.length).toBeLessThan(4_200)
    expect(out).toMatch(/\[truncated: showing \d+ of 500 lines — request again with offset\/limit for more\]$/)
    const lastKept = out.split("\n").at(-2)
    expect(lastKept).toMatch(/^unique project line \d+$/)
  })

  test("read cap is larger and configurable via overrides", () => {
    const input = Array.from({ length: 1000 }, (_, i) => `source line ${i}`).join("\n")
    expect(normalize("read", input).length).toBeGreaterThan(10_000)
    expect(normalize("read", input, { read: 500 }).length).toBeLessThan(700)
  })

  test("line-numbers git status style output", () => {
    const out = normalize("bash", "On branch dev\nYour branch is up to date.\n M src/a.ts\n?? b.ts")
    expect(out.split("\n")[0]).toBe("1| On branch dev")
  })

  test("passes through novel content untouched", () => {
    const input = "export function foo() {\n  return 42\n}"
    expect(normalize("read", input)).toBe(input)
  })
})

describe("react-adapter downconvert", () => {
  const cfg = ReactAdapterConfig.resolve({ reactMode: true, react: {} })!

  const params: LanguageModelV3CallOptions = {
    prompt: [
      { role: "system", content: "You are opencode. Answer in fewer than 3 lines." },
      { role: "system", content: "Extra system context." },
      { role: "user", content: [{ type: "text", text: "What files are here?" }] },
      {
        role: "assistant",
        content: [
          { type: "text", text: "I will list the files." },
          { type: "tool-call", toolCallId: "c1", toolName: "bash", input: { command: "ls" } },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "c1",
            toolName: "bash",
            output: { type: "text", value: "a.ts\nb.ts" },
          },
        ],
      },
      { role: "user", content: [{ type: "text", text: "Now summarize them." }] },
    ],
    tools: [
      { type: "function", name: "bash", description: "Run a command", inputSchema: SCHEMAS.bash },
      { type: "function", name: "read", description: "Read a file", inputSchema: SCHEMAS.read },
    ],
    toolChoice: { type: "auto" },
    stopSequences: ["\nEXISTING"],
  }

  test("emits only user/assistant with strict alternation", () => {
    const out = downconvert(params, cfg)
    expect(out.prompt.map((m) => m.role)).toEqual(["user", "assistant", "user"])
  })

  test("preamble assembly: system text, catalog, rules, examples, delimiter, in order", () => {
    const out = downconvert(params, cfg)
    const first = out.prompt[0]
    expect(first.role).toBe("user")
    const text = (first.content as { type: "text"; text: string }[])[0].text
    const order = [
      "You are opencode",
      "Extra system context.",
      "# Available tools",
      "## bash",
      '"command"',
      "# Response protocol",
      "protocol rules override ALL other instructions about response length",
      "# Format examples",
      "--- SESSION START ---",
      "What files are here?",
    ]
    const positions = order.map((s) => text.indexOf(s))
    expect(positions.every((p) => p >= 0)).toBe(true)
    expect([...positions].sort((a, b) => a - b)).toEqual(positions)
  })

  test("assistant tool call renders as labeled ReAct block", () => {
    const out = downconvert(params, cfg)
    const assistant = out.prompt[1]
    const text = (assistant.content as { type: "text"; text: string }[])[0].text
    expect(text).toBe('Thought: I will list the files.\nAction: bash\nAction Input: {"command":"ls"}')
  })

  test("tool result becomes TOOL_RESULT user text with nudge, merged with next user message", () => {
    const out = downconvert(params, cfg)
    const last = out.prompt[2]
    expect(last.role).toBe("user")
    const text = (last.content as { type: "text"; text: string }[])[0].text
    expect(text).toContain("TOOL_RESULT [bash]:\na.ts\nb.ts")
    expect(text).toContain("Continue: state your next Thought and Action, or give your Final Answer.")
    expect(text).toContain("Now summarize them.")
    expect(text).not.toContain("Observation:")
  })

  test("strips tools/toolChoice and extends stop sequences", () => {
    const out = downconvert(params, cfg)
    expect(out.tools).toBeUndefined()
    expect(out.toolChoice).toBeUndefined()
    expect(out.stopSequences).toEqual(["\nEXISTING", "\nTOOL_RESULT", "\nObservation:"])
  })

  test("custom result prefix flows through", () => {
    const custom = ReactAdapterConfig.resolve({ reactMode: true, react: { resultPrefix: "RESULT_BLOCK" } })!
    const out = downconvert(params, custom)
    const text = (out.prompt[2].content as { type: "text"; text: string }[])[0].text
    expect(text).toContain("RESULT_BLOCK [bash]:")
    expect(out.stopSequences).toContain("\nRESULT_BLOCK")
  })
})

describe("react-adapter downconvert single-message mode", () => {
  const single = ReactAdapterConfig.resolve({ reactMode: true, react: { singleMessage: true } })!
  const alternating = ReactAdapterConfig.resolve({ reactMode: true, react: {} })!
  const nonce = () => "NONCE-1234"
  const textOf = (out: LanguageModelV3CallOptions, i = 0) =>
    (out.prompt[i].content as { type: "text"; text: string }[])[0].text

  const params: LanguageModelV3CallOptions = {
    prompt: [
      { role: "system", content: "You are opencode. Answer in fewer than 3 lines." },
      { role: "user", content: [{ type: "text", text: "What files are here?" }] },
      {
        role: "assistant",
        content: [
          { type: "text", text: "I will list the files." },
          { type: "tool-call", toolCallId: "c1", toolName: "bash", input: { command: "ls" } },
        ],
      },
      {
        role: "tool",
        content: [{ type: "tool-result", toolCallId: "c1", toolName: "bash", output: { type: "text", value: "a.ts\nb.ts" } }],
      },
      { role: "user", content: [{ type: "text", text: "Now summarize them." }] },
    ],
    tools: [{ type: "function", name: "bash", description: "Run a command", inputSchema: SCHEMAS.bash }],
    toolChoice: { type: "auto" },
    stopSequences: ["\nEXISTING"],
  }

  test("emits exactly one user message", () => {
    const out = downconvert(params, single, { nonce })
    expect(out.prompt.map((m) => m.role)).toEqual(["user"])
  })

  test("nonce first, then preamble, transcript rules, delimiter, labeled turns in order, end marker", () => {
    const text = textOf(downconvert(params, single, { nonce }))
    expect(text.startsWith("Request NONCE-1234\n")).toBe(true)
    const order = [
      "Request NONCE-1234",
      "You are opencode",
      "# Available tools",
      "## bash",
      "# Response protocol",
      "# Format examples",
      "# Transcript format",
      "--- SESSION START ---",
      "[USER]\nWhat files are here?",
      '[ASSISTANT]\nThought: I will list the files.\nAction: bash\nAction Input: {"command":"ls"}',
      "[USER]\nTOOL_RESULT [bash]:\na.ts\nb.ts",
      "Continue: state your next Thought and Action, or give your Final Answer.",
      "Now summarize them.",
      "--- END OF TRANSCRIPT ---",
    ]
    const positions = order.map((s) => text.indexOf(s))
    expect(positions.every((p) => p >= 0)).toBe(true)
    expect([...positions].sort((a, b) => a - b)).toEqual(positions)
    // tool result and the following user message stay one merged [USER] turn
    // (count label lines after the delimiter; the rules paragraph mentions the labels too)
    const body = text.slice(text.indexOf("--- SESSION START ---"))
    expect(body.split("\n[USER]\n").length - 1).toBe(2)
    expect(body.split("\n[ASSISTANT]\n").length - 1).toBe(1)
  })

  test("differs between calls only by the nonce", () => {
    const a = textOf(downconvert(params, single, { nonce: () => "A" }))
    const b = textOf(downconvert(params, single, { nonce: () => "B" }))
    expect(a).not.toBe(b)
    expect(a.replace("Request A", "Request X")).toBe(b.replace("Request B", "Request X"))
  })

  test("strips tools/toolChoice and extends stop sequences like alternating mode", () => {
    const out = downconvert(params, single, { nonce })
    expect(out.tools).toBeUndefined()
    expect(out.toolChoice).toBeUndefined()
    expect(out.stopSequences).toEqual(["\nEXISTING", "\nTOOL_RESULT", "\nObservation:"])
  })

  test("no user turn at all still yields one message ending with the end marker", () => {
    const out = downconvert({ ...params, prompt: [params.prompt[0]] }, single, { nonce })
    expect(out.prompt).toHaveLength(1)
    expect(textOf(out).endsWith(END_MARKER)).toBe(true)
  })

  test("default config is unaffected (alternating layout)", () => {
    const out = downconvert(params, alternating, { nonce })
    expect(out.prompt.map((m) => m.role)).toEqual(["user", "assistant", "user"])
    expect(textOf(out)).not.toContain("Request NONCE-1234")
    expect(textOf(out)).not.toContain("[USER]")
  })

  test("rewind: output derives only from the current prompt, in both modes", () => {
    const A = params.prompt.slice(0, 2)
    const B: LanguageModelV3CallOptions["prompt"] = [
      { role: "assistant", content: [{ type: "text", text: "UNIQUE-B-TURN" }] },
      { role: "user", content: [{ type: "text", text: "UNIQUE-B-USER" }] },
    ]
    const C: LanguageModelV3CallOptions["prompt"] = [
      { role: "assistant", content: [{ type: "text", text: "UNIQUE-C-TURN" }] },
      { role: "user", content: [{ type: "text", text: "UNIQUE-C-USER" }] },
    ]
    const NEW: LanguageModelV3CallOptions["prompt"] = [
      { role: "assistant", content: [{ type: "text", text: "Final Answer: ok" }] },
      { role: "user", content: [{ type: "text", text: "UNIQUE-NEW" }] },
    ]
    for (const cfg of [single, alternating]) {
      const full = downconvert({ ...params, prompt: [...A, ...B, ...C] }, cfg, { nonce })
      const rewound = downconvert({ ...params, prompt: [...A, ...NEW] }, cfg, { nonce })
      const fullText = full.prompt.map((m, i) => textOf(full, i)).join("\n")
      const rewoundText = rewound.prompt.map((m, i) => textOf(rewound, i)).join("\n")
      expect(fullText).toContain("UNIQUE-C-USER")
      for (const s of ["UNIQUE-B-TURN", "UNIQUE-B-USER", "UNIQUE-C-TURN", "UNIQUE-C-USER"]) {
        expect(rewoundText).not.toContain(s)
      }
      expect(rewoundText).toContain("UNIQUE-NEW")
    }
  })
})

const END_MARKER = "--- END OF TRANSCRIPT ---\nWrite the assistant's next reply now."

describe("react-adapter downconvert epoch mode", () => {
  const epochCfg = ReactAdapterConfig.resolve({ reactMode: true, react: { mode: "epoch" } })!
  const textOf = (out: LanguageModelV3CallOptions, i = 0) =>
    (out.prompt[i].content as { type: "text"; text: string }[])[0].text
  const nonces = (...ids: string[]) => {
    const queue = [...ids]
    return () => queue.shift() ?? "NONCE-OVERFLOW"
  }

  // Seed prompt: 3 transcript turns (user, assistant action, merged tool-result+user).
  const seedPrompt: LanguageModelV3CallOptions = {
    prompt: [
      { role: "system", content: "You are opencode." },
      { role: "user", content: [{ type: "text", text: "What files are here?" }] },
      {
        role: "assistant",
        content: [
          { type: "text", text: "I will list the files." },
          { type: "tool-call", toolCallId: "c1", toolName: "bash", input: { command: "ls" } },
        ],
      },
      {
        role: "tool",
        content: [{ type: "tool-result", toolCallId: "c1", toolName: "bash", output: { type: "text", value: "a.ts\nb.ts" } }],
      },
      { role: "user", content: [{ type: "text", text: "Now summarize them." }] },
    ],
    tools: [{ type: "function", name: "bash", description: "Run a command", inputSchema: SCHEMAS.bash }],
  }
  // Two more turns after the seed: the model's reply and the next user prompt.
  const later: LanguageModelV3CallOptions["prompt"] = [
    { role: "assistant", content: [{ type: "text", text: "Final Answer: two TypeScript files." }] },
    { role: "user", content: [{ type: "text", text: "Next question." }] },
  ]
  const extended: LanguageModelV3CallOptions = { ...seedPrompt, prompt: [...seedPrompt.prompt, ...later] }

  const seed = (params = seedPrompt, cfg = epochCfg, nonce = nonces("EPOCH-1")) => {
    const seeds: Epoch[] = []
    const out = downconvert(params, cfg, { nonce, onSeed: (e) => seeds.push(e) })
    return { out, seeds }
  }

  test("no epoch: seeds with a single message and reports the epoch to persist", () => {
    const { out, seeds } = seed()
    expect(out.prompt.map((m) => m.role)).toEqual(["user"])
    expect(seeds).toHaveLength(1)
    const epoch = seeds[0]
    expect(epoch.id).toBe("EPOCH-1")
    expect(textOf(out).startsWith("Request EPOCH-1\n")).toBe(true)
    expect(epoch.snapshot).toBe(textOf(out))
    expect(epoch.covered).toBe(3)
    expect(epoch.coveredHash).toMatch(/^[0-9a-f]{64}$/)
    expect(epoch.preambleHash).toMatch(/^[0-9a-f]{64}$/)
  })

  test("matching epoch: snapshot first, then only the new turns, no re-seed", () => {
    const { seeds } = seed()
    const epoch = seeds[0]
    const reseeds: unknown[] = []
    const out = downconvert(extended, epochCfg, { epoch, nonce: nonces("EPOCH-2"), onSeed: (e) => reseeds.push(e) })
    expect(reseeds).toHaveLength(0)
    expect(out.prompt.map((m) => m.role)).toEqual(["user", "assistant", "user"])
    expect(textOf(out, 0)).toBe(epoch.snapshot)
    expect(textOf(out, 1)).toBe("Final Answer: two TypeScript files.")
    expect(textOf(out, 2)).toBe("Next question.")
    // preamble lives only in the snapshot
    const all = out.prompt.map((_, i) => textOf(out, i)).join("\n")
    expect(all.split("# Available tools").length - 1).toBe(1)
    expect(out.tools).toBeUndefined()
    expect(out.stopSequences).toEqual(["\nTOOL_RESULT", "\nObservation:"])
  })

  test("rewind (covered history changed): hash mismatch triggers re-seed with a fresh nonce", () => {
    const { seeds } = seed()
    const epoch = seeds[0]
    const rewound: LanguageModelV3CallOptions = {
      ...extended,
      prompt: extended.prompt.map((m, i) =>
        i === 1 ? { role: "user" as const, content: [{ type: "text" as const, text: "A different first prompt." }] } : m,
      ),
    }
    const reseeds: { id: string }[] = []
    const out = downconvert(rewound, epochCfg, { epoch, nonce: nonces("EPOCH-2"), onSeed: (e) => reseeds.push(e) })
    expect(reseeds.map((e) => e.id)).toEqual(["EPOCH-2"])
    expect(out.prompt).toHaveLength(1)
    expect(textOf(out)).toContain("A different first prompt.")
  })

  test("system prompt change: re-seeds", () => {
    const { seeds } = seed()
    const changed: LanguageModelV3CallOptions = {
      ...extended,
      prompt: [{ role: "system", content: "You are opencode. (AGENTS.md edited)" }, ...extended.prompt.slice(1)],
    }
    const reseeds: unknown[] = []
    const out = downconvert(changed, epochCfg, { epoch: seeds[0], nonce: nonces("EPOCH-2"), onSeed: (e) => reseeds.push(e) })
    expect(reseeds).toHaveLength(1)
    expect(out.prompt).toHaveLength(1)
  })

  test("periodic: re-seeds once epochTurns new turns have accumulated; 0 disables", () => {
    const { seeds } = seed()
    const two = ReactAdapterConfig.resolve({ reactMode: true, react: { mode: "epoch", epochTurns: 2 } })!
    const never = ReactAdapterConfig.resolve({ reactMode: true, react: { mode: "epoch", epochTurns: 0 } })!
    const a: unknown[] = []
    const outA = downconvert(extended, two, { epoch: seeds[0], nonce: nonces("EPOCH-2"), onSeed: (e) => a.push(e) })
    expect(a).toHaveLength(1)
    expect(outA.prompt).toHaveLength(1)
    const b: unknown[] = []
    const outB = downconvert(extended, never, { epoch: seeds[0], nonce: nonces("EPOCH-2"), onSeed: (e) => b.push(e) })
    expect(b).toHaveLength(0)
    expect(outB.prompt).toHaveLength(3)
  })

  test("no new turns since the seed: re-seeds rather than resending the snapshot alone", () => {
    const { seeds } = seed()
    const reseeds: unknown[] = []
    const out = downconvert(seedPrompt, epochCfg, { epoch: seeds[0], nonce: nonces("EPOCH-2"), onSeed: (e) => reseeds.push(e) })
    expect(reseeds).toHaveLength(1)
    expect(out.prompt).toHaveLength(1)
  })

  test("next turn after the snapshot is not an assistant reply: re-seeds", () => {
    // Seed on the first two turns only, so the third (user) turn follows the snapshot directly.
    const short: LanguageModelV3CallOptions = { ...seedPrompt, prompt: seedPrompt.prompt.slice(0, 3) }
    const { seeds } = seed(short)
    expect(seeds[0].covered).toBe(2)
    const reseeds: unknown[] = []
    const out = downconvert(seedPrompt, epochCfg, { epoch: seeds[0], nonce: nonces("EPOCH-2"), onSeed: (e) => reseeds.push(e) })
    expect(reseeds).toHaveLength(1)
    expect(out.prompt).toHaveLength(1)
  })

  test("config: mode parsing, singleMessage alias, epochTurns default and clamp", () => {
    expect(ReactAdapterConfig.resolve({ reactMode: true, react: { singleMessage: true } })!.mode).toBe("single")
    expect(ReactAdapterConfig.resolve({ reactMode: true, react: { mode: "epoch" } })!.mode).toBe("epoch")
    expect(ReactAdapterConfig.resolve({ reactMode: true, react: { mode: "bogus" } })!.mode).toBe("alternating")
    expect(ReactAdapterConfig.resolve({ reactMode: true, react: {} })!.epochTurns).toBe(40)
    expect(ReactAdapterConfig.resolve({ reactMode: true, react: { epochTurns: 7.9 } })!.epochTurns).toBe(7)
    expect(ReactAdapterConfig.resolve({ reactMode: true, react: { epochTurns: -1 } })!.epochTurns).toBe(40)
  })

  test("middleware: runtime mode override wins over config; onSeed is awaited before the request", async () => {
    const asStream = (params: LanguageModelV3CallOptions) => ({ type: "stream" as const, params, model: {} as any })
    const single = ReactAdapter.middleware({ reactMode: true }, { mode: "single" })[0]
    const outSingle = await single.transformParams!(asStream(seedPrompt))
    expect(outSingle.prompt).toHaveLength(1)
    expect(textOf(outSingle).startsWith("Request ")).toBe(true)

    let saved: { id: string; covered: number } | undefined
    const epochMw = ReactAdapter.middleware(
      { reactMode: true, react: { mode: "epoch" } },
      { onSeed: async (e) => void (saved = e) },
    )[0]
    const outEpoch = await epochMw.transformParams!(asStream(seedPrompt))
    expect(outEpoch.prompt).toHaveLength(1)
    expect(saved?.covered).toBe(3)
    expect(textOf(outEpoch)).toBe(`Request ${saved!.id}` + textOf(outEpoch).slice(`Request ${saved!.id}`.length))
  })
})

describe("react-adapter activation", () => {
  test("flag absent or false resolves to no middleware (stock behavior)", () => {
    expect(ReactAdapter.middleware(undefined)).toEqual([])
    expect(ReactAdapter.middleware({})).toEqual([])
    expect(ReactAdapter.middleware({ reactMode: false })).toEqual([])
    expect(ReactAdapter.middleware({ baseURL: "https://x/v1", apiKey: "k" })).toEqual([])
  })

  test("reactMode true with empty react object uses defaults", () => {
    const cfg = ReactAdapterConfig.resolve({ reactMode: true, react: {} })!
    expect(cfg).toEqual({ resultPrefix: "TOOL_RESULT", retries: 2, caps: {}, mode: "alternating", epochTurns: 40 })
    expect(ReactAdapter.middleware({ reactMode: true })).toHaveLength(1)
  })
})
