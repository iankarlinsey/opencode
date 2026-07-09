import { describe, expect, test } from "bun:test"
import type { LanguageModelV3CallOptions } from "@ai-sdk/provider"
import { parse } from "@/provider/react-adapter/parse"
import { normalize } from "@/provider/react-adapter/normalize"
import { downconvert } from "@/provider/react-adapter/downconvert"
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

  test("malformed JSON falls back through the ladder", () => {
    const parsed = parse('Action: bash\nAction Input: {"command": "echo hi",}', SCHEMAS)
    expect(parsed.kind).toBe("action")
    // trailing comma is invalid JSON; bare-string wrap catches it
    expect(JSON.parse((parsed as any).input)).toEqual({ command: '{"command": "echo hi",}' })
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

describe("react-adapter activation", () => {
  test("flag absent or false resolves to no middleware (stock behavior)", () => {
    expect(ReactAdapter.middleware(undefined)).toEqual([])
    expect(ReactAdapter.middleware({})).toEqual([])
    expect(ReactAdapter.middleware({ reactMode: false })).toEqual([])
    expect(ReactAdapter.middleware({ baseURL: "https://x/v1", apiKey: "k" })).toEqual([])
  })

  test("reactMode true with empty react object uses defaults", () => {
    const cfg = ReactAdapterConfig.resolve({ reactMode: true, react: {} })!
    expect(cfg).toEqual({ resultPrefix: "TOOL_RESULT", retries: 2, caps: {} })
    expect(ReactAdapter.middleware({ reactMode: true })).toHaveLength(1)
  })
})
