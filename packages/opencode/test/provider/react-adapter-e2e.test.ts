import { describe, expect, test } from "bun:test"
import type { LanguageModelV3, LanguageModelV3CallOptions, LanguageModelV3StreamPart } from "@ai-sdk/provider"
import { jsonSchema, stepCountIs, streamText, tool, wrapLanguageModel } from "ai"
import { ReactAdapter } from "@/provider/react-adapter"

/**
 * End-to-end: a mocked degraded LanguageModel that only speaks ReAct text,
 * driven through the real streamText with the adapter installed. Asserts the
 * opencode-visible stream parts AND the transcripts the "provider" receives.
 */

function textStream(text: string): ReadableStream<LanguageModelV3StreamPart> {
  return new ReadableStream({
    start(controller) {
      controller.enqueue({ type: "stream-start", warnings: [] })
      controller.enqueue({ type: "text-start", id: "t" })
      for (const chunk of text.match(/[\s\S]{1,7}/g) ?? [])
        controller.enqueue({ type: "text-delta", id: "t", delta: chunk })
      controller.enqueue({ type: "text-end", id: "t" })
      controller.enqueue({
        type: "finish",
        finishReason: { unified: "stop", raw: "stop" },
        usage: {
          inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 5, text: 5, reasoning: 0 },
        },
      })
      controller.close()
    },
  })
}

function mockModel(responses: string[]) {
  const calls: LanguageModelV3CallOptions[] = []
  const model: LanguageModelV3 = {
    specificationVersion: "v3",
    provider: "mock-degraded",
    modelId: "gemini-shim",
    supportedUrls: {},
    async doGenerate() {
      throw new Error("not used")
    },
    async doStream(options) {
      calls.push(options)
      return { stream: textStream(responses[Math.min(calls.length - 1, responses.length - 1)]) }
    },
  }
  return { model, calls }
}

const tools = {
  bash: tool({
    description: "Run a shell command",
    inputSchema: jsonSchema<{ command: string }>({
      type: "object",
      properties: { command: { type: "string" } },
      required: ["command"],
    }),
    execute: async ({ command }) => `ran: ${command}\nsrc\npackage.json`,
  }),
  read: tool({
    description: "Read a file",
    inputSchema: jsonSchema<{ filePath: string }>({
      type: "object",
      properties: { filePath: { type: "string" } },
      required: ["filePath"],
    }),
    execute: async ({ filePath }) => `contents of ${filePath}: {"version":"2.1.0"}`,
  }),
}

function adapted(model: LanguageModelV3) {
  return wrapLanguageModel({ model, middleware: ReactAdapter.middleware({ reactMode: true }) })
}

describe("react-adapter e2e through streamText", () => {
  test("3-turn ReAct exchange produces native-looking stream parts", async () => {
    const { model, calls } = mockModel([
      'Thought: I should list the project files.\nAction: bash\nAction Input: {"command": "ls"}',
      'Thought: package.json exists, read it.\nAction: read\nAction Input: {"filePath": "package.json"}',
      "Final Answer: The project is at version 2.1.0.",
    ])

    const result = streamText({
      model: adapted(model),
      messages: [{ role: "user", content: "What version is this project?" }],
      tools,
      stopWhen: stepCountIs(5),
    })

    const parts: any[] = []
    for await (const part of result.fullStream) parts.push(part)

    const toolCalls = parts.filter((p) => p.type === "tool-call")
    expect(toolCalls.map((p: any) => p.toolName)).toEqual(["bash", "read"])
    expect(toolCalls[0].input).toEqual({ command: "ls" })
    expect(toolCalls[1].input).toEqual({ filePath: "package.json" })

    const toolResults = parts.filter((p) => p.type === "tool-result")
    expect(toolResults).toHaveLength(2)
    expect(toolResults[0].output).toContain("ran: ls")

    expect(await result.text).toContain("version 2.1.0")
    expect(await result.finishReason).toBe("stop")
    expect(parts.filter((p) => p.type === "error")).toEqual([])

    // The degraded provider only ever saw user/assistant text messages.
    expect(calls).toHaveLength(3)
    for (const call of calls) {
      expect(call.tools ?? []).toEqual([])
      expect(call.toolChoice).toBeUndefined()
      for (const message of call.prompt) expect(["user", "assistant"]).toContain(message.role)
      const roles = call.prompt.map((m) => m.role)
      for (let i = 1; i < roles.length; i++) expect(roles[i]).not.toBe(roles[i - 1])
    }

    // Second call: preamble in first user message, ReAct history, TOOL_RESULT + nudge last.
    const second = calls[1]
    const firstText = (second.prompt[0].content as { type: "text"; text: string }[])[0].text
    expect(firstText).toContain("# Response protocol")
    expect(firstText).toContain("--- SESSION START ---")
    const lastMessage = second.prompt.at(-1)!
    expect(lastMessage.role).toBe("user")
    const lastText = (lastMessage.content as { type: "text"; text: string }[])[0].text
    expect(lastText).toContain("TOOL_RESULT [bash]:")
    expect(lastText).toContain("Continue: state your next Thought and Action, or give your Final Answer.")
    const assistantText = (second.prompt[1].content as { type: "text"; text: string }[])[0].text
    expect(assistantText).toContain("Thought: I should list the project files.")
    expect(assistantText).toContain("Action: bash")
    expect(second.stopSequences).toContain("\nTOOL_RESULT")
  })

  test("empty response retries with nudge on clean history, then succeeds", async () => {
    const { model, calls } = mockModel(["", "Final Answer: recovered."])

    const result = streamText({
      model: adapted(model),
      messages: [{ role: "user", content: "hello" }],
      tools,
      stopWhen: stepCountIs(2),
    })

    expect(await result.text).toBe("recovered.")
    expect(calls).toHaveLength(2)
    const retryLast = calls[1].prompt.at(-1)!
    const retryText = (retryLast.content as { type: "text"; text: string }[]).map((p) => p.text).join("")
    expect(retryText).toContain("[Your previous response was empty.")
    // First call had no nudge (clean history).
    const firstText = (calls[0].prompt.at(-1)!.content as { type: "text"; text: string }[])
      .map((p) => p.text)
      .join("")
    expect(firstText).not.toContain("[Your previous response was empty.")
  })

  test("retries exhausted surfaces a distinguishable error, not a silent stop", async () => {
    const { model, calls } = mockModel([""])

    const result = streamText({
      model: adapted(model),
      messages: [{ role: "user", content: "hello" }],
      tools,
      stopWhen: stepCountIs(2),
      onError: () => {},
    })

    const parts: any[] = []
    for await (const part of result.fullStream) parts.push(part)
    const errors = parts.filter((p) => p.type === "error")
    expect(errors).toHaveLength(1)
    expect(String(errors[0].error)).toContain("react-adapter: provider returned an empty response")
    expect(calls).toHaveLength(3) // initial + 2 retries (default)
  })
})
