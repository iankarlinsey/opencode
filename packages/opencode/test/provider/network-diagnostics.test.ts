import { afterEach, expect } from "bun:test"
import { createServer, type Server } from "node:http"
import { readFile } from "node:fs/promises"
import path from "node:path"
import { streamText } from "ai"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { Global } from "@opencode-ai/core/global"
import { Effect } from "effect"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { disposeAllInstances, provideTmpdirInstance } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { testProviderConfig } from "../lib/test-provider"
import { Env } from "@/env"
import { Plugin } from "@/plugin"
import { Provider } from "@/provider/provider"

const previous = process.env.OPENCODE_REACT_DIAGNOSTICS

afterEach(async () => {
  if (previous === undefined) delete process.env.OPENCODE_REACT_DIAGNOSTICS
  else process.env.OPENCODE_REACT_DIAGNOSTICS = previous
  await disposeAllInstances()
})

const it = testEffect(
  LayerNode.compile(LayerNode.group([Provider.node, Env.node, Plugin.node, CrossSpawnSpawner.node])),
)

it.live("network diagnostics record request and paired error response with redacted headers", () =>
  Effect.gen(function* () {
    process.env.OPENCODE_REACT_DIAGNOSTICS = "1"
    const server = yield* Effect.acquireRelease(
      Effect.promise(() => errorServer()),
      (server) =>
        Effect.sync(() => {
          server.server.closeAllConnections()
          server.server.close()
        }),
    )
    const logPath = path.join(Global.Path.log, "opencode.log")
    const before = yield* Effect.promise(() => readFile(logPath, "utf8").catch(() => ""))

    yield* provideTmpdirInstance(
      () =>
        Effect.gen(function* () {
          const provider = yield* Provider.Service
          const model = yield* provider.getModel(ProviderV2.ID.make("test"), ModelV2.ID.make("test-model"))
          const result = streamText({
            model: yield* provider.getLanguage(model),
            onError() {},
            maxRetries: 0,
            messages: [{ role: "user", content: "hello" }],
          })
          yield* Effect.promise(async () => {
            for await (const _ of result.fullStream) {
            }
          })
        }),
      { config: testProviderConfig(server.url) },
    )

    const appended = (yield* Effect.promise(() => readFile(logPath, "utf8"))).slice(before.length)
    const request = appended.match(/\[react-adapter network-debug\] request=(\w+) stage=request[\s\S]*?={10,}\n/)
    const response = appended.match(/\[react-adapter network-debug\] request=(\w+) stage=response[\s\S]*?={10,}\n/)
    expect(request).not.toBeNull()
    expect(response).not.toBeNull()
    // paired by id
    expect(response![1]).toBe(request![1])

    expect(request![0]).toContain("method=POST")
    expect(request![0]).toContain("/chat/completions")
    expect(request![0]).toContain('"authorization":"<redacted>"')
    expect(request![0]).toContain("messagesCount=1")
    expect(request![0]).not.toContain("hello")

    expect(response![0]).toContain("status=502 Bad Gateway")
    expect(response![0]).toContain('"x-test-diag":"present"')
    expect(response![0]).toContain('"set-cookie":"<redacted>"')
    expect(response![0]).toContain("bodyPreview=")
    expect(response![0]).toContain("model backend encountered an error")
    expect(response![0]).toMatch(/elapsedMs=\d+/)
  }),
)

it.live("network diagnostics stay silent when the flag is off", () =>
  Effect.gen(function* () {
    delete process.env.OPENCODE_REACT_DIAGNOSTICS
    const server = yield* Effect.acquireRelease(
      Effect.promise(() => errorServer()),
      (server) =>
        Effect.sync(() => {
          server.server.closeAllConnections()
          server.server.close()
        }),
    )
    const logPath = path.join(Global.Path.log, "opencode.log")
    const before = yield* Effect.promise(() => readFile(logPath, "utf8").catch(() => ""))

    yield* provideTmpdirInstance(
      () =>
        Effect.gen(function* () {
          const provider = yield* Provider.Service
          const model = yield* provider.getModel(ProviderV2.ID.make("test"), ModelV2.ID.make("test-model"))
          const result = streamText({
            model: yield* provider.getLanguage(model),
            onError() {},
            maxRetries: 0,
            messages: [{ role: "user", content: "hello" }],
          })
          yield* Effect.promise(async () => {
            for await (const _ of result.fullStream) {
            }
          })
        }),
      { config: testProviderConfig(server.url) },
    )

    const after = yield* Effect.promise(() => readFile(logPath, "utf8").catch(() => ""))
    expect(after.slice(before.length)).not.toContain("network-debug")
  }),
)

async function errorServer(): Promise<{ server: Server; url: string }> {
  const server = createServer((_, res) => {
    res.writeHead(502, {
      "content-type": "application/json",
      "x-test-diag": "present",
      "set-cookie": "session=secret; Path=/",
    })
    res.end(JSON.stringify({ error: { message: "The model backend encountered an error.", type: "upstream_error" } }))
  })
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve))
  const address = server.address()
  if (!address || typeof address === "string") throw new Error("server did not bind to a TCP port")
  return { server, url: `http://127.0.0.1:${address.port}` }
}
