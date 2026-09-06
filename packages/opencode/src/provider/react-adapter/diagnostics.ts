import * as crypto from "node:crypto"
import type { LanguageModelV3CallOptions, LanguageModelV3Message, LanguageModelV3 } from "@ai-sdk/provider"
import * as os from "node:os"
import * as path from "node:path"
import * as fs from "node:fs"

export * as ReactAdapterDiagnostics from "./diagnostics"

export function isEnabled(): boolean {
  return process.env.OPENCODE_REACT_DIAGNOSTICS === "1" || process.env.OPENCODE_REACT_DIAGNOSTICS === "true" || process.env.OPENCODE_LOG_LEVEL === "DEBUG"
}

function sha256(data: string): string {
  return crypto.createHash("sha256").update(data).digest("hex")
}

function hashMessage(msg: LanguageModelV3Message): string {
  let contentStr = ""
  if (Array.isArray(msg.content)) {
    for (const part of msg.content) {
      if (part.type === "text") contentStr += `text:${part.text}`
      else if (part.type === "tool-call") contentStr += `tool-call:${part.toolName}:${JSON.stringify(part.input)}`
      else if (part.type === "tool-result") {
        let res = ""
        if (part.output.type === "text") res = part.output.value
        else if (part.output.type === "json") res = JSON.stringify(part.output.value)
        else if (part.output.type === "error-text") res = `error:${part.output.value}`
        else if (part.output.type === "error-json") res = `error:${JSON.stringify(part.output.value)}`
        else if (part.output.type === "execution-denied") res = `denied:${part.output.reason ?? ""}`
        else if (part.output.type === "content") res = part.output.value.map(c => c.type === "text" ? c.text : "[media]").join("")
        contentStr += `tool-result:${part.toolName}:${res}`
      }
    }
  } else if (typeof msg.content === "string") {
    contentStr = msg.content
  }
  return sha256(`${msg.role}:${contentStr}`)
}

export function logContext(params: LanguageModelV3CallOptions, flattened: LanguageModelV3CallOptions, model: LanguageModelV3) {
  if (!isEnabled()) return

  const reqId = crypto.randomUUID().slice(0, 8)
  const timestamp = new Date().toISOString()
  let out = ""

  out += `[react-adapter context-debug] request=${reqId} timestamp=${timestamp} stage=input\n`
  
  // Extract session if we can find it
  let sessionId = "unavailable"
  if (params.headers && params.headers['x-opencode-session-id']) {
     sessionId = params.headers['x-opencode-session-id']
  } else if (model.provider === "opencode" || model.provider === "opencode-client") {
     // maybe session is passed some other way, we'll try our best
  }
  
  out += `session=${sessionId}\n`
  out += `model=${model.modelId}\n`
  out += `endpoint=${model.provider}\n`
  out += `messages=${params.prompt.length}\n`

  let totalChars = 0
  let contextStr = ""
  const msgLines = []

  for (let i = 0; i < params.prompt.length; i++) {
    const msg = params.prompt[i]
    let chars = 0
    if (Array.isArray(msg.content)) {
        for (const p of msg.content) {
            if (p.type === "text") chars += p.text.length
            else if (p.type === "tool-call") chars += JSON.stringify(p.input).length + p.toolName.length
            else if (p.type === "tool-result") chars += JSON.stringify(p.output).length // approx
        }
    } else {
        chars += typeof msg.content === "string" ? msg.content.length : 0
    }
    totalChars += chars
    const h = hashMessage(msg)
    contextStr += h + "\n"
    msgLines.push(`msg[${i}] role=${msg.role} chars=${chars} sha256=${h}`)
  }
  
  out += `totalChars=${totalChars}\n`
  out += `contextSha256=${sha256(contextStr)}\n\n`
  out += msgLines.join("\n") + "\n\n"

  let flatChars = 0
  let flatStr = ""
  const flatRoles = []
  for (const msg of flattened.prompt) {
    flatRoles.push(msg.role)
    const content = Array.isArray(msg.content) ? msg.content.map(p => p.type === "text" ? p.text : "").join("") : msg.content
    flatChars += content.length
    flatStr += `${msg.role}:${content}\n`
  }
  
  const flatBytes = Buffer.byteLength(flatStr, "utf8")

  out += `[react-adapter context-debug] request=${reqId} stage=flattened\n`
  out += `chars=${flatChars}\n`
  out += `bytes=${flatBytes}\n`
  out += `sha256=${sha256(flatStr)}\n\n`

  out += `[react-adapter context-debug] request=${reqId} stage=outbound\n`
  out += `messages=${flattened.prompt.length}\n`
  out += `roles=${flatRoles.join(", ")}\n`
  out += `bodyBytes=${flatBytes}\n`
  out += "==========================================================\n"

  // Use existing log path if possible or fallback to standard log location
  const logPath = path.join(os.homedir(), ".local/share/opencode/log/opencode.log")
  try {
    fs.appendFileSync(logPath, out)
  } catch (e) {
    // ignore
  }
}
