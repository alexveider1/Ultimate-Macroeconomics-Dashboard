/**
 * SSE client for the agent chat. `EventSource` can't POST, so we POST via
 * `fetch` and hand-parse `data:` frames off the `ReadableStream`. Mirrors
 * `app/core/api_client.py:agent_chat_stream`.
 */

export interface ChatHistoryTurn {
  role: string;
  content: string;
}

export interface ChatStreamBody {
  user_message: string;
  chat_history: ChatHistoryTurn[];
  session_id?: string;
  images?: string[];
}

/** One decoded SSE event from the agent's chat stream. */
export interface ChatStreamEvent {
  type: "step" | "token" | "final" | "error" | string;
  node?: string;
  delta?: string;
  answer?: string;
  artifacts?: Record<string, unknown>;
  usage?: Record<string, unknown>;
}

/**
 * POST the chat body and yield each decoded SSE event. The caller drives it with
 * `for await`; pass an `AbortSignal` to cancel mid-stream (e.g. on unmount).
 */
export async function* streamAgentChat(
  body: ChatStreamBody,
  signal?: AbortSignal,
): AsyncGenerator<ChatStreamEvent> {
  const response = await fetch("/api/agent/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Agent stream failed: ${response.status} ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      let sep = buffer.indexOf("\n\n");
      while (sep !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const event = parseFrame(frame);
        if (event) yield event;
        sep = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** Extract and JSON-parse the concatenated `data:` lines of one SSE frame. */
function parseFrame(frame: string): ChatStreamEvent | null {
  const dataLines = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart());
  if (dataLines.length === 0) return null;
  const payload = dataLines.join("\n");
  try {
    return JSON.parse(payload) as ChatStreamEvent;
  } catch {
    return null;
  }
}
