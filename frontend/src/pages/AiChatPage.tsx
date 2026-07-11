import { SendHorizontal } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { streamAgentChat } from "@/api/sse";
import { ChatArtifacts } from "@/components/ChatArtifacts";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const STEP_DISPLAY_NAMES: Record<string, string> = {
  guardrail: "guardrail",
  supervisor: "router",
  sql_agent: "sql_agent",
  plotly_agent: "plotly_agent",
  table_agent: "table_agent",
  rag_agent: "rag_agent",
  web_search: "web_search",
  downloader_agent: "downloader_agent",
  chat_agent: "chat_agent",
  FINISH: "FINISH",
};

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  steps?: string[];
  artifacts?: Record<string, unknown>;
  error?: boolean;
}

function StepBreadcrumb({ steps }: { steps: string[] }) {
  if (steps.length === 0) return null;
  const chain = steps.map((s) => STEP_DISPLAY_NAMES[s] ?? s).join(" → ");
  return <code className="text-xs text-muted-foreground">{chain}</code>;
}

/** AI Analyst chat — streams the LangGraph agent over SSE with steps + artifacts. */
export function AiChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [live, setLive] = useState<{ steps: string[]; content: string } | null>(null);
  const [input, setInput] = useState("");
  const sessionId = useRef<string>(crypto.randomUUID());
  const scrollRef = useRef<HTMLDivElement>(null);
  const streaming = live !== null;

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, live]);

  const send = async () => {
    const prompt = input.trim();
    if (!prompt || streaming) return;
    setInput("");

    const history = messages
      .filter((m) => m.content.trim())
      .slice(-24)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, { role: "user", content: prompt }]);
    setLive({ steps: [], content: "" });

    const steps: string[] = [];
    let buffer = "";
    let artifacts: Record<string, unknown> = {};
    let finalAnswer = "";
    let errorText = "";

    try {
      for await (const event of streamAgentChat({
        user_message: prompt,
        chat_history: history,
        session_id: sessionId.current,
      })) {
        if (event.type === "step" && event.node) {
          if (steps[steps.length - 1] !== event.node) steps.push(event.node);
          setLive({ steps: [...steps], content: buffer });
        } else if (event.type === "token" && event.delta) {
          buffer += event.delta;
          setLive({ steps: [...steps], content: buffer });
        } else if (event.type === "final") {
          finalAnswer = event.answer || buffer;
          artifacts = event.artifacts ?? {};
          break;
        } else if (event.type === "error") {
          errorText = event.answer || "Agent error.";
          break;
        }
      }
    } catch (error) {
      errorText = `Agent request failed: ${(error as Error).message}`;
    }

    const isError = Boolean(errorText) && !finalAnswer;
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: isError ? errorText : finalAnswer || buffer || "No answer returned.",
        steps,
        artifacts,
        error: isError,
      },
    ]);
    setLive(null);
  };

  const clear = () => {
    setMessages([]);
    setLive(null);
    sessionId.current = crypto.randomUUID();
  };

  return (
    <div className="flex h-[calc(100vh-8rem)] flex-col">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold">AI Analyst</h2>
          <p className="text-sm text-muted-foreground">
            Chat with the LangGraph agent — SQL, plots, RAG, and web search.
          </p>
        </div>
        <Button variant="outline" onClick={clear} disabled={streaming}>
          Clear chat
        </Button>
      </div>

      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto rounded-lg border bg-card p-4">
        {messages.length === 0 && !live && (
          <p className="text-sm text-muted-foreground">
            Ask a question about the macroeconomic data — e.g. “Plot US and China GDP since 2000”.
          </p>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={cn("flex", message.role === "user" ? "justify-end" : "justify-start")}
          >
            <div
              className={cn(
                "max-w-[85%] space-y-1 rounded-lg px-3 py-2",
                message.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-foreground",
              )}
            >
              {message.role === "assistant" && message.steps && (
                <StepBreadcrumb steps={message.steps} />
              )}
              <p
                className={cn(
                  "whitespace-pre-wrap text-sm",
                  message.error && "text-negative",
                )}
              >
                {message.content}
              </p>
              {message.role === "assistant" && message.artifacts && (
                <ChatArtifacts artifacts={message.artifacts} />
              )}
            </div>
          </div>
        ))}

        {live && (
          <div className="flex justify-start">
            <div className="max-w-[85%] space-y-1 rounded-lg bg-muted px-3 py-2 text-foreground">
              <StepBreadcrumb steps={[...live.steps, "…"]} />
              <p className="whitespace-pre-wrap text-sm">{live.content || "Thinking…"}</p>
            </div>
          </div>
        )}
      </div>

      <div className="mt-3 flex items-end gap-2">
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void send();
            }
          }}
          rows={1}
          placeholder="Ask the AI analyst…"
          className="max-h-40 flex-1 resize-none rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
        <Button
          onClick={() => void send()}
          disabled={streaming || !input.trim()}
          aria-label="Send message"
        >
          <SendHorizontal className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
