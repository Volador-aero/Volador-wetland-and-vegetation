
import { useState } from "react";
import { Bot, Send, User, Loader2, Sparkles } from "lucide-react";
import { sendAssistantMessage } from "../../api/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import { useProject } from "../../context/useProject";
type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
};

const suggestions = [
  "What is NDVI?",
  "Explain wetland degradation.",
  "What is the difference between NDVI and NDWI?",
  "How does satellite-based vegetation monitoring work?",
];

export default function AssistantPage() {
  const { project, runStatus } = useProject();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSend = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const userQuestion = question.trim();

    if (!userQuestion || loading) return;

    const userMessage: ChatMessage = {
      id: Date.now(),
      role: "user",
      content: userQuestion,
    };

    setMessages((previous) => [...previous, userMessage]);
    setQuestion("");
    setError("");
    setLoading(true);

    try {
      const response = await sendAssistantMessage(userQuestion, {
        projectId: project?.id,
        runId:
          runStatus && runStatus.project_id === project?.id
          ? runStatus.run_id : project?.latest_run_id,
      });

      const assistantMessage: ChatMessage = {
        id: Date.now() + 1,
        role: "assistant",
        content: response.answer,
      };

      setMessages((previous) => [...previous, assistantMessage]);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the AI Assistant."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestion = (suggestion: string) => {
    setQuestion(suggestion);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--surface)]">
      <div className="flex items-center gap-3 border-b border-[var(--border)] p-5">
        <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[var(--accent-soft)] text-[var(--accent)]">
          <Bot size={24} />
        </div>

        <div>
          <h2 className="text-base font-semibold text-[var(--text)]">
            AETHRA AI Assistant
          </h2>
        <p className="mt-1 text-xs text-[var(--text-secondary)]">
            {project
                ? `Answering questions about: ${project.name}`
                : "No project selected — asking general GIS questions"}
        </p>
        </div>

        <div className="ml-auto flex items-center gap-2 text-xs text-[var(--success)]">
          <span className="h-2 w-2 rounded-full bg-[var(--success)]" />
          Ready
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-5">
        {messages.length === 0 ? (
          <div className="m-auto flex w-full max-w-2xl flex-col items-center text-center">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-[var(--accent-soft)] text-[var(--accent)]">
              <Sparkles size={30} />
            </div>

            <h3 className="text-xl font-semibold text-[var(--text)]">
              How can I help you today?
            </h3>

            <p className="mt-2 max-w-lg text-sm text-[var(--text-secondary)]">
              Ask questions about GIS, remote sensing, vegetation indices,
              wetland monitoring, and AETHRA functionality.
            </p>

            <div className="mt-7 grid w-full gap-3 sm:grid-cols-2">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => handleSuggestion(suggestion)}
                  className="rounded-lg border border-[var(--border)] p-3 text-left text-sm text-[var(--text-secondary)] transition-colors hover:border-[var(--accent)] hover:bg-[var(--background)] hover:text-[var(--text)]"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${
                message.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {message.role === "assistant" && (
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--accent-soft)] text-[var(--accent)]">
                  <Bot size={17} />
                </div>
              )}

              <div
                className={`max-w-[85%] overflow-x-auto rounded-xl px-4 py-3 text-sm leading-6 ${
                  message.role === "user"
                    ? "bg-[var(--accent)] text-white"
                    : "border border-[var(--border)] bg-[var(--background)] text-[var(--text)]"
                }`}
              >
                {message.role === "assistant" ? (
                    <div className="markdown-content space-y-3 [&_h1]:text-lg [&_h1]:font-bold [&_h2]:text-base [&_h2]:font-semibold [&_h3]:font-semibold [&_p]:leading-6 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-1 [&_strong]:font-semibold [&_table]:w-full [&_table]:border-collapse [&_th]:border [&_th]:border-[var(--border)] [&_th]:bg-[var(--surface)] [&_th]:p-2 [&_td]:border [&_td]:border-[var(--border)] [&_td]:p-2 [&_code]:rounded [&_code]:bg-[var(--surface)] [&_code]:px-1">
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm, remarkMath]}
                            rehypePlugins={[rehypeKatex]}
                        >
                            {message.content}
                        </ReactMarkdown>
                    </div>
                    ) : (
                    message.content
                )}
              </div>

              {message.role === "user" && (
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--background)] text-[var(--text-secondary)]">
                  <User size={17} />
                </div>
              )}
            </div>
          ))
        )}

        {loading && (
          <div className="flex items-center gap-3 text-sm text-[var(--text-secondary)]">
            <Loader2 size={18} className="animate-spin" />
            AETHRA Assistant is thinking...
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
            {error}
          </div>
        )}
      </div>

      <form
        onSubmit={handleSend}
        className="border-t border-[var(--border)] p-4"
      >
        <div className="mx-auto flex max-w-4xl items-end gap-3 rounded-xl border border-[var(--border)] bg-[var(--background)] p-2">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="Ask AETHRA Assistant a question..."
            rows={1}
            maxLength={2000}
            disabled={loading}
            className="max-h-32 min-h-10 flex-1 resize-y bg-transparent px-2 py-2 text-sm text-[var(--text)] outline-none placeholder:text-[var(--text-secondary)] disabled:opacity-50"
          />

          <button
            type="submit"
            disabled={!question.trim() || loading}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[var(--accent)] text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
            aria-label="Send message"
          >
            {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
          </button>
        </div>

        <p className="mt-2 text-center text-[10px] text-[var(--text-secondary)]">
          AI-generated responses may contain errors. Verify important GIS findings against actual analysis results.
        </p>
      </form>
    </div>
  );
}
