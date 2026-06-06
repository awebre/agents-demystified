import { useState, useEffect, useRef, type ReactElement } from "react";
import "./App.css";

interface TokenCandidate {
  token: string;
  logProbability: number;
  probability: number;
}

interface PredictResponse {
  predictedToken: string;
  candidates: TokenCandidate[];
}

interface GeneratedToken {
  token: string;
  done: boolean;
  candidates: TokenCandidate[];
}

interface ChatMessage {
  role: string;
  content: string;
}

type AgentStep =
  | { type: "text"; tokens: GeneratedToken[] }
  | { type: "tool_call"; name: string; arguments: Record<string, unknown> }
  | { type: "tool_result"; name: string; result: string };

interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, string>;
}

const SPECIAL_TOKENS = {
  system: "<|system|>",
  user: "<|user|>",
  assistant: "<|assistant|>",
  end: "<|end|>",
} as const;

function displayToken(token: string): string {
  return token
    .replace(/ /g, "\u2423")
    .replace(/\n/g, "\u21B5")
    .replace(/\t/g, "\u21E5")
    .replace(/\r/g, "\u23CE");
}

function CandidateList({
  candidates,
  highlightToken,
  onSelect,
}: {
  candidates: TokenCandidate[];
  highlightToken?: string;
  onSelect?: (token: string) => void;
}) {
  if (candidates.length === 0) return null;
  const maxProb = candidates[0].probability;
  return (
    <ul className="candidate-list">
      {candidates.map((candidate, index) => {
        const barWidth = maxProb > 0 ? (candidate.probability / maxProb) * 100 : 0;
        const isHighlighted = highlightToken
          ? candidate.token === highlightToken
          : index === 0;
        return (
          <li
            key={index}
            className={"candidate-row" + (isHighlighted ? " top-candidate" : "")}
            onClick={() => onSelect?.(candidate.token)}
          >
            <span className="candidate-token">{displayToken(candidate.token)}</span>
            <span className="candidate-bar-container">
              <span className="candidate-bar" style={{ width: barWidth + "%" }} />
            </span>
            <span className="candidate-prob">
              {(candidate.probability * 100).toFixed(1) + "%"}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function TokenPredictionView() {
  const [prompt, setPrompt] = useState("The capital of France is");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!prompt.trim()) {
      setResult(null);
      return;
    }

    const controller = new AbortController();
    const timeout = setTimeout(async () => {
      setLoading(true);
      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, topN: 10 }),
          signal: controller.signal,
        });
        const data: PredictResponse = await response.json();
        setResult(data);
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") return;
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, [prompt]);

  return (
    <div className="view">
      <textarea
        className="prompt-input"
        rows={3}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Type a prompt..."
      />

      <div className="candidates">
        {loading && !result && (
          <div className="loading">Predicting...</div>
        )}

        {result && (
          <CandidateList
            candidates={result.candidates}
            onSelect={(token) => setPrompt((prev) => prev + token)}
          />
        )}
      </div>
    </div>
  );
}

function GenerationLoopView() {
  const [genPrompt, setGenPrompt] = useState("Once upon a time");
  const [generatedTokens, setGeneratedTokens] = useState<GeneratedToken[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [temperature, setTemperature] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  const startGeneration = async () => {
    setGeneratedTokens([]);
    setSelectedStep(null);
    setIsGenerating(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: genPrompt, maxTokens: 200, topN: 5, temperature }),
        signal: controller.signal,
      });

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const json = line.slice(6);
          if (!json) continue;
          const event: GeneratedToken = JSON.parse(json);
          setGeneratedTokens((prev) => [...prev, event]);
          if (event.done) break;
        }
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
  };

  const selectedCandidates =
    selectedStep !== null && generatedTokens[selectedStep]
      ? generatedTokens[selectedStep].candidates
      : [];

  const selectedToken =
    selectedStep !== null && generatedTokens[selectedStep]
      ? generatedTokens[selectedStep].token
      : undefined;

  return (
    <div className="view">
      <textarea
        className="prompt-input"
        rows={3}
        value={genPrompt}
        onChange={(e) => setGenPrompt(e.target.value)}
        placeholder="Enter a prompt to generate from..."
        disabled={isGenerating}
      />

      <div className="temperature-control">
        <label className="temperature-label">
          Temperature: <span className="temperature-value">{temperature.toFixed(1)}</span>
          <span className="temperature-hint">
            {temperature === 0 ? "(greedy)" : temperature <= 0.5 ? "(focused)" : temperature <= 1 ? "(balanced)" : "(creative)"}
          </span>
        </label>
        <input
          type="range"
          className="temperature-slider"
          min="0"
          max="2"
          step="0.1"
          value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))}
          disabled={isGenerating}
        />
      </div>

      <div className="button-row">
        <button
          className="generate-btn"
          onClick={startGeneration}
          disabled={isGenerating || !genPrompt.trim()}
        >
          Generate
        </button>
        {isGenerating && (
          <button className="stop-btn" onClick={stopGeneration}>
            Stop
          </button>
        )}
      </div>

      {(generatedTokens.length > 0 || isGenerating) && (
        <div className="generated-text">
          <span className="generated-prompt">{genPrompt}</span>
          {generatedTokens.map((gt, index) => (
            <span
              key={index}
              className={
                "generated-token-span" +
                (selectedStep === index ? " selected-token" : "") +
                (gt.done ? " eos-token" : "")
              }
              onClick={() =>
                setSelectedStep(selectedStep === index ? null : index)
              }
            >
              {gt.done ? gt.token || "\u23F9" : gt.token}
            </span>
          ))}
          {isGenerating && <span className="generating-cursor">|</span>}
        </div>
      )}

      {selectedStep !== null && selectedCandidates.length > 0 && (
        <div className="step-candidates">
          <div className="step-candidates-header">
            Step {selectedStep + 1} candidates
          </div>
          <CandidateList
            candidates={selectedCandidates}
            highlightToken={selectedToken}
          />
        </div>
      )}
    </div>
  );
}

function RawSection({ role, content }: { role: string; content: string }) {
  const token = SPECIAL_TOKENS[role as keyof typeof SPECIAL_TOKENS];
  return (
    <div className={`raw-section raw-section-${role}`}>
      <span className="raw-section-label">{role}</span>
      <span className="raw-special-token">{token}</span>
      {"\n"}
      {content}
      {"\n"}
      <span className="raw-special-token">{SPECIAL_TOKENS.end}</span>
      {"\n"}
    </div>
  );
}

function AssistantTokenSpan({
  gt,
  index,
  selectedStep,
  onSelectStep,
}: {
  gt: GeneratedToken;
  index: number;
  selectedStep: number | null;
  onSelectStep: (index: number | null) => void;
}) {
  return (
    <span
      className={
        "generated-token-span" +
        (selectedStep === index ? " selected-token" : "") +
        (gt.done ? " eos-token" : "")
      }
      onClick={() => onSelectStep(selectedStep === index ? null : index)}
    >
      {gt.done ? gt.token || "\u23F9" : gt.token}
    </span>
  );
}

function RawPromptView({
  systemPrompt,
  messages,
  streamingTokens,
  agentSteps,
  isGenerating,
  selectedStep,
  onSelectStep,
}: {
  systemPrompt: string;
  messages: ChatMessage[];
  streamingTokens?: GeneratedToken[];
  agentSteps?: AgentStep[];
  isGenerating: boolean;
  selectedStep: number | null;
  onSelectStep: (index: number | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [streamingTokens, agentSteps]);

  const tokens = streamingTokens ?? [];
  const steps = agentSteps ?? [];
  const hasActiveGeneration =
    isGenerating || tokens.length > 0 || steps.length > 0;

  let tokenCounter = 0;

  return (
    <div className="raw-prompt-container" ref={containerRef}>
      <RawSection role="system" content={systemPrompt} />
      {messages.map((msg, i) => (
        <RawSection key={i} role={msg.role} content={msg.content} />
      ))}
      {hasActiveGeneration && streamingTokens && (
        <div className="raw-section raw-section-assistant">
          <span className="raw-section-label">assistant</span>
          <span className="raw-special-token">{SPECIAL_TOKENS.assistant}</span>
          {"\n"}
          {tokens.map((gt, index) => (
            <AssistantTokenSpan
              key={index}
              gt={gt}
              index={index}
              selectedStep={selectedStep}
              onSelectStep={onSelectStep}
            />
          ))}
        </div>
      )}
      {hasActiveGeneration &&
        agentSteps &&
        (() => {
          const blocks: ReactElement[] = [];
          let currentTokens: ReactElement[] = [];
          let blockKey = 0;

          const flushAssistant = () => {
            if (currentTokens.length === 0) return;
            blocks.push(
              <div
                key={`assistant-${blockKey++}`}
                className="raw-section raw-section-assistant"
              >
                <span className="raw-section-label">assistant</span>
                <span className="raw-special-token">
                  {SPECIAL_TOKENS.assistant}
                </span>
                {"\n"}
                {currentTokens}
              </div>
            );
            currentTokens = [];
          };

          steps.forEach((step, si) => {
            if (step.type === "text") {
              step.tokens.forEach((gt) => {
                const idx = tokenCounter++;
                currentTokens.push(
                  <AssistantTokenSpan
                    key={`t-${idx}`}
                    gt={gt}
                    index={idx}
                    selectedStep={selectedStep}
                    onSelectStep={onSelectStep}
                  />
                );
              });
            } else if (step.type === "tool_call") {
              flushAssistant();
              blocks.push(
                <div
                  key={`tc-${si}`}
                  className="raw-section raw-section-tool-call"
                >
                  <span className="raw-section-label">tool_call</span>
                  <span className="raw-special-token">{"<tool_call>"}</span>
                  {"\n"}
                  {JSON.stringify(
                    { name: step.name, arguments: step.arguments },
                    null,
                    2
                  )}
                  {"\n"}
                  <span className="raw-special-token">{"</tool_call>"}</span>
                  {"\n"}
                </div>
              );
            } else if (step.type === "tool_result") {
              blocks.push(
                <div
                  key={`tr-${si}`}
                  className="raw-section raw-section-tool-result"
                >
                  <span className="raw-section-label">tool_result</span>
                  <span className="raw-special-token">{"<tool_result>"}</span>
                  {"\n"}
                  {step.result}
                  {"\n"}
                  <span className="raw-special-token">{"</tool_result>"}</span>
                  {"\n"}
                </div>
              );
            }
          });
          flushAssistant();
          return blocks;
        })()}
    </div>
  );
}

function ChatView() {
  const [systemPrompt, setSystemPrompt] = useState(
    "You are a helpful assistant. Keep your responses brief."
  );
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [streamingTokens, setStreamingTokens] = useState<GeneratedToken[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [temperature, setTemperature] = useState(0);
  const [viewMode, setViewMode] = useState<"chat" | "raw">("chat");
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const responseRef = useRef("");
  const chatMessagesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages, streamingTokens]);

  const sendMessage = async () => {
    if (!input.trim() || isGenerating) return;

    // Finalize previous assistant response into conversation history
    const updatedMessages = [...messages];
    if (responseRef.current) {
      updatedMessages.push({ role: "assistant", content: responseRef.current });
    }
    updatedMessages.push({ role: "user", content: input.trim() });

    setMessages(updatedMessages);
    setInput("");
    setStreamingTokens([]);
    setSelectedStep(null);
    setIsGenerating(true);
    responseRef.current = "";

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          systemPrompt,
          messages: updatedMessages,
          maxTokens: 200,
          topN: 5,
          temperature,
        }),
        signal: controller.signal,
      });

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const json = line.slice(6);
          if (!json) continue;
          const event: GeneratedToken = JSON.parse(json);
          setStreamingTokens((prev) => [...prev, event]);
          if (!event.done) {
            responseRef.current += event.token;
          }
          if (event.done) break;
        }
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
      inputRef.current?.focus();
    }
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
  };

  const clearConversation = () => {
    abortRef.current?.abort();
    setMessages([]);
    setStreamingTokens([]);
    setSelectedStep(null);
    setIsGenerating(false);
    responseRef.current = "";
    inputRef.current?.focus();
  };

  return (
    <div className="view">
      <div className="system-prompt-section">
        <div className="system-prompt-header">
          <span className="system-prompt-label">System Prompt</span>
          <button
            className="view-toggle"
            onClick={() => setViewMode(viewMode === "chat" ? "raw" : "chat")}
          >
            {viewMode === "chat" ? "Show Raw" : "Show Chat"}
          </button>
        </div>
        <textarea
          className="system-prompt-textarea"
          rows={2}
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          disabled={isGenerating}
        />
      </div>

      {viewMode === "chat" ? (
        <div className="chat-messages" ref={chatMessagesRef}>
          {messages.map((msg, i) => (
            <div key={i} className={`chat-bubble chat-bubble-${msg.role}`}>
              <span className="chat-role">{msg.role}</span>
              <p>{msg.content}</p>
            </div>
          ))}
          {streamingTokens.length > 0 && (
            <div className="chat-bubble chat-bubble-assistant">
              <span className="chat-role">assistant</span>
              <p>
                {streamingTokens
                  .filter((t) => !t.done)
                  .map((t) => t.token)
                  .join("")}
              </p>
            </div>
          )}
        </div>
      ) : (
        <RawPromptView
          systemPrompt={systemPrompt}
          messages={messages}
          streamingTokens={streamingTokens}
          isGenerating={isGenerating}
          selectedStep={selectedStep}
          onSelectStep={setSelectedStep}
        />
      )}

      {selectedStep !== null && streamingTokens[selectedStep] && (
        <div className="step-candidates">
          <div className="step-candidates-header">
            Step {selectedStep + 1}: chose "{displayToken(streamingTokens[selectedStep].token)}"
          </div>
          <CandidateList
            candidates={streamingTokens[selectedStep].candidates}
            highlightToken={streamingTokens[selectedStep].token}
          />
        </div>
      )}

      <div className="temperature-control">
        <label className="temperature-label">
          Temperature: <span className="temperature-value">{temperature.toFixed(1)}</span>
          <span className="temperature-hint">
            {temperature === 0 ? "(greedy)" : temperature <= 0.5 ? "(focused)" : temperature <= 1 ? "(balanced)" : "(creative)"}
          </span>
        </label>
        <input
          type="range"
          className="temperature-slider"
          min="0"
          max="2"
          step="0.1"
          value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))}
          disabled={isGenerating}
        />
      </div>

      <div className="chat-input-row">
        <input
          ref={inputRef}
          className="chat-input"
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") sendMessage();
          }}
          placeholder="Type a message..."
          disabled={isGenerating}
        />
        <button
          className="generate-btn"
          onClick={sendMessage}
          disabled={isGenerating || !input.trim()}
        >
          Send
        </button>
        {isGenerating && (
          <button className="stop-btn" onClick={stopGeneration}>
            Stop
          </button>
        )}
        <button className="stop-btn" onClick={clearConversation} style={{ background: "#6b7280" }}>
          Clear
        </button>
      </div>
    </div>
  );
}

function toolCallSummary(name: string, args: Record<string, unknown>): {
  icon: string;
  verb: string;
  target: string | null;
} {
  if (name === "search_files") {
    const q = typeof args.query === "string" ? args.query : "";
    return {
      icon: "🔎",
      verb: "Searching files for",
      target: q ? q : "(empty query)",
    };
  }
  if (name === "read_file") {
    const p = typeof args.path === "string" ? args.path : "";
    return {
      icon: "📄",
      verb: "Reading",
      target: p ? p : "(no path)",
    };
  }
  return { icon: "⚡", verb: `Calling ${name}`, target: null };
}

function ToolCallCard({
  name,
  args,
}: {
  name: string;
  args: Record<string, unknown>;
}) {
  const [showRaw, setShowRaw] = useState(false);
  const summary = toolCallSummary(name, args);
  return (
    <div className="tool-call-card">
      <div className="tool-call-header">
        <span className="tool-call-icon">{summary.icon}</span>
        <span className="tool-call-summary">
          {summary.verb}
          {summary.target !== null && (
            <>
              {" "}
              <code className="tool-call-target">{summary.target}</code>
            </>
          )}
        </span>
        <button
          className="tool-card-raw-toggle"
          onClick={() => setShowRaw((v) => !v)}
          type="button"
          title="Show raw tool call"
        >
          {showRaw ? "hide" : "raw"}
        </button>
      </div>
      {showRaw && (
        <pre className="tool-call-args">
          {JSON.stringify({ name, arguments: args }, null, 2)}
        </pre>
      )}
    </div>
  );
}

function ToolResultCard({ name, result }: { name: string; result: string }) {
  const [expanded, setExpanded] = useState(false);

  let parsed: unknown = null;
  try {
    parsed = JSON.parse(result);
  } catch {
    /* leave null */
  }

  const err =
    parsed && typeof parsed === "object" && "error" in (parsed as object)
      ? String((parsed as { error: unknown }).error)
      : null;

  if (err) {
    return (
      <div className="tool-result-card tool-result-error">
        <div className="tool-result-header">
          <span className="tool-result-icon">⚠️</span>
          <span className="tool-result-name">{name} failed</span>
        </div>
        <div className="tool-result-body">{err}</div>
      </div>
    );
  }

  if (name === "search_files" && parsed && typeof parsed === "object") {
    const files = Array.isArray((parsed as { files?: unknown }).files)
      ? ((parsed as { files: unknown[] }).files as string[])
      : [];
    return (
      <div className="tool-result-card">
        <div className="tool-result-header">
          <span className="tool-result-icon">✓</span>
          <span className="tool-result-name">
            Found {files.length} file{files.length === 1 ? "" : "s"}
          </span>
        </div>
        {files.length === 0 ? (
          <div className="tool-result-body tool-result-empty">
            No matches.
          </div>
        ) : (
          <ul className="tool-result-files">
            {files.map((f) => (
              <li key={f}>
                <code>{f}</code>
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  if (name === "read_file" && parsed && typeof parsed === "object") {
    const obj = parsed as { path?: string; content?: string };
    const path = obj.path ?? "";
    const content = obj.content ?? "";
    const truncated = content.length > 600 && !expanded;
    const shown = truncated ? content.slice(0, 600) + "\n… (click expand)" : content;
    return (
      <div className="tool-result-card">
        <div className="tool-result-header">
          <span className="tool-result-icon">📄</span>
          <span className="tool-result-name">
            Read <code>{path}</code>
          </span>
          {content.length > 600 && (
            <button
              className="tool-card-raw-toggle"
              onClick={() => setExpanded((v) => !v)}
              type="button"
            >
              {expanded ? "collapse" : "expand"}
            </button>
          )}
        </div>
        <pre className="tool-result-content">{shown}</pre>
      </div>
    );
  }

  // Fallback — unknown tool or unparseable result
  return (
    <div className="tool-result-card">
      <div className="tool-result-header">
        <span className="tool-result-icon">✓</span>
        <span className="tool-result-name">{name} result</span>
      </div>
      <pre className="tool-result-content">
        {result.length > 500 ? result.slice(0, 500) + "\n… (truncated)" : result}
      </pre>
    </div>
  );
}

function AgentStepsBlock({
  steps,
  keyPrefix,
}: {
  steps: AgentStep[];
  keyPrefix: string;
}) {
  return (
    <>
      {steps.map((step, i) => {
        const k = `${keyPrefix}-${i}`;
        if (step.type === "text") {
          const raw = step.tokens
            .filter((t) => !t.done)
            .map((t) => t.token)
            .join("");
          // Detect partial tool_call (open tag, no close) — model emitted
          // a malformed call and the server couldn't parse it. Surface as
          // an incomplete card so the demo audience sees what happened.
          const partial = raw.match(/<tool_call>([\s\S]*)$/);
          const hasComplete = /<tool_call>[\s\S]*?<\/tool_call>/.test(raw);
          const partialFragment =
            partial && !hasComplete ? partial[1].trim() : null;
          const text = raw
            .replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "")
            .replace(/<tool_call>[\s\S]*$/g, "")
            .trim();
          return (
            <div key={k}>
              {text && (
                <div className="chat-bubble chat-bubble-assistant">
                  <span className="chat-role">assistant</span>
                  <p>{text}</p>
                </div>
              )}
              {partialFragment !== null && (
                <div className="tool-call-card tool-call-incomplete">
                  <div className="tool-call-header">
                    <span className="tool-call-icon">⚠️</span>
                    <span className="tool-call-summary">
                      Incomplete tool call (no closing tag)
                    </span>
                  </div>
                  <pre className="tool-call-args">{partialFragment}</pre>
                </div>
              )}
            </div>
          );
        }
        if (step.type === "tool_call") {
          return <ToolCallCard key={k} name={step.name} args={step.arguments} />;
        }
        return <ToolResultCard key={k} name={step.name} result={step.result} />;
      })}
    </>
  );
}

function AgentView() {
  const [systemPrompt, setSystemPrompt] = useState(
    "You are a helpful assistant. Keep your responses brief."
  );
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pastTurnSteps, setPastTurnSteps] = useState<AgentStep[][]>([]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [temperature, setTemperature] = useState(0);
  const [viewMode, setViewMode] = useState<"chat" | "raw">("chat");
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [showTools, setShowTools] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const responseRef = useRef("");
  const chatMessagesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/tools")
      .then((r) => r.json())
      .then(setTools)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages, steps]);

  const sendMessage = async () => {
    if (!input.trim() || isGenerating) return;

    const updatedMessages = [...messages];
    let updatedPast = pastTurnSteps;
    if (responseRef.current) {
      // Strip tool_call markup from saved assistant content — those render as
      // ToolCallCard while live, but the raw text would leak into the chat
      // bubble once finalized into messages.
      const cleaned = responseRef.current
        .replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "")
        .replace(/<tool_call>[\s\S]*$/g, "")
        .trim();
      if (cleaned) {
        updatedMessages.push({ role: "assistant", content: cleaned });
        // Snapshot the live steps so the tool-call/result history is preserved.
        updatedPast = [...pastTurnSteps, steps];
      }
    }
    updatedMessages.push({ role: "user", content: input.trim() });

    setMessages(updatedMessages);
    setPastTurnSteps(updatedPast);
    setInput("");
    setSteps([]);
    setSelectedStep(null);
    setIsGenerating(true);
    responseRef.current = "";

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          systemPrompt,
          messages: updatedMessages,
          maxTokens: 500,
          topN: 5,
          temperature,
        }),
        signal: controller.signal,
      });

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const json = line.slice(6);
          if (!json) continue;
          const event = JSON.parse(json);

          if (event.type === "token") {
            const token: GeneratedToken = {
              token: event.token,
              done: event.done,
              candidates: event.candidates,
            };
            if (!token.done) {
              responseRef.current += token.token;
            }
            setSteps((prev) => {
              const last = prev[prev.length - 1];
              if (last && last.type === "text") {
                return [
                  ...prev.slice(0, -1),
                  { ...last, tokens: [...last.tokens, token] },
                ];
              }
              return [...prev, { type: "text", tokens: [token] }];
            });
          } else if (event.type === "tool_call") {
            setSteps((prev) => [
              ...prev,
              {
                type: "tool_call",
                name: event.name,
                arguments: event.arguments,
              },
            ]);
          } else if (event.type === "tool_result") {
            setSteps((prev) => [
              ...prev,
              {
                type: "tool_result",
                name: event.name,
                result: event.result,
              },
            ]);
          }
        }
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
      inputRef.current?.focus();
    }
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
  };

  const clearConversation = () => {
    abortRef.current?.abort();
    setMessages([]);
    setPastTurnSteps([]);
    setSteps([]);
    setSelectedStep(null);
    setIsGenerating(false);
    responseRef.current = "";
    inputRef.current?.focus();
  };

  const flatTextTokens: GeneratedToken[] = steps.flatMap((s) =>
    s.type === "text" ? s.tokens : []
  );
  const selectedToken =
    selectedStep !== null ? flatTextTokens[selectedStep] : null;

  return (
    <div className="view">
      <div className="system-prompt-section">
        <div className="system-prompt-header">
          <span className="system-prompt-label">System Prompt</span>
          <button
            className="tools-toggle"
            onClick={() => setShowTools(!showTools)}
          >
            Tools ({tools.length})
          </button>
          <button
            className="view-toggle"
            onClick={() => setViewMode(viewMode === "chat" ? "raw" : "chat")}
          >
            {viewMode === "chat" ? "Show Raw" : "Show Chat"}
          </button>
        </div>
        <textarea
          className="system-prompt-textarea"
          rows={2}
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          disabled={isGenerating}
        />
        {showTools && (
          <div className="tools-panel">
            {tools.map((tool) => (
              <div key={tool.name} className="tool-definition">
                <div className="tool-def-name">{tool.name}</div>
                <div className="tool-def-description">{tool.description}</div>
                <div className="tool-def-params">
                  {Object.entries(tool.parameters).map(([param, desc]) => (
                    <div key={param} className="tool-def-param">
                      <code>{param}</code>: {desc}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {viewMode === "chat" ? (
        <div className="chat-messages" ref={chatMessagesRef}>
          {(() => {
            // Walk messages; for each assistant message, render its archived
            // step blocks (tool calls + results + text) instead of a plain
            // bubble — this preserves tool-use history across turns.
            let assistantIdx = 0;
            return messages.map((msg, i) => {
              if (msg.role === "user") {
                return (
                  <div key={i} className="chat-bubble chat-bubble-user">
                    <span className="chat-role">user</span>
                    <p>{msg.content}</p>
                  </div>
                );
              }
              const turnSteps = pastTurnSteps[assistantIdx++];
              if (turnSteps && turnSteps.length > 0) {
                return (
                  <AgentStepsBlock key={i} keyPrefix={`past-${i}`} steps={turnSteps} />
                );
              }
              return (
                <div key={i} className="chat-bubble chat-bubble-assistant">
                  <span className="chat-role">assistant</span>
                  <p>{msg.content}</p>
                </div>
              );
            });
          })()}
          <AgentStepsBlock keyPrefix="live" steps={steps} />
        </div>
      ) : (
        <RawPromptView
          systemPrompt={systemPrompt}
          messages={messages}
          agentSteps={steps}
          isGenerating={isGenerating}
          selectedStep={selectedStep}
          onSelectStep={setSelectedStep}
        />
      )}

      {selectedToken && (
        <div className="step-candidates">
          <div className="step-candidates-header">
            Step {selectedStep! + 1}: chose "{displayToken(selectedToken.token)}"
          </div>
          <CandidateList
            candidates={selectedToken.candidates}
            highlightToken={selectedToken.token}
          />
        </div>
      )}

      <div className="temperature-control">
        <label className="temperature-label">
          Temperature: <span className="temperature-value">{temperature.toFixed(1)}</span>
          <span className="temperature-hint">
            {temperature === 0 ? "(greedy)" : temperature <= 0.5 ? "(focused)" : temperature <= 1 ? "(balanced)" : "(creative)"}
          </span>
        </label>
        <input
          type="range"
          className="temperature-slider"
          min="0"
          max="2"
          step="0.1"
          value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))}
          disabled={isGenerating}
        />
      </div>

      <div className="chat-input-row">
        <input
          ref={inputRef}
          className="chat-input"
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") sendMessage();
          }}
          placeholder="Ask the agent something..."
          disabled={isGenerating}
        />
        <button
          className="generate-btn"
          onClick={sendMessage}
          disabled={isGenerating || !input.trim()}
        >
          Send
        </button>
        {isGenerating && (
          <button className="stop-btn" onClick={stopGeneration}>
            Stop
          </button>
        )}
        <button className="stop-btn" onClick={clearConversation} style={{ background: "#6b7280" }}>
          Clear
        </button>
      </div>
    </div>
  );
}

function App() {
  const [mode, setMode] = useState<"predict" | "generate" | "chat" | "agent">("agent");

  return (
    <div className="app">
      <header className="header">
        <h1>Agents Demystified</h1>
      </header>

      <nav className="tab-bar">
        <button
          className={"tab" + (mode === "predict" ? " tab-active" : "")}
          onClick={() => setMode("predict")}
        >
          Token Prediction
        </button>
        <button
          className={"tab" + (mode === "generate" ? " tab-active" : "")}
          onClick={() => setMode("generate")}
        >
          Generation Loop
        </button>
        <button
          className={"tab" + (mode === "chat" ? " tab-active" : "")}
          onClick={() => setMode("chat")}
        >
          Chat
        </button>
        <button
          className={"tab" + (mode === "agent" ? " tab-active" : "")}
          onClick={() => setMode("agent")}
        >
          Agent
        </button>
      </nav>

      <main className="main">
        {mode === "predict" && <TokenPredictionView />}
        {mode === "generate" && <GenerationLoopView />}
        {mode === "chat" && <ChatView />}
        {mode === "agent" && <AgentView />}
      </main>
    </div>
  );
}

export default App;
