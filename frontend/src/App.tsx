import { useState, useEffect, useRef } from "react";
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

function RawPromptView({
  systemPrompt,
  messages,
  streamingTokens,
  isGenerating,
  selectedStep,
  onSelectStep,
}: {
  systemPrompt: string;
  messages: ChatMessage[];
  streamingTokens: GeneratedToken[];
  isGenerating: boolean;
  selectedStep: number | null;
  onSelectStep: (index: number | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [streamingTokens]);

  const hasActiveGeneration = isGenerating || streamingTokens.length > 0;

  return (
    <div className="raw-prompt-container" ref={containerRef}>
      <RawSection role="system" content={systemPrompt} />
      {messages.map((msg, i) => (
        <RawSection key={i} role={msg.role} content={msg.content} />
      ))}
      {hasActiveGeneration && (
        <div className="raw-section raw-section-assistant">
          <span className="raw-section-label">assistant</span>
          <span className="raw-special-token">
            {SPECIAL_TOKENS.assistant}
          </span>
          {"\n"}
          {streamingTokens.map((gt, index) => (
            <span
              key={index}
              className={
                "generated-token-span" +
                (selectedStep === index ? " selected-token" : "") +
                (gt.done ? " eos-token" : "")
              }
              onClick={() =>
                onSelectStep(selectedStep === index ? null : index)
              }
            >
              {gt.done ? gt.token || "\u23F9" : gt.token}
            </span>
          ))}
        </div>
      )}
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

function App() {
  const [mode, setMode] = useState<"predict" | "generate" | "chat">("chat");

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
      </nav>

      <main className="main">
        {mode === "predict" && <TokenPredictionView />}
        {mode === "generate" && <GenerationLoopView />}
        {mode === "chat" && <ChatView />}
      </main>
    </div>
  );
}

export default App;
