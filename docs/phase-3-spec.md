# Phase 3: Chat Framing

## Goal

Add a chat interface that demonstrates how "chat" is entirely a frontend convention layered on top of raw text completion. The same `/api/generate` endpoint from Phase 2 powers everything — the server doesn't know it's having a conversation.

The UI is a side-by-side split view: a familiar chat interface on the left, and the raw assembled prompt on the right. As the user converses with the model, the audience watches the raw prompt grow with each turn — special tokens, system prompts, and role markers all visible. The core revelation: **chat models aren't a different kind of AI. They're completion models with a specific text format that creates the illusion of conversation.**

Phase 2 likely created the "steering problem" — raw completion wanders aimlessly. Phase 3 shows how chat framing solves that: structure the prompt with roles, and suddenly the model knows it's supposed to be helpful.

### Architecture: No New Backend

The frontend assembles the chat template (system prompt + message history) into a single prompt string and sends it to the existing `/api/generate`. No new endpoint. No server-side chat logic. This architectural choice IS the point — the server is still doing the same text completion it did in Phase 2. All the "chat" intelligence is just string formatting on the client.

**Prerequisite:** Phase 2 is already implemented. The `/api/generate` SSE endpoint, `TokenPredictor` service, `CandidateList` component, and generation UI all exist.

---

## Backend

No changes. The existing `/api/generate` endpoint accepts a raw prompt string and streams tokens via SSE. Phase 3's entire implementation lives in the frontend.

This is intentional and pedagogical. The audience should see that the server is still doing the same text completion it did in Phase 2. The "intelligence" of chat is in how the prompt is formatted, not in any server-side chat logic.

---

## Step 1: Chat Template Helper

Add the following types and helpers to `frontend/src/App.tsx`. These encode phi4-mini's chat template as a named constant and provide a function to assemble structured messages into a raw prompt string.

### Types and Constants

```typescript
const SPECIAL_TOKENS = {
  system: "<|system|>",
  user: "<|user|>",
  assistant: "<|assistant|>",
  end: "<|end|>",
} as const;

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}
```

### Prompt Builder

```typescript
function buildChatPrompt(
  systemPrompt: string,
  messages: ChatMessage[]
): string {
  let prompt = `${SPECIAL_TOKENS.system}\n${systemPrompt}\n${SPECIAL_TOKENS.end}\n`;
  for (const msg of messages) {
    prompt += `${SPECIAL_TOKENS[msg.role]}\n${msg.content}\n${SPECIAL_TOKENS.end}\n`;
  }
  prompt += `${SPECIAL_TOKENS.assistant}\n`;
  return prompt;
}
```

This function takes a system prompt and an array of conversation messages, and produces the full raw prompt string. Every role block is wrapped with the model's special tokens. The trailing `<|assistant|>\n` tells the model to start generating the assistant's response.

For multi-turn conversations, the prompt grows with each exchange:

```
<|system|>
You are a helpful assistant. Keep your responses brief.
<|end|>
<|user|>
Hello!
<|end|>
<|assistant|>
Hi there! How can I help?
<|end|>
<|user|>
What's 2+2?
<|end|>
<|assistant|>
```

---

## Step 2: ChatView Component

### State

```typescript
const [systemPrompt, setSystemPrompt] = useState(
  "You are a helpful assistant. Keep your responses brief."
);
const [messages, setMessages] = useState<ChatMessage[]>([]);
const [input, setInput] = useState("");
const [isGenerating, setIsGenerating] = useState(false);
const [streamingTokens, setStreamingTokens] = useState<GeneratedToken[]>([]);
const [selectedStep, setSelectedStep] = useState<number | null>(null);
const [temperature, setTemperature] = useState(0);
const abortRef = useRef<AbortController | null>(null);
const inputRef = useRef<HTMLInputElement>(null);
const responseRef = useRef("");
```

**Data model:** `messages` contains the finalized conversation history (user and assistant messages from completed turns). `streamingTokens` holds the tokens from the current/most recent generation — these are the clickable, inspectable tokens in the raw prompt view. The current assistant response text is accumulated in `responseRef` and only promoted into `messages` when the user sends their next message. This keeps the raw prompt view accurate: `messages` represents the "prompt" portion, and `streamingTokens` represents the "response" portion.

### Send Message

```typescript
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

  const prompt = buildChatPrompt(systemPrompt, updatedMessages);
  const controller = new AbortController();
  abortRef.current = controller;

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, maxTokens: 200, topN: 5, temperature }),
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
```

The SSE parsing is identical to Phase 2's `startGeneration` — because it IS the same endpoint. The only difference is that the prompt was assembled by `buildChatPrompt` instead of typed directly by the user.

### Stop & Clear

```typescript
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
```

### UI Layout

```
ChatView (left column of split view)
├── SystemPromptInput (labeled textarea, ~2 rows, disabled during generation)
├── ChatMessages (scrollable message list)
│   ├── UserBubble (right-aligned, blue-ish background)
│   ├── AssistantBubble (left-aligned, gray background)
│   └── StreamingBubble (assistant bubble showing current generation in progress)
├── TemperatureControl (slider 0–2, step 0.1, same behavior as Phase 2)
└── InputRow
    ├── TextInput (placeholder "Type a message...", Enter sends, disabled during generation)
    ├── SendButton (disabled during generation or empty input)
    ├── StopButton (visible only during generation)
    └── ClearButton (resets conversation, disabled during generation)
```

### System Prompt Input

```tsx
<div className="system-prompt-section">
  <label className="system-prompt-label">System Prompt</label>
  <textarea
    className="system-prompt-textarea"
    value={systemPrompt}
    onChange={(e) => setSystemPrompt(e.target.value)}
    rows={2}
    disabled={isGenerating}
  />
</div>
```

### Chat Messages Rendering

```tsx
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
        {streamingTokens.some((t) => t.done) && (
          <span className="eos-token">
            {streamingTokens.find((t) => t.done)?.token || "⏹"}
          </span>
        )}
      </p>
    </div>
  )}
</div>
```

Auto-scroll the chat messages container as tokens arrive:

```typescript
const chatMessagesRef = useRef<HTMLDivElement>(null);

useEffect(() => {
  if (chatMessagesRef.current) {
    chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
  }
}, [streamingTokens, messages]);
```

### Input Handling

```tsx
<div className="chat-input-row">
  <input
    ref={inputRef}
    type="text"
    className="chat-input"
    placeholder="Type a message..."
    value={input}
    onChange={(e) => setInput(e.target.value)}
    onKeyDown={(e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    }}
    disabled={isGenerating}
  />
  <button onClick={sendMessage} disabled={isGenerating || !input.trim()}>
    Send
  </button>
  {isGenerating && <button onClick={stopGeneration}>Stop</button>}
  <button onClick={clearConversation} disabled={isGenerating}>
    Clear
  </button>
</div>
```

---

## Step 3: RawPromptView Component

The raw prompt view is the "behind the curtain" panel. It renders the full prompt — exactly as sent to `/api/generate` — with highlighted and labeled sections for each role block. Generated tokens in the current assistant section are visually distinct and clickable for candidate inspection.

### RawSection Helper

Renders a single completed role block (system, user, or a past assistant turn) with special token highlighting and a section label.

```typescript
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
```

### RawPromptView Component

```typescript
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
              {gt.done ? gt.token || "⏹" : gt.token}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
```

**Section rendering logic:**

- Before any messages are sent: only the system section appears. The audience sees the starting state.
- After the user sends a message: system + user section + assistant section with `<|assistant|>` tag (tokens start streaming in).
- After generation completes: same view, but streaming has stopped. The assistant's tokens remain clickable for inspection.
- After the user sends a follow-up: previous response is finalized into `messages` (renders as a static `RawSection`), new user section appears, new assistant section starts streaming.

This means the raw view always accurately reflects what was sent to `/api/generate` plus what came back.

### Candidate Inspector

Below the raw prompt container in the right column, show the `CandidateList` when a generated token is selected:

```tsx
{selectedStep !== null && streamingTokens[selectedStep] && (
  <div className="step-candidates">
    <h4>
      Step {selectedStep + 1}: chose "
      {displayToken(streamingTokens[selectedStep].token)}"
    </h4>
    <CandidateList
      candidates={streamingTokens[selectedStep].candidates}
      highlightToken={streamingTokens[selectedStep].token}
    />
  </div>
)}
```

This reuses the shared `CandidateList` from Phase 2. The audience can click any generated token in the raw prompt's assistant section to see what alternatives the model considered — reinforcing that it's still the same prediction loop under the hood.

---

## Step 4: Navigation and Layout

### Tab Bar

Add a third tab to the existing tab bar:

- **Token Prediction** — Phase 1
- **Generation Loop** — Phase 2
- **Chat** — Phase 3

Update the `mode` state type:

```typescript
const [mode, setMode] = useState<"predict" | "generate" | "chat">("chat");
```

Default to `"chat"` so Phase 3 is immediately visible when the tab opens.

### Split Layout

The Chat tab renders a two-column grid: chat on the left, raw prompt on the right.

```tsx
{mode === "chat" && (
  <div className="chat-split-view">
    <div className="chat-column">
      {/* SystemPromptInput */}
      {/* ChatMessages */}
      {/* TemperatureControl */}
      {/* InputRow */}
    </div>
    <div className="raw-column">
      <h3 className="raw-column-header">Raw Prompt</h3>
      <RawPromptView
        systemPrompt={systemPrompt}
        messages={messages}
        streamingTokens={streamingTokens}
        isGenerating={isGenerating}
        selectedStep={selectedStep}
        onSelectStep={setSelectedStep}
      />
      {/* CandidateList below when token selected */}
    </div>
  </div>
)}
```

### File: `frontend/src/App.tsx`

Replace the entire contents. The file should contain:

1. Shared types (`TokenCandidate`, `PredictResponse`, `GeneratedToken`, `ChatMessage`), `SPECIAL_TOKENS` constant, `displayToken` helper, `buildChatPrompt` helper, and `CandidateList` component
2. **Tab bar** with three tabs (default to "Chat")
3. **TokenPredictionView** — Phase 1's autocomplete UI (unchanged from Phase 2)
4. **GenerationLoopView** — Phase 2's streaming generation UI (unchanged from Phase 2)
5. **ChatView** — Phase 3's chat interface (left column)
6. **RawPromptView** — Phase 3's raw prompt panel (right column)
7. **RawSection** — helper for rendering completed role blocks

---

## Step 5: Styling

### File: `frontend/src/App.css`

Extend (do not replace) the existing styles with classes for the split view, chat bubbles, raw prompt sections, and controls.

#### Split View Layout

```css
.chat-split-view {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  height: calc(100vh - 80px);
}

.chat-column,
.raw-column {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
```

#### Chat Bubbles

- `.chat-messages`: flex column, `overflow-y: auto`, `flex: 1`, gap `0.5rem` between messages, padding `0.5rem`
- `.chat-bubble`: padding `0.75rem 1rem`, border-radius `0.75rem`, max-width `80%`, word-wrap `break-word`
- `.chat-bubble-user`: align-self `flex-end`, background `#dbeafe` (light blue)
- `.chat-bubble-assistant`: align-self `flex-start`, background `#f3f4f6` (light gray)
- `.chat-role`: display `block`, font-size `0.65rem`, font-weight `bold`, text-transform `uppercase`, letter-spacing `0.05em`, color `#6b7280`, margin-bottom `0.25rem`
- `.chat-bubble p`: margin `0`, white-space `pre-wrap`

#### System Prompt

- `.system-prompt-section`: margin-bottom `0.5rem`
- `.system-prompt-label`: display `block`, font-size `0.75rem`, font-weight `bold`, text-transform `uppercase`, letter-spacing `0.05em`, color `#6b7280`, margin-bottom `0.25rem`
- `.system-prompt-textarea`: width `100%`, font-family monospace, font-size `0.85rem`, padding `0.5rem`, border `1px solid #d1d5db`, border-radius `0.375rem`, resize `vertical`, box-sizing `border-box`
- Disabled state: opacity `0.5`

#### Chat Input Row

- `.chat-input-row`: display `flex`, gap `0.5rem`, padding-top `0.5rem`
- `.chat-input`: flex `1`, padding `0.5rem`, font-family monospace, border `1px solid #d1d5db`, border-radius `0.375rem`
- Buttons: consistent with Phase 2 button styling

#### Raw Prompt View

- `.raw-prompt-container`: font-family monospace, font-size `0.8rem`, white-space `pre-wrap`, overflow-y `auto`, flex `1`, padding `0.75rem`, background `#1e1e1e` (dark), color `#d4d4d4` (light text), border-radius `0.375rem`, line-height `1.4`
- `.raw-column-header`: margin `0 0 0.5rem 0`, font-size `0.85rem`, color `#6b7280`, text-transform `uppercase`, letter-spacing `0.05em`

#### Raw Prompt Sections

Each section has a left border and faint background tint that identifies the role, plus a small label in the corner:

- `.raw-section`: position `relative`, padding `0.25rem 0.5rem`, margin `0.125rem 0`, border-left `3px solid`
- `.raw-section-system`: border-left-color `#3b82f6` (blue), background `rgba(59, 130, 246, 0.08)`
- `.raw-section-user`: border-left-color `#22c55e` (green), background `rgba(34, 197, 94, 0.08)`
- `.raw-section-assistant`: border-left-color `#a855f7` (purple), background `rgba(168, 85, 247, 0.08)`
- `.raw-section-label`: position `absolute`, top `0.125rem`, right `0.5rem`, font-size `0.625rem`, text-transform `uppercase`, letter-spacing `0.05em`, opacity `0.6`, matching the section's border color (blue for system, green for user, purple for assistant)
- `.raw-special-token`: color `#6b7280` (gray), font-size `0.85em`

#### Generated Tokens in Raw View

Generated tokens in the assistant section are visually distinct from the static prompt text — they're interactive and inspectable:

- `.generated-token-span` (within raw view): cursor `pointer`, border-radius `2px`, transition `background 0.15s`
- `.generated-token-span:hover`: background `rgba(168, 85, 247, 0.2)` (purple tint)
- `.selected-token`: background `rgba(168, 85, 247, 0.3)`, outline `1px solid #a855f7`
- `.eos-token` styles carry over from Phase 2 (red text `#ef4444`, light red background `#fef2f2`, red border `#fecaca`, smaller font `0.75rem`, bold, pill shape with padding and border-radius)

#### Temperature Control

Reuse the same temperature slider styling from Phase 2 (range 0–2, step 0.1, label with hint text, disabled during generation).

---

## Verification

1. Run `dotnet build` — must succeed with zero errors (no backend changes, but verify nothing broke)
2. Run `cd frontend && npx tsc --noEmit` — no type errors
3. Open the frontend — tab bar should show "Token Prediction", "Generation Loop", and "Chat"
4. Phase 1 tab ("Token Prediction") still works exactly as before
5. Phase 2 tab ("Generation Loop") still works exactly as before
6. Switch to "Chat" tab — split view appears with chat on the left, raw prompt on the right
7. System prompt is pre-filled with "You are a helpful assistant. Keep your responses brief."
8. System prompt textarea is editable (before generation starts)
9. Raw prompt view shows just the system section before any messages are sent
10. Type "Hello!" and press Enter — message appears as a right-aligned user bubble in the chat view
11. Raw prompt view immediately shows: system section + user section + `<|assistant|>` tag
12. Tokens stream into the assistant chat bubble AND the raw prompt's assistant section simultaneously
13. Raw prompt auto-scrolls to keep the latest tokens visible during streaming
14. Generation ends with a visible `<|endoftext|>` badge in both views
15. After generation completes, the input field auto-focuses
16. Send a follow-up message — the raw prompt now shows the full multi-turn history (system → user → assistant → user → assistant)
17. Verify the raw prompt grows correctly with each turn — all special tokens visible
18. Click a generated token in the raw prompt's assistant section — it highlights and the candidate list appears below in the right column
19. The chosen token is highlighted in the candidate list; at temperature 0 it is always the #1 candidate
20. Only assistant-generated tokens are clickable — the static prompt sections (system, user, past assistant turns) are not interactive
21. Click "Stop" mid-generation — streaming stops cleanly in both views
22. Click "Clear" — conversation resets, raw prompt returns to showing just the system section
23. Change the system prompt to "You are a pirate. Respond in pirate speak." and send a message — verify the model's behavior changes and the raw prompt shows the new system prompt
24. Temperature slider works: 0 = deterministic, higher values = more variety between runs
25. Verify that the browser's network tab shows calls to `/api/generate` only — no new backend endpoints
