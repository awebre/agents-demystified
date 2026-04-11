# Phase 3: Chat Framing

## Goal

Add a `POST /api/chat` endpoint and a chat UI that demonstrates how "chat" is just a text formatting convention over raw completion. The server assembles a chat template from structured messages and runs the same generation loop from Phase 2 — the audience sees that `/api/chat` is just `/api/generate` with a prompt builder in front of it.

The UI has a toggle between **Chat view** (familiar chat bubbles) and **Raw view** (the actual prompt string with section highlighting). The audience flips back and forth and realizes: the polished chat interface they're used to is just a pretty face on a formatted text string.

### Architecture: Each Phase Composes the Previous

The progression of endpoints tells the whole story:

- **Phase 1:** `/api/predict` — predict one token
- **Phase 2:** `/api/generate` — prediction in a loop (composes `TokenPredictor`)
- **Phase 3:** `/api/chat` — chat template + generation loop (composes `TokenPredictor` + `ChatTemplateBuilder`)

Each endpoint is a thin layer over the previous abstraction. The audience walks through the backend code and sees how simple each step is.

**Prerequisite:** Phase 2 is already implemented. The `/api/generate` SSE endpoint, `TokenPredictor` service, `CandidateList` component, and generation UI all exist.

---

## Backend

### Step 1: Chat Template Builder

#### File: `AgentsDemystified.Server/ChatTemplateBuilder.cs` (new file)

Encodes phi4-mini's chat template format. The special token constants are public so they can be reused by Phase 4.

```csharp
namespace AgentsDemystified.Server;

public static class ChatTemplateBuilder
{
    public const string SystemToken = "<|system|>";
    public const string UserToken = "<|user|>";
    public const string AssistantToken = "<|assistant|>";
    public const string EndToken = "<|end|>";

    /// <summary>
    /// Assemble a chat prompt from a system prompt and message history.
    /// This is the entire "trick" of chat — it's just string concatenation
    /// with special tokens.
    /// </summary>
    public static string BuildPrompt(string systemPrompt, List<ChatMessage> messages)
    {
        var prompt = $"{SystemToken}\n{systemPrompt}\n{EndToken}\n";
        foreach (var msg in messages)
        {
            var roleToken = msg.Role == "user" ? UserToken : AssistantToken;
            prompt += $"{roleToken}\n{msg.Content}\n{EndToken}\n";
        }
        prompt += $"{AssistantToken}\n";
        return prompt;
    }
}
```

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

### Step 2: `/api/chat` Endpoint

#### File: `AgentsDemystified.Server/Program.cs`

#### New Records

```csharp
record ChatMessage(string Role, string Content);

record ChatRequest(
    string SystemPrompt,
    List<ChatMessage> Messages,
    int MaxTokens = 200,
    int TopN = 5,
    double Temperature = 0);
```

#### Required Usings

Add at the top (if not already present):

```csharp
using AgentsDemystified.Server;
```

#### Endpoint

The endpoint is structurally identical to `/api/generate` — the only new thing is `ChatTemplateBuilder.BuildPrompt` assembling the prompt from structured messages. That's the point: chat is just a prompt format.

SSE events include a `type` field to distinguish event kinds. Phase 3 only has `"token"` events, but the field sets up Phase 4's additional event types (`tool_call`, `tool_result`).

```csharp
api.MapPost("/chat", async (ChatRequest request, TokenPredictor predictor, HttpContext context) =>
{
    context.Response.ContentType = "text/event-stream";
    context.Response.Headers.CacheControl = "no-cache";
    context.Response.Headers.Connection = "keep-alive";

    // This is the entire "chat" logic — build a prompt string from messages
    var prompt = ChatTemplateBuilder.BuildPrompt(request.SystemPrompt, request.Messages);

    for (var i = 0; i < request.MaxTokens; i++)
    {
        context.RequestAborted.ThrowIfCancellationRequested();

        // Same prediction call as Phase 2 — nothing new here
        var result = await predictor.PredictNextAsync(
            prompt, request.TopN, request.Temperature, context.RequestAborted);

        var payload = JsonSerializer.Serialize(new
        {
            type = "token",
            token = result.Token,
            done = result.Done,
            candidates = result.Candidates
        }, JsonSerializerOptions.Web);

        await context.Response.WriteAsync($"data: {payload}\n\n");
        await context.Response.Body.FlushAsync();

        if (result.Done || string.IsNullOrEmpty(result.Token))
            break;

        prompt += result.Token;
    }
})
.WithName("ChatStream")
.WithDescription("Chat with the model using structured messages — Phase 2's generation loop with a chat template");
```

---

## Frontend

### Step 3: Update ChatView to Use `/api/chat`

The `ChatView` component changes from building the prompt client-side and calling `/api/generate` to sending structured messages to `/api/chat`. The `buildChatPrompt` function is removed from the frontend API call path — the server handles prompt assembly now.

#### State

Same as before, but `responseRef` tracks the current assistant response for promoting to `messages` on the next send:

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
const [viewMode, setViewMode] = useState<"chat" | "raw">("chat");
const abortRef = useRef<AbortController | null>(null);
const inputRef = useRef<HTMLInputElement>(null);
const responseRef = useRef("");
```

#### Send Message

The key change: send structured messages to `/api/chat` instead of a raw prompt to `/api/generate`.

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
```

#### Stop & Clear

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

### Step 4: Chat/Raw Toggle

A toggle button in the header area of the chat view swaps the message display between chat bubbles and raw prompt view. The system prompt input, temperature slider, and message input stay visible in both modes — only the message area switches.

#### Toggle Button

```tsx
<button
  className="view-toggle"
  onClick={() => setViewMode(viewMode === "chat" ? "raw" : "chat")}
>
  {viewMode === "chat" ? "Show Raw" : "Show Chat"}
</button>
```

Place the toggle button next to the "System Prompt" label, right-aligned.

#### Message Area

```tsx
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
          {streamingTokens.some((t) => t.done) && (
            <span className="eos-token">
              {streamingTokens.find((t) => t.done)?.token || "\u23F9"}
            </span>
          )}
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
```

### RawPromptView and RawSection

The raw prompt view reconstructs the prompt from the messages array client-side. It renders each role block with section highlighting (colored left borders and faint backgrounds) and labels. The `SPECIAL_TOKENS` constant and these components remain in the frontend for display purposes.

**RawSection** renders a completed role block:

```typescript
const SPECIAL_TOKENS = {
  system: "<|system|>",
  user: "<|user|>",
  assistant: "<|assistant|>",
  end: "<|end|>",
} as const;

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

**RawPromptView** assembles sections and renders streaming tokens as clickable spans:

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
              {gt.done ? gt.token || "\u23F9" : gt.token}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
```

**Candidate inspector** shown below the message area when a token is selected (works in both Chat and Raw view mode):

```tsx
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
```

### Step 5: Navigation

Tab bar with three tabs, default to "Chat":

```typescript
const [mode, setMode] = useState<"predict" | "generate" | "chat">("chat");
```

- **Token Prediction** — Phase 1
- **Generation Loop** — Phase 2
- **Chat** — Phase 3

### UI Layout

```
ChatView (single column, no split)
├── Header Row
│   ├── SystemPromptLabel ("System Prompt")
│   └── ViewToggle ("Show Raw" / "Show Chat")
├── SystemPromptTextarea (editable, disabled during generation)
├── MessageArea (swapped by toggle)
│   ├── [Chat mode] ChatMessages (bubbles + streaming bubble)
│   └── [Raw mode] RawPromptView (section-highlighted prompt)
├── CandidateInspector (shown when a token is selected, either mode)
├── TemperatureControl (slider 0-2, same as Phase 2)
└── InputRow
    ├── TextInput (Enter sends, disabled during generation)
    ├── SendButton
    ├── StopButton (visible during generation)
    └── ClearButton
```

### File: `frontend/src/App.tsx`

Replace the entire contents. The file should contain:

1. Shared types (`TokenCandidate`, `PredictResponse`, `GeneratedToken`, `ChatMessage`), `SPECIAL_TOKENS` constant, `displayToken` helper, and `CandidateList` component
2. **Tab bar** with three tabs (default to "Chat")
3. **TokenPredictionView** — Phase 1 (unchanged)
4. **GenerationLoopView** — Phase 2 (unchanged)
5. **ChatView** — Phase 3's chat interface with toggle
6. **RawPromptView** + **RawSection** — raw prompt display components

---

## Styling

### File: `frontend/src/App.css`

Extend (do not replace) existing styles. Remove the `.chat-split-view` grid layout from the current Phase 3 implementation. The chat view is now single-column.

#### View Toggle

- `.view-toggle`: small button, font-size `0.75rem`, padding `0.25rem 0.75rem`, border `1px solid #d1d5db`, border-radius `0.375rem`, background `white`, cursor `pointer`, float `right`
- `.view-toggle:hover`: background `#f9fafb`

#### Chat Messages

- `.chat-messages`: flex column, `overflow-y: auto`, `flex: 1`, `min-height: 300px`, `max-height: 50vh`, gap `0.5rem`, padding `0.5rem`
- `.chat-bubble`: padding `0.75rem 1rem`, border-radius `0.75rem`, max-width `80%`, word-wrap `break-word`
- `.chat-bubble-user`: align-self `flex-end`, background `#dbeafe`
- `.chat-bubble-assistant`: align-self `flex-start`, background `#f3f4f6`
- `.chat-role`: display `block`, font-size `0.65rem`, font-weight `bold`, text-transform `uppercase`, letter-spacing `0.05em`, color `#6b7280`, margin-bottom `0.25rem`
- `.chat-bubble p`: margin `0`, white-space `pre-wrap`

#### System Prompt

- `.system-prompt-section`: margin-bottom `0.5rem`
- `.system-prompt-header`: display `flex`, justify-content `space-between`, align-items `center`, margin-bottom `0.25rem`
- `.system-prompt-label`: font-size `0.75rem`, font-weight `bold`, text-transform `uppercase`, letter-spacing `0.05em`, color `#6b7280`
- `.system-prompt-textarea`: width `100%`, font-family monospace, font-size `0.85rem`, padding `0.5rem`, border `1px solid #d1d5db`, border-radius `0.375rem`, resize `vertical`, box-sizing `border-box`
- Disabled state: opacity `0.5`

#### Chat Input Row

- `.chat-input-row`: display `flex`, gap `0.5rem`, padding-top `0.5rem`
- `.chat-input`: flex `1`, padding `0.5rem`, font-family monospace, border `1px solid #d1d5db`, border-radius `0.375rem`

#### Raw Prompt View

- `.raw-prompt-container`: font-family monospace, font-size `0.8rem`, white-space `pre-wrap`, overflow-y `auto`, `min-height: 300px`, `max-height: 50vh`, padding `0.75rem`, background `#1e1e1e`, color `#d4d4d4`, border-radius `0.375rem`, line-height `1.4`

#### Raw Prompt Sections

- `.raw-section`: position `relative`, padding `0.25rem 0.5rem`, margin `0.125rem 0`, border-left `3px solid`
- `.raw-section-system`: border-left-color `#3b82f6`, background `rgba(59, 130, 246, 0.08)`
- `.raw-section-user`: border-left-color `#22c55e`, background `rgba(34, 197, 94, 0.08)`
- `.raw-section-assistant`: border-left-color `#a855f7`, background `rgba(168, 85, 247, 0.08)`
- `.raw-section-label`: position `absolute`, top `0.125rem`, right `0.5rem`, font-size `0.625rem`, text-transform `uppercase`, letter-spacing `0.05em`, opacity `0.6`, color matching section border
- `.raw-special-token`: color `#6b7280`, font-size `0.85em`
- `.raw-prompt-container .generated-token-span:hover`: background `rgba(168, 85, 247, 0.2)`
- `.raw-prompt-container .selected-token`: background `rgba(168, 85, 247, 0.3)`, outline `1px solid #a855f7`

#### Mobile

- At `max-width: 768px`: tighten padding, `font-size: 16px` on inputs (prevent iOS zoom), adjust chat message max-height

---

## Verification

1. Run `dotnet build` — must succeed with zero errors
2. Run `cd frontend && npx tsc --noEmit` — no type errors
3. Open the frontend — tab bar shows "Token Prediction", "Generation Loop", and "Chat"
4. Phase 1 and Phase 2 tabs still work exactly as before
5. Switch to "Chat" tab — single-column chat view with system prompt and input
6. System prompt is pre-filled with "You are a helpful assistant. Keep your responses brief."
7. Type "Hello!" and press Enter — user bubble appears, tokens stream into an assistant bubble
8. After generation, input auto-focuses
9. Send a follow-up message — multi-turn works, previous messages appear in history
10. Click "Show Raw" toggle — the message area switches to the raw prompt view
11. Raw view shows section-highlighted blocks: system (blue), user (green), assistant (purple)
12. Special tokens (`<|system|>`, `<|end|>`, etc.) are visible as gray text within sections
13. The current assistant generation shows as clickable token spans in the assistant section
14. Click a generated token — candidate list appears below with the chosen token highlighted
15. Toggle back to "Show Chat" — chat bubbles reappear, candidate inspector still works
16. Click "Stop" mid-generation — streaming stops cleanly
17. Click "Clear" — conversation resets
18. Change system prompt and verify the model's behavior changes
19. Temperature slider works: 0 = deterministic, higher = varied
20. Verify `/api/chat` appears in Scalar API docs at `/scalar/v1`
21. Verify `/api/chat` is the only new endpoint — `/api/predict` and `/api/generate` still work
