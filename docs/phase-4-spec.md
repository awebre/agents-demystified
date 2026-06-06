# Phase 4: Tool Calling

## Goal

Add a `POST /api/agent` endpoint that runs a full agent loop: generate text, detect tool calls, execute tools, inject results back into the prompt, and continue generating. The audience sees that an "AI agent" is just a while loop with some string parsing.

This is the capstone phase. The complete progression:

- **Phase 1:** `/api/predict` — predict one token
- **Phase 2:** `/api/generate` — prediction in a loop
- **Phase 3:** `/api/chat` — generation loop with a chat template
- **Phase 4:** `/api/agent` — chat with a tool-calling loop

Each endpoint composes the previous layer. The audience can trace the entire path from a single token prediction to an agent that searches and reads files.

### Two-Beat Reveal

1. **Tools are just text.** The model doesn't "execute" anything — it generates text that happens to match a schema (`<tool_call>{"name": "search_files", ...}</tool_call>`). The system prompt told it to do this. It's still just next-token prediction.
2. **The agent loop.** When the system detects that text pattern, it parses the JSON, calls a function, and pastes the result back into the prompt. Then generation continues. An "agent" is just this loop: generate → parse → execute → inject → generate.

### Tools

Two tools that operate on the actual project source code:

- **`search_files`** — search for files matching a query (name or content). Returns matching file paths.
- **`read_file`** — read the contents of a file. Returns the file text.

The agent can inspect its own codebase. "How does the predict endpoint work?" triggers a search → read → answer chain. Meta, memorable, and immediately verifiable by the audience.

**Prerequisite:** Phase 3 is already implemented. The `/api/chat` endpoint, `ChatTemplateBuilder`, and chat UI with raw view toggle all exist.

---

## Backend

### Step 1: Tool Registry

#### File: `AgentsDemystified.Server/ToolRegistry.cs` (new file)

The tool registry defines available tools, generates the prompt instructions, and executes tool calls. Tools are simple C# methods — no framework, no abstraction. The audience can read the code and see there's nothing magical.

```csharp
using System.Text.Json;

namespace AgentsDemystified.Server;

public record ToolDefinition(string Name, string Description, Dictionary<string, string> Parameters);

public class ToolRegistry
{
    private readonly string projectRoot;

    public ToolRegistry(string projectRoot)
    {
        this.projectRoot = projectRoot;
    }

    public List<ToolDefinition> GetTools() =>
    [
        new("search_files",
            "Search for files in the project whose name or content matches the query. Returns matching file paths.",
            new() { ["query"] = "The search term to match against file names and contents" }),
        new("read_file",
            "Read the full contents of a file. Returns the file text.",
            new() { ["path"] = "File path relative to the project root" }),
    ];

    /// <summary>
    /// Generate the tool instruction block for injection into the system prompt.
    /// This is how the model "learns" about tools — just text in the prompt.
    /// </summary>
    public string GetToolPromptSection()
    {
        var section = "\n\nYou have access to the following tools:\n\n";
        foreach (var tool in GetTools())
        {
            section += $"## {tool.Name}\n{tool.Description}\n";
            section += "Parameters: " + JsonSerializer.Serialize(tool.Parameters) + "\n\n";
        }
        // This section is intentionally MECHANICAL ONLY — it teaches the model
        // the tool-call SYNTAX and nothing about when or whether to use the
        // tools. All behavior steering (when to call, search→read chaining, "you
        // can read files") lives in the user-editable System Prompt. See the
        // "Tool prompt vs system prompt" note below for why this split is a
        // deliberate teaching beat.
        section += """
            To call a tool, output EXACTLY this format on its own line and NOTHING else in that turn — no prose before or after:
            <tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>

            Fill in every required argument with a concrete value; never emit empty arguments. After each tool call you will receive a <tool_result>...</tool_result> containing the output.
            """;
        return section;
    }

    public async Task<string> ExecuteAsync(string name, JsonElement arguments)
    {
        return name switch
        {
            "search_files" => await SearchFiles(arguments.GetProperty("query").GetString()!),
            "read_file" => await ReadFile(arguments.GetProperty("path").GetString()!),
            _ => JsonSerializer.Serialize(new { error = $"Unknown tool: {name}" })
        };
    }

    private Task<string> SearchFiles(string query)
    {
        // Broad, forgiving match: split the query into terms and match any file
        // whose path or content contains ANY term, ranked by how many distinct
        // terms it matches. This lets the model get away with sloppy natural-
        // language queries (e.g. "ChatTemplateBuilder special tokens" still finds
        // ChatTemplateBuilder.cs) instead of having to guess a single exact
        // substring. See "Forgiving search" below for why we chose breadth.
        var terms = query
            .Split([' ', '\t', '\n', ',', '.', ':', ';', '"', '\'', '(', ')', '[', ']', '{', '}', '/', '\\'],
                StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(t => t.ToLowerInvariant())
            .Distinct()
            .ToArray();

        if (terms.Length == 0)
            return Task.FromResult(JsonSerializer.Serialize(new { files = Array.Empty<string>() }));

        var scored = new List<(string Path, int Score)>();
        var searchDir = new DirectoryInfo(projectRoot);

        foreach (var file in searchDir.EnumerateFiles("*", SearchOption.AllDirectories))
        {
            // Skip build artifacts, hidden directories, node_modules
            var relativePath = Path.GetRelativePath(projectRoot, file.FullName);
            if (relativePath.Contains("/bin/") || relativePath.Contains("/obj/") ||
                relativePath.Contains("/node_modules/") || relativePath.StartsWith("."))
                continue;

            // Match against path + content (small files only); count distinct terms hit
            var haystack = relativePath.ToLowerInvariant();
            if (file.Length < 100_000)
            {
                try { haystack += "\n" + File.ReadAllText(file.FullName).ToLowerInvariant(); }
                catch { /* skip unreadable content, still match on path */ }
            }

            var score = terms.Count(haystack.Contains);
            if (score > 0)
                scored.Add((relativePath, score));
        }

        var files = scored
            .OrderByDescending(x => x.Score)
            .ThenBy(x => x.Path)
            .Select(x => x.Path)
            .Take(10);

        return Task.FromResult(JsonSerializer.Serialize(new { files }));
    }

    private Task<string> ReadFile(string path)
    {
        // Prevent path traversal — must stay within project root
        var fullPath = Path.GetFullPath(Path.Combine(projectRoot, path));
        if (!fullPath.StartsWith(projectRoot))
            return Task.FromResult(JsonSerializer.Serialize(new { error = "Path outside project directory" }));

        if (!File.Exists(fullPath))
            return Task.FromResult(JsonSerializer.Serialize(new { error = $"File not found: {path}" }));

        // Truncate large files to keep prompt manageable
        var content = File.ReadAllText(fullPath);
        if (content.Length > 5000)
            content = content[..5000] + "\n... (truncated)";

        return Task.FromResult(JsonSerializer.Serialize(new { path, content }));
    }
}
```

**Security notes:**
- `ReadFile` validates the resolved path stays within the project root (prevents `../../etc/passwd` style attacks)
- `SearchFiles` skips `bin/`, `obj/`, `node_modules/`, and hidden directories
- Large files are truncated to 5000 characters to keep the prompt manageable

### Step 2: `/api/agent` Endpoint

#### File: `AgentsDemystified.Server/Program.cs`

#### Register ToolRegistry

Add after the existing `builder.Services.AddTransient<TokenPredictor>()`:

```csharp
builder.Services.AddSingleton(new ToolRegistry(builder.Environment.ContentRootPath));
```

#### New Records

```csharp
record AgentRequest(
    string SystemPrompt,
    List<ChatMessage> Messages,
    int MaxTokens = 500,
    int TopN = 5,
    double Temperature = 0,
    int MaxToolCalls = 5);

record ToolCallParsed(string Name, JsonElement Arguments);
```

#### Tool Call Detection Helper

```csharp
static ToolCallParsed? ParseToolCall(string text)
{
    var startTag = "<tool_call>";
    var endTag = "</tool_call>";
    var startIdx = text.LastIndexOf(startTag);
    if (startIdx < 0) return null;
    var endIdx = text.IndexOf(endTag, startIdx);
    if (endIdx < 0) return null;

    var json = text[(startIdx + startTag.Length)..endIdx].Trim();
    try
    {
        var doc = JsonDocument.Parse(json);
        var name = doc.RootElement.GetProperty("name").GetString()!;
        var args = doc.RootElement.GetProperty("arguments");
        return new ToolCallParsed(name, args);
    }
    catch
    {
        return null;
    }
}
```

#### SSE Helper

```csharp
static async Task StreamEvent(HttpContext context, object data)
{
    var payload = JsonSerializer.Serialize(data, JsonSerializerOptions.Web);
    await context.Response.WriteAsync($"data: {payload}\n\n");
    await context.Response.Body.FlushAsync();
}
```

#### Endpoint

The agent loop is right here in the endpoint — no framework, no abstraction. The audience reads it top to bottom and sees the entire agent:

```csharp
api.MapPost("/agent", async (AgentRequest request, TokenPredictor predictor, ToolRegistry tools, HttpContext context) =>
{
    context.Response.ContentType = "text/event-stream";
    context.Response.Headers.CacheControl = "no-cache";
    context.Response.Headers.Connection = "keep-alive";

    // Inject tool definitions into the system prompt — this is how the model "learns" about tools
    var systemPromptWithTools = request.SystemPrompt + tools.GetToolPromptSection();
    var prompt = ChatTemplateBuilder.BuildPrompt(systemPromptWithTools, request.Messages);

    var toolCallCount = 0;

    // Guardrail: remember which tool calls we already ran this turn so we can
    // short-circuit duplicates. See "Duplicate tool-call guardrail" below for
    // why this is essential with a small local model.
    var executedCalls = new HashSet<string>();

    // THE AGENT LOOP — generate, detect tool call, execute, inject result, repeat
    while (toolCallCount <= request.MaxToolCalls)
    {
        var generatedText = "";

        // Inner generation loop — same as /api/chat, token by token
        for (var i = 0; i < request.MaxTokens; i++)
        {
            context.RequestAborted.ThrowIfCancellationRequested();

            var result = await predictor.PredictNextAsync(
                prompt, request.TopN, request.Temperature, context.RequestAborted);

            await StreamEvent(context, new
            {
                type = "token",
                token = result.Token,
                done = result.Done,
                candidates = result.Candidates
            });

            if (result.Done || string.IsNullOrEmpty(result.Token))
                break;

            prompt += result.Token;
            generatedText += result.Token;

            // Check if the model just completed a tool call
            if (generatedText.Contains("</tool_call>"))
                break;
        }

        // Try to parse a tool call from the generated text
        var toolCall = ParseToolCall(generatedText);
        if (toolCall is null)
            break; // No tool call — model is done, exit the agent loop

        toolCallCount++;

        // Stream the tool call event so the frontend can show it
        await StreamEvent(context, new
        {
            type = "tool_call",
            name = toolCall.Name,
            arguments = toolCall.Arguments
        });

        // Duplicate guardrail: if the model re-issued a call we already ran this
        // turn, do NOT re-execute it — that re-injects the full result and bloats
        // the prompt until an Ollama call times out and the stream dies. Feed
        // back a short nudge instead so the model answers. (Counts toward the
        // tool-call cap, so the loop still terminates.)
        var callKey = $"{toolCall.Name}:{toolCall.Arguments.GetRawText()}";
        if (!executedCalls.Add(callKey))
        {
            var nudge = JsonSerializer.Serialize(new
            {
                note = $"You already called {toolCall.Name} with these arguments and received the result above. Do not call it again — answer the user's question now using what you already have."
            });
            await StreamEvent(context, new { type = "tool_result", name = toolCall.Name, result = nudge });
            prompt += $"\n<tool_result>{nudge}</tool_result>\n";
            continue;
        }

        // Execute the tool
        var toolResult = await tools.ExecuteAsync(toolCall.Name, toolCall.Arguments);

        // Stream the tool result event
        await StreamEvent(context, new
        {
            type = "tool_result",
            name = toolCall.Name,
            result = toolResult
        });

        // Inject the result into the prompt and continue generating
        prompt += $"\n<tool_result>{toolResult}</tool_result>\n";
    }

    // Signal that the agent is done
    await StreamEvent(context, new { type = "done" });
})
.WithName("AgentStream")
.WithDescription("Agent loop — chat with tool calling. The capstone: predict → generate → chat → agent");
```

#### GET `/api/tools`

A simple endpoint for the frontend to fetch available tool schemas:

```csharp
api.MapGet("/tools", (ToolRegistry tools) =>
{
    return Results.Ok(tools.GetTools());
})
.WithName("ListTools")
.WithDescription("List available tools and their schemas");
```

---

## Frontend

### Step 3: AgentView Component

#### Types

```typescript
type AgentStep =
  | { type: "text"; tokens: GeneratedToken[] }
  | { type: "tool_call"; name: string; arguments: Record<string, unknown> }
  | { type: "tool_result"; name: string; result: string };

interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, string>;
}
```

#### State

```typescript
const [systemPrompt, setSystemPrompt] = useState(
  "You are a helpful assistant. Keep your responses brief."
);
const [messages, setMessages] = useState<ChatMessage[]>([]);
const [input, setInput] = useState("");
const [isGenerating, setIsGenerating] = useState(false);
const [steps, setSteps] = useState<AgentStep[]>([]);
const [selectedStep, setSelectedStep] = useState<{
  stepIndex: number;
  tokenIndex: number;
} | null>(null);
const [temperature, setTemperature] = useState(0);
const [viewMode, setViewMode] = useState<"chat" | "raw">("chat");
const [tools, setTools] = useState<ToolDefinition[]>([]);
const [showTools, setShowTools] = useState(false);
const abortRef = useRef<AbortController | null>(null);
const inputRef = useRef<HTMLInputElement>(null);
const responseRef = useRef("");
```

#### Fetch Tools on Mount

```typescript
useEffect(() => {
  fetch("/api/tools")
    .then((r) => r.json())
    .then(setTools)
    .catch(() => {});
}, []);
```

#### Send Message

```typescript
const sendMessage = async () => {
  if (!input.trim() || isGenerating) return;

  const updatedMessages = [...messages];
  if (responseRef.current) {
    updatedMessages.push({ role: "assistant", content: responseRef.current });
  }
  updatedMessages.push({ role: "user", content: input.trim() });

  setMessages(updatedMessages);
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

        switch (event.type) {
          case "token": {
            const token: GeneratedToken = event;
            if (!token.done) {
              responseRef.current += token.token;
            }
            setSteps((prev) => {
              const last = prev[prev.length - 1];
              if (last && last.type === "text") {
                // Append to current text step
                return [
                  ...prev.slice(0, -1),
                  { ...last, tokens: [...last.tokens, token] },
                ];
              }
              // Start a new text step
              return [...prev, { type: "text", tokens: [token] }];
            });
            break;
          }
          case "tool_call":
            setSteps((prev) => [
              ...prev,
              {
                type: "tool_call",
                name: event.name,
                arguments: event.arguments,
              },
            ]);
            break;
          case "tool_result":
            setSteps((prev) => [
              ...prev,
              {
                type: "tool_result",
                name: event.name,
                result: event.result,
              },
            ]);
            break;
          case "done":
            break;
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
```

### Step 4: Step Rendering (Chat View)

Each step type gets a distinct visual block in the chat view:

#### Text Blocks

Regular assistant text, rendered as a chat bubble. Each text step is a separate generation run (before and after tool calls).

```tsx
{step.type === "text" && (
  <div className="chat-bubble chat-bubble-assistant">
    <span className="chat-role">assistant</span>
    <p>
      {step.tokens
        .filter((t) => !t.done)
        .map((t) => t.token)
        .join("")}
    </p>
  </div>
)}
```

#### Tool Call Cards

Visually distinct card showing the tool name and arguments. Appears inline in the message flow.

```tsx
{step.type === "tool_call" && (
  <div className="tool-call-card">
    <div className="tool-call-header">
      <span className="tool-call-icon">⚡</span>
      <span className="tool-call-name">{step.name}</span>
    </div>
    <pre className="tool-call-args">
      {JSON.stringify(step.arguments, null, 2)}
    </pre>
  </div>
)}
```

#### Tool Result Cards

Shows the result returned by the tool. Collapsible if the result is long (e.g., file contents).

```tsx
{step.type === "tool_result" && (
  <div className="tool-result-card">
    <div className="tool-result-header">
      <span className="tool-result-icon">✓</span>
      <span className="tool-result-name">{step.name} result</span>
    </div>
    <pre className="tool-result-content">
      {step.result.length > 500
        ? step.result.slice(0, 500) + "\n... (click to expand)"
        : step.result}
    </pre>
  </div>
)}
```

### Step 5: Tools Panel

A collapsible panel showing available tools and their schemas. Toggle via a button near the system prompt.

```tsx
<button
  className="tools-toggle"
  onClick={() => setShowTools(!showTools)}
>
  Tools ({tools.length})
</button>

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
```

### Step 6: Enhanced Raw View

The raw prompt view adds two new section types for tool interactions:

- **Tool call** — orange left border and tint, labeled "tool_call"
- **Tool result** — teal left border and tint, labeled "tool_result"

The `RawPromptView` renders tool call/result steps as additional sections within the current assistant block:

```tsx
{step.type === "tool_call" && (
  <div className="raw-section raw-section-tool-call">
    <span className="raw-section-label">tool_call</span>
    <span className="raw-special-token">{"<tool_call>"}</span>
    {"\n"}
    {JSON.stringify({ name: step.name, arguments: step.arguments }, null, 2)}
    {"\n"}
    <span className="raw-special-token">{"</tool_call>"}</span>
    {"\n"}
  </div>
)}

{step.type === "tool_result" && (
  <div className="raw-section raw-section-tool-result">
    <span className="raw-section-label">tool_result</span>
    <span className="raw-special-token">{"<tool_result>"}</span>
    {"\n"}
    {step.result}
    {"\n"}
    <span className="raw-special-token">{"</tool_result>"}</span>
    {"\n"}
  </div>
)}
```

### Step 7: Navigation

Add a fourth tab, set as default:

```typescript
const [mode, setMode] = useState<"predict" | "generate" | "chat" | "agent">("agent");
```

- **Token Prediction** — Phase 1
- **Generation Loop** — Phase 2
- **Chat** — Phase 3
- **Agent** — Phase 4

### File: `frontend/src/App.tsx`

Replace the entire contents. The file should contain:

1. Shared types (`TokenCandidate`, `PredictResponse`, `GeneratedToken`, `ChatMessage`, `AgentStep`, `ToolDefinition`), `SPECIAL_TOKENS` constant, `displayToken` helper, and `CandidateList` component
2. **Tab bar** with four tabs (default to "Agent")
3. **TokenPredictionView** — Phase 1 (unchanged)
4. **GenerationLoopView** — Phase 2 (unchanged)
5. **ChatView** — Phase 3's chat with toggle (unchanged)
6. **AgentView** — Phase 4's agent interface
7. **RawPromptView** + **RawSection** — extended with tool_call and tool_result sections

---

## Styling

### File: `frontend/src/App.css`

Extend existing styles with new classes for tool cards, tools panel, and raw prompt tool sections.

#### Tool Call Cards

- `.tool-call-card`: margin `0.5rem 0`, border `1px solid #f59e0b` (amber), border-left `4px solid #f59e0b`, border-radius `0.5rem`, background `#fffbeb`, overflow `hidden`
- `.tool-call-header`: padding `0.5rem 0.75rem`, display `flex`, align-items `center`, gap `0.5rem`, font-weight `600`, font-size `0.85rem`, color `#92400e`, background `#fef3c7`
- `.tool-call-icon`: font-size `1rem`
- `.tool-call-name`: font-family monospace
- `.tool-call-args`: margin `0`, padding `0.5rem 0.75rem`, font-size `0.8rem`, background `transparent`, white-space `pre-wrap`, color `#78350f`

#### Tool Result Cards

- `.tool-result-card`: margin `0.5rem 0`, border `1px solid #14b8a6` (teal), border-left `4px solid #14b8a6`, border-radius `0.5rem`, background `#f0fdfa`, overflow `hidden`
- `.tool-result-header`: padding `0.5rem 0.75rem`, display `flex`, align-items `center`, gap `0.5rem`, font-weight `600`, font-size `0.85rem`, color `#134e4a`, background `#ccfbf1`
- `.tool-result-icon`: font-size `1rem`
- `.tool-result-name`: font-family monospace
- `.tool-result-content`: margin `0`, padding `0.5rem 0.75rem`, font-size `0.75rem`, background `transparent`, white-space `pre-wrap`, color `#115e59`, max-height `200px`, overflow-y `auto`

#### Tools Panel

- `.tools-toggle`: same style as `.view-toggle` but with a slightly different treatment — could use a border-color of `#f59e0b` to associate with tool color
- `.tools-panel`: border `1px solid #e5e7eb`, border-radius `0.375rem`, margin-bottom `0.5rem`, overflow `hidden`
- `.tool-definition`: padding `0.5rem 0.75rem`, border-bottom `1px solid #f3f4f6`
- `.tool-definition:last-child`: no border-bottom
- `.tool-def-name`: font-family monospace, font-weight `600`, font-size `0.85rem`, color `#1f2937`
- `.tool-def-description`: font-size `0.8rem`, color `#6b7280`, margin-top `0.125rem`
- `.tool-def-params`: margin-top `0.25rem`
- `.tool-def-param`: font-size `0.75rem`, color `#4b5563`
- `.tool-def-param code`: background `#f3f4f6`, padding `0.1rem 0.3rem`, border-radius `0.2rem`, font-size `0.75rem`

#### Raw Prompt Tool Sections

- `.raw-section-tool-call`: border-left-color `#f59e0b` (amber/orange), background `rgba(245, 158, 11, 0.08)`
- `.raw-section-tool-call .raw-section-label`: color `#f59e0b`
- `.raw-section-tool-result`: border-left-color `#14b8a6` (teal), background `rgba(20, 184, 166, 0.08)`
- `.raw-section-tool-result .raw-section-label`: color `#14b8a6`

---

## Important Notes

### Debug Observability (add this from the start)

The agent loop is opaque without instrumentation — when it "hangs" or "gives the wrong answer", you need to see the generated text, the parsed tool call JSON, and the tool result to know which of three things broke: (1) the model didn't emit a tool call, (2) the parser rejected what it emitted, (3) the tool threw on the arguments. Add tracing and structured logs from the start, not after you're already debugging.

#### ActivitySource

Add a named `ActivitySource` and register it with tracing so it flows to the Aspire dashboard:

```csharp
using System.Diagnostics;

var agentActivitySource = new ActivitySource("AgentsDemystified.Server.Agent");

builder.Services.AddOpenTelemetry()
    .WithTracing(tracing => tracing.AddSource("AgentsDemystified.Server.Agent"));
```

#### What to instrument

Inside `/api/agent`:
- **`agent.loop`** (parent span): message count, max tokens, max tool calls, temperature, **system prompt**, **last user message**, **full conversation** (`[role] content` per message). Logging the user message and conversation is what makes it possible to reconstruct an entire chat session from telemetry — without it, you can see what the model said but not what was asked.
- **`agent.iteration`** (one per outer loop turn): iteration index, tokens generated, whether `</tool_call>` was hit, whether EOS was hit, the full generated text
- **`agent.tool_call` log line**: parsed tool name + raw arguments JSON (or "no tool call parsed")
- **`tool.execute`** (child span): tool name, arguments JSON, result length, result preview (first 500 chars)
- **`agent.loop.done` log line**: total tool calls executed

Concrete pattern for the loop span:

```csharp
using var loopActivity = agentActivitySource.StartActivity("agent.loop");
loopActivity?.SetTag("agent.messages.count", request.Messages.Count);
loopActivity?.SetTag("agent.max_tokens", request.MaxTokens);
loopActivity?.SetTag("agent.max_tool_calls", request.MaxToolCalls);
loopActivity?.SetTag("agent.temperature", request.Temperature);
loopActivity?.SetTag("agent.system_prompt", request.SystemPrompt);
var lastUserMessage = request.Messages.LastOrDefault(m => m.Role == "user")?.Content ?? "";
loopActivity?.SetTag("agent.user_message", lastUserMessage);
loopActivity?.SetTag("agent.conversation",
    string.Join("\n", request.Messages.Select(m => $"[{m.Role}] {m.Content}")));
```

Use `ILogger<Program>` for the flat lines so they show up in `aspire otel logs server`. Use span tags for the structured data so they show in `aspire otel traces server` and the dashboard's trace view.

#### How to inspect

```bash
# Structured logs (agent.loop.start, agent.iteration, agent.tool_call, tool.execute, agent.loop.done)
aspire otel logs server --non-interactive | grep -E "agent\.|tool\."

# Per-request trace list with span counts and durations
aspire otel traces server --non-interactive

# Unhandled exceptions (e.g. tool executor throwing on bad args)
aspire otel logs server --non-interactive | grep -iE "exception|error|fail"

# Raw SSE stream direct from the endpoint — confirms what the frontend actually receives
curl -sN -X POST http://localhost:5192/api/agent \
  -H "Content-Type: application/json" \
  -d '{"systemPrompt":"...","messages":[{"role":"user","content":"..."}],"maxTokens":200,"topN":5,"temperature":0}' \
  -o /tmp/agent-stream.log
grep -E '"type":"(tool_call|tool_result|done)"' /tmp/agent-stream.log
```

If the agent "never responds" in the UI but traces show 200 OK with 90+ spans, the request is working — the UI is either waiting for slow generation or has a rendering bug. If you see a `tool_call` event with no matching `tool_result` event, the tool executor threw and killed the request mid-stream.

### Tool Result Injection Format

Do **not** inject the tool result inline inside the same assistant turn:

```csharp
// WRONG — model emits EOS immediately on the next iteration because the
// assistant turn's "answer" (the tool_call) has already ended.
prompt += $"\n<tool_result>{toolResult}</tool_result>\n";
```

Instead, close the assistant turn, open a new user turn carrying the tool result, then reopen the assistant turn — and **neutralize control tokens in the result content first** (see below):

```csharp
prompt +=
    $"{ChatTemplateBuilder.EndToken}\n" +
    $"{ChatTemplateBuilder.UserToken}\n<tool_result>{NeutralizeControlTokens(toolResult)}</tool_result>\n{ChatTemplateBuilder.EndToken}\n" +
    $"{ChatTemplateBuilder.AssistantToken}\n";
```

This matches how chat-tuned models are trained: every assistant turn is bracketed by `<|end|>` and the next turn starts with `<|user|>` or similar. Without this, phi4-mini treats the `<tool_call>` as the completed answer and picks EOS on the next generation step.

#### Neutralize control tokens in tool-result content

A tool result can itself contain the framing tokens — a read file may include `<|end|>` / `<|assistant|>` / `<|user|>` (e.g. this project's chat-framing spec) or `<tool_call>…` tags (this very spec). Injected raw, the tokenizer treats those embedded tokens as **real turn boundaries or tool calls**: the conversation structure corrupts and the model derails — re-issuing tool calls, parroting the tags, never answering, and (because the flail bloats the prompt) eventually timing out the Ollama call and killing the stream.

Defuse them before injection. Rewrite each control token to a look-alike that reads the same to a human but is inert to the tokenizer. Sanitize only the **prompt copy** — stream the verbatim result to the UI so the audience still sees true file contents:

```csharp
static string NeutralizeControlTokens(string content) => content
    .Replace("<|system|>", "‹system›")
    .Replace("<|user|>", "‹user›")
    .Replace("<|assistant|>", "‹assistant›")
    .Replace("<|end|>", "‹end›")
    .Replace("<tool_call>", "‹tool_call›")
    .Replace("</tool_call>", "‹/tool_call›")
    .Replace("<tool_result>", "‹tool_result›")
    .Replace("</tool_result>", "‹/tool_result›");
```

Apply it at every injection site (the normal tool result and the duplicate-guardrail nudge). This is also a mild prompt-injection defense: tool output can no longer forge framing tokens. It is what lets the agent read `phase-3-spec.md` (delimiter-heavy) and `phase-4-spec.md` (tool-call-heavy) and still produce a clean summary.

### Tool Executor Error Handling

The model will emit malformed arguments — empty objects (`"arguments": {}`), missing required parameters, the wrong shape. If `ExecuteAsync` throws, the SSE stream dies mid-flight (the `tool_call` event goes out but `tool_result` never does), and you'll see `An unhandled exception has occurred while executing the request` in the server logs with no stack trace context.

Guard every argument access and wrap execution in try/catch so the agent loop can continue even when the model emits bad JSON:

```csharp
public async Task<string> ExecuteAsync(string name, JsonElement arguments)
{
    try
    {
        return name switch
        {
            "search_files" => await SearchFiles(GetStringArg(arguments, "query")),
            "read_file"    => await ReadFile(GetStringArg(arguments, "path")),
            _              => JsonSerializer.Serialize(new { error = $"Unknown tool: {name}" })
        };
    }
    catch (Exception ex)
    {
        return JsonSerializer.Serialize(new { error = ex.Message });
    }
}

private static string GetStringArg(JsonElement arguments, string name)
{
    if (arguments.ValueKind != JsonValueKind.Object || !arguments.TryGetProperty(name, out var prop))
        throw new ArgumentException($"Missing required argument '{name}'");
    return prop.ValueKind == JsonValueKind.String ? prop.GetString() ?? "" : prop.GetRawText().Trim('"');
}
```

The error JSON flows back to the model as a `tool_result`, and the model often recovers by calling the tool again with corrected arguments — which is exactly the behavior you want to demo.

### Tool prompt vs system prompt — a deliberate teaching split

`GetToolPromptSection()` is intentionally MECHANICAL ONLY: it teaches the model the `<tool_call>` syntax and how results come back, nothing else. Everything about *when* and *whether* to use the tools — "you can read files," "search returns paths so chain to read," "stop once you can answer" — is **behavior steering** and belongs in the user-editable System Prompt, which is blank/generic by default ("You are a helpful assistant. Keep your responses brief.").

This split is the phase's best live beat:

1. **Fail first.** With the generic system prompt, ask *"Summarize the contents of the phase-4-spec file."* The model has the tools (visible in the raw view and the Tools panel) but no guidance, so it flails — typically it guesses `read_file` on a bad path (`"phase-4-spec"`, no directory, no extension), gets `File not found`, and gives up. The audience sees that **a tool is just a capability; the model still needs to be told how to behave.**
2. **Fix live.** Paste behavior steering into the System Prompt — capability ("you CAN read files, never claim you can't"), the search→read chain, and a worked example — then re-ask and watch it improve. This demonstrates that the system prompt, not the tool wiring, is what steers the agent.

**Why the steering must be the strong version.** Testing showed a one-line system prompt is not enough for phi4-mini at temperature 0 — it flails on query formulation and often never reaches the read step. The steering that reliably drives search→read→answer includes a **worked multi-step example** (search → tool_result → read_file → tool_result → answer). Keep that example in the pasted prompt; a terse "use your tools to read files" regresses.

It must also **draw the search/read boundary explicitly.** Because `search_files` matches file *content* (not just names — see "Forgiving search" below), the model otherwise treats search as a content/read tool: it issues `search_files` with full file paths as the query, gets back a path list (never contents), and loops searching instead of reading. The cure is one sharp line in the system prompt: *"search_files returns ONLY a list of paths, never file contents — to see inside a file you MUST call read_file; never search for a full path."* A recommended steering block that reliably produces a clean search→read→answer:

```
You are a helpful assistant with tools to explore this project. You CAN read files — never claim you can't.

IMPORTANT: search_files returns ONLY a list of file paths, never file contents. To see what is inside a file you MUST call read_file with its path. Never call search_files with a full file path — once you have a path, call read_file on it.

To answer a question about the code:
1. Call search_files with a short concrete term to locate the file.
2. Call read_file on a path from the results to get its contents.
3. Answer in plain text from those contents. Stop calling tools once you can answer.
```

### Instructive failure modes (keep them — don't engineer them away)

This phase is more honest, and more memorable, *because* a small local model on dumb tools fails in visible ways. Leave these rough edges in:

- **Refusal.** phi4-mini has a base-training prior to disclaim file access (*"I am unable to read files directly..."*). It calls `search_files` (matches "I can look things up") but balks at `read_file`. The system-prompt capability line is what overrides this — show the before/after.
- **Forgiving search (chosen on purpose).** `search_files` tokenizes the query and matches any file whose path or content contains *any* term, ranked by term-match count. This is deliberately broad so the model's sloppy natural-language queries (e.g. `"ChatTemplateBuilder special tokens"`) still land a hit and the chain proceeds to `read_file` — rather than the model getting `{"files":[]}` from an exact-substring match and flailing. Two tradeoffs we accept: (a) broad matching returns more false positives, so the model has to pick the right path from the result list; (b) because search matches *content*, the model is tempted to use it as a read tool — issuing `search_files` with full file paths and looping on path lists instead of calling `read_file`. We do **not** narrow the tool to fix (b); instead the system-prompt steering draws the boundary explicitly ("search returns only paths — you MUST read_file for contents"). Keeping search broad but steering the behavior was the deliberate call.
- **Flailing / repeated calls.** At temperature 0 the model often re-issues the same `read_file` or `search_files` instead of answering. Left unchecked this re-injects the full file contents each turn, bloating the prompt until a single Ollama `/api/generate` call exceeds the resilience-handler timeout and the SSE stream dies with no `done` event — the UI just hangs. The **duplicate tool-call guardrail** (below) is the one rough edge we DO smooth, because a hung stream isn't instructive, it's just broken.
- **Control tokens in read files (fixed via neutralization).** The two big spec docs are adversarial input: `phase-3-spec.md` is full of chat-template delimiters (`<|end|>`, `<|assistant|>`, `<|user|>`) and `phase-4-spec.md` is full of `<tool_call>{...}</tool_call>` tags. Read raw into the prompt, those embedded tokens collide with the framing — the model breaks into fake turns or parrots the tags, flails to the `MaxToolCalls` cap, and the bloated prompt eventually times out the Ollama call and kills the stream. This is the one class we **fixed** (see *Neutralize control tokens in tool-result content* above): the injected copy has its control tokens defused, so reading either spec now yields a clean summary. Worth showing on stage as the payoff — "the agent can even read its own spec." (Earlier iterations demoed around this by reading only normal source files; with neutralization that workaround is no longer required.)

Use telemetry to name the failure on stage: a prose refusal shows as `agent.iteration` with `tool_call_parsed=false`; flailing shows as repeated `agent.tool_call` lines with `duplicate=True`; a brittle-search miss shows as `tool.execute` with `result_preview={"files":[]}`. (Before neutralization, reading a delimiter-heavy doc showed up as `agent.iteration` text fragmenting into stray `<|end|>`/`<tool_call>` tokens — if you ever see that again, a tool result is reaching the prompt unsanitized.)

### Duplicate tool-call guardrail

The one failure mode worth smoothing: repeated identical tool calls bloating the prompt until Ollama times out and kills the stream. Track executed calls in a `HashSet<string>` keyed by `name + arguments`; on a repeat, skip `ExecuteAsync`, stream back a short nudge as the `tool_result` instead of the full content, and `continue`. The duplicate still counts toward `MaxToolCalls`, so the loop is guaranteed to terminate and emit `done`.

```csharp
var executedCalls = new HashSet<string>();
// ...inside the loop, after streaming the tool_call event:
var callKey = $"{toolCall.Name}:{toolCall.Arguments.GetRawText()}";
if (!executedCalls.Add(callKey))
{
    var nudge = JsonSerializer.Serialize(new
    {
        note = $"You already called {toolCall.Name} with these arguments and received the result above. Do not call it again — answer the user's question now using what you already have."
    });
    await StreamEvent(context, new { type = "tool_result", name = toolCall.Name, result = nudge });
    prompt += /* inject nudge as a new user turn, see Tool Result Injection Format */;
    continue;
}
```

This keeps context bounded so the stream always completes — the model may still flail to the `MaxToolCalls` cap, but the demo degrades gracefully instead of hanging.

### Tool Call Format Experimentation

The exact tool call format (`<tool_call>...</tool_call>`) and system prompt instructions may need experimentation with phi4-mini. The model needs to reliably:

1. Generate well-formed JSON inside the tool call tags
2. Stop generating after the closing `</tool_call>` tag (or at least include it)
3. Continue coherently after receiving a tool result

If phi4-mini struggles with this format, alternatives to try:
- Simpler JSON without tags: `{"tool": "name", "args": {...}}`
- Function-call syntax: `search_files("query")`
- More explicit system prompt instructions with few-shot examples

The spec provides the `<tool_call>` tag format as the starting point because it's easy to parse and visually distinct.

### Project Root Path

The `ToolRegistry` is initialized with `builder.Environment.ContentRootPath` which points to the `AgentsDemystified.Server` directory. The tools search and read files relative to this path. This means the tools see the server project's files. To expose the full repo (including frontend, docs, etc.), you may need to pass the repo root instead:

```csharp
builder.Services.AddSingleton(new ToolRegistry(
    Path.GetFullPath(Path.Combine(builder.Environment.ContentRootPath, ".."))));
```

---

## Verification

1. Run `dotnet build` — must succeed with zero errors
2. Run `cd frontend && npx tsc --noEmit` — no type errors
3. Open the frontend — tab bar shows "Token Prediction", "Generation Loop", "Chat", and "Agent"
4. All previous tabs (Phase 1, 2, 3) still work
5. Switch to "Agent" tab — looks like the Chat tab but with a "Tools" button
6. Click "Tools" — panel expands showing `search_files` and `read_file` with descriptions and parameters
7. Type "What files are in this project?" and send
8. The model should generate text, then a `<tool_call>` for `search_files`
9. A tool call card appears (orange border) showing the tool name and arguments
10. A tool result card appears (teal border) showing the matching files
11. The model continues generating text using the tool result to answer the question
12. The full interaction should complete within the max tool call cap
13. Click "Show Raw" — the raw prompt view shows all sections: system (blue, including tool definitions), user (green), assistant (purple), tool_call (orange), tool_result (teal)
14. Special tokens and tool tags are visible in the raw view
15. Click generated tokens in the raw view — candidate list appears, shows alternatives
16. Try "How does the predict endpoint work?" — should trigger search_files → read_file → answer (multi-step)
17. Click "Stop" mid-generation — agent loop stops cleanly
18. Click "Clear" — conversation resets
19. Temperature and system prompt controls work as before
20. Verify `/api/agent` and `/api/tools` appear in Scalar API docs at `/scalar/v1`
21. Verify `read_file` rejects paths outside the project directory (e.g., `../../etc/passwd` returns an error)
