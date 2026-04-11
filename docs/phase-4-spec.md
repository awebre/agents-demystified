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
        section += """
            When you need to use a tool, output EXACTLY this format:
            <tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>

            Wait for the tool result before continuing your response. You may call multiple tools in sequence to answer a question.
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
        var results = new List<string>();
        var searchDir = new DirectoryInfo(projectRoot);

        foreach (var file in searchDir.EnumerateFiles("*", SearchOption.AllDirectories))
        {
            // Skip build artifacts, hidden directories, node_modules
            var relativePath = Path.GetRelativePath(projectRoot, file.FullName);
            if (relativePath.Contains("/bin/") || relativePath.Contains("/obj/") ||
                relativePath.Contains("/node_modules/") || relativePath.StartsWith("."))
                continue;

            // Match against file name
            if (file.Name.Contains(query, StringComparison.OrdinalIgnoreCase))
            {
                results.Add(relativePath);
                continue;
            }

            // Match against file contents (small files only)
            if (file.Length < 100_000)
            {
                try
                {
                    var content = File.ReadAllText(file.FullName);
                    if (content.Contains(query, StringComparison.OrdinalIgnoreCase))
                        results.Add(relativePath);
                }
                catch { /* skip unreadable files */ }
            }
        }

        return Task.FromResult(JsonSerializer.Serialize(new { files = results.Take(10) }));
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
    int MaxTokens = 200,
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
