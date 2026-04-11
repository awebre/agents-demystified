# Phase 2: Autoregressive Generation Loop

## Goal

Add a streaming `POST /api/generate` endpoint and a generation UI that shows text building up token-by-token. The server runs a prediction loop — repeatedly predicting the next token and selecting the top result — until a stop token (like `<|endoftext|>`) is detected. Each step's token and candidates are streamed to the client, including the final EOS token so the audience can see exactly why generation stopped.

The key conceptual point: **generation is just Phase 1 (single token prediction) run in a loop.** The spec reflects this by first refactoring Phase 1's prediction logic into a server-side abstraction, then building Phase 2 as a loop over that same abstraction.

### Important: Ollama EOS Token Behavior

Ollama suppresses special tokens (e.g., `<|endoftext|>`) in the `Response` field, returning an empty string. However, the actual token **is** present in the logprobs. The `TokenPredictor` must detect this case and recover the real token from the top logprob candidate. Stop detection (`Done`) is based on whether the token is a special token (matches `<...>` pattern) or empty — **not** on Ollama's `Done` flag, which is always `true` for `NumPredict = 1` requests.

**Prerequisite:** Phase 1 is already implemented. The `/api/predict` endpoint and token prediction UI exist.

---

## Step 1: Refactor Phase 1 into a Server-Side Abstraction

Extract the core "predict one token" operation into a service class that both endpoints share. The abstraction does the real work — calling Ollama, extracting logprobs, ranking candidates. Endpoints become thin wrappers.

### File: `AgentsDemystified.Server/TokenPredictor.cs` (new file)

```csharp
using OllamaSharp;
using OllamaSharp.Models;

namespace AgentsDemystified.Server;

public record TokenPredictionResult(
    string Token,
    List<TokenCandidate> Candidates,
    bool Done);

public class TokenPredictor(IOllamaApiClient client)
{
    /// <summary>
    /// Predict the single next token for a prompt. Returns the chosen token
    /// and the top N alternative candidates with probabilities.
    /// This is the fundamental operation — everything else is built on this.
    /// </summary>
    public async Task<TokenPredictionResult> PredictNextAsync(
        string prompt, int topN = 10, CancellationToken cancellationToken = default)
    {
        var request = new GenerateRequest
        {
            Model = "phi3",
            Prompt = prompt,
            Raw = true,
            Stream = false,
            Logprobs = true,
            TopLogprobs = topN,
            Options = new RequestOptions { NumPredict = 1 }
        };

        GenerateResponseStream? lastResponse = null;
        await foreach (var response in client.GenerateAsync(request, cancellationToken))
        {
            if (response is not null)
                lastResponse = response;
        }

        var candidates = ExtractCandidates(lastResponse);
        var token = lastResponse?.Response ?? "";

        // IMPORTANT: Ollama suppresses special tokens (like <|endoftext|>) in the
        // Response field, returning "". When that happens, grab the actual token
        // from the top logprob candidate so the frontend can display what the model
        // really predicted.
        if (string.IsNullOrEmpty(token) && candidates.Count > 0)
            token = candidates[0].Token;

        var isEos = string.IsNullOrEmpty(token.Trim()) || IsSpecialToken(token.Trim());
        return new TokenPredictionResult(
            token,
            candidates,
            isEos);
    }

    private static bool IsSpecialToken(string token) =>
        token.StartsWith('<') && token.EndsWith('>');

    private static List<TokenCandidate> ExtractCandidates(GenerateResponseStream? response)
    {
        var firstLogprob = response?.Logprobs?.FirstOrDefault();
        return firstLogprob?.TopLogprobs?
            .Where(lp => lp.Token is not null)
            .Select(lp => new TokenCandidate(
                lp.Token!,
                lp.LogProbability ?? 0,
                Math.Exp(lp.LogProbability ?? double.NegativeInfinity)))
            .OrderByDescending(c => c.Probability)
            .ToList() ?? [];
    }
}
```

### File: `AgentsDemystified.Server/Program.cs`

#### Register the service

Add after the existing `builder.AddOllamaApiClient("phi3")` line:

```csharp
builder.Services.AddTransient<TokenPredictor>();
```

#### Required Usings

Add at the top:

```csharp
using AgentsDemystified.Server;
```

#### Update `/api/predict` to use TokenPredictor

Replace the existing `/api/predict` endpoint with:

```csharp
api.MapPost("/predict", async (PredictRequest request, TokenPredictor predictor) =>
{
    var result = await predictor.PredictNextAsync(request.Prompt, request.TopN);
    return Results.Ok(new PredictResponse(result.Token, result.Candidates));
})
.WithName("PredictNextToken")
.WithDescription("Predict the next token for a given prompt and return top N candidates with probabilities");
```

The endpoint is now a one-liner over the shared abstraction. The inline logprob extraction logic is gone.

### Frontend Refactor

#### File: `frontend/src/App.tsx`

Extract the candidate bar chart into a reusable `CandidateList` component defined in the same file. Phase 1's prediction view and Phase 2's step inspector will both use it.

```typescript
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
```

Phase 1's prediction view should use `CandidateList` with `onSelect` wired to append the token (the click-to-insert behavior). The `displayToken` helper and `TokenCandidate` interface from Phase 1 remain as-is.

---

## Step 2: Add Phase 2 Backend

### File: `AgentsDemystified.Server/Program.cs`

Add the following endpoint after `/api/predict`. It streams each step of the generation loop as a Server-Sent Event.

#### New Request Record

```csharp
record GenerateStreamRequest(string Prompt, int MaxTokens = 200, int TopN = 5);
```

#### Endpoint (Server-Sent Events)

The loop is right here in the endpoint — the audience can see it's just `PredictNextAsync` called repeatedly, appending each token to the prompt.

```csharp
api.MapPost("/generate", async (GenerateStreamRequest request, TokenPredictor predictor, HttpContext context) =>
{
    context.Response.ContentType = "text/event-stream";
    context.Response.Headers.CacheControl = "no-cache";
    context.Response.Headers.Connection = "keep-alive";

    var currentPrompt = request.Prompt;

    for (var i = 0; i < request.MaxTokens; i++)
    {
        context.RequestAborted.ThrowIfCancellationRequested();

        // Same prediction call as Phase 1 — just in a loop now
        var result = await predictor.PredictNextAsync(
            currentPrompt, request.TopN, context.RequestAborted);

        var payload = JsonSerializer.Serialize(new
        {
            token = result.Token,
            done = result.Done,
            candidates = result.Candidates
        });

        await context.Response.WriteAsync($"data: {payload}\n\n");
        await context.Response.Body.FlushAsync();

        if (result.Done || string.IsNullOrEmpty(result.Token))
            break;

        // Append the predicted token and predict the next one
        currentPrompt += result.Token;
    }
})
.WithName("GenerateStream")
.WithDescription("Run token prediction in a loop, streaming each step — Phase 1 in a loop");
```

#### Required Usings

Add at the top of Program.cs (if not already present):

```csharp
using System.Text.Json;
```

---

## Step 3: Add Phase 2 Frontend

### Navigation

Add a simple tab bar at the top of the app to switch between modes.

- Add a `mode` state: `"predict" | "generate"`
- Render a tab bar with two tabs: **"Token Prediction"** and **"Generation Loop"**
- Conditionally render the appropriate view based on `mode`
- Both views use the shared `CandidateList` component

### File: `frontend/src/App.tsx`

Replace the entire contents. The file should contain:

1. Shared types (`TokenCandidate`, `PredictResponse`, `GeneratedToken`), `displayToken` helper, and `CandidateList` component
2. **Tab bar** at the top with two tabs
3. **TokenPredictionView** — Phase 1's autocomplete UI, using `CandidateList` with `onSelect` for click-to-insert
4. **GenerationLoopView** — Phase 2's streaming UI, using `CandidateList` for the step inspector

### GenerationLoopView

#### State

```typescript
const [genPrompt, setGenPrompt] = useState("Once upon a time");
const [generatedTokens, setGeneratedTokens] = useState<GeneratedToken[]>([]);
const [isGenerating, setIsGenerating] = useState(false);
const [selectedStep, setSelectedStep] = useState<number | null>(null);
const abortRef = useRef<AbortController | null>(null);

interface GeneratedToken {
  token: string;
  done: boolean;
  candidates: TokenCandidate[];
}
```

#### Start Generation

```typescript
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
      body: JSON.stringify({ prompt: genPrompt, maxTokens: 200, topN: 5 }),
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
```

#### Stop Generation

```typescript
const stopGeneration = () => {
  abortRef.current?.abort();
};
```

#### UI Layout

```
GenerationLoopView
├── PromptInput (textarea, ~3 rows)
├── ButtonRow ("Generate" button + "Stop" button)
├── GeneratedText (the prompt + generated tokens displayed inline, each token is a clickable span)
└── StepCandidates (when a token span is clicked, show its top candidates via CandidateList)
```

**GeneratedText display:**
- Show the original prompt in a muted/gray color
- Show each generated token as a `<span>` appended after the prompt
- Each token span is clickable — clicking it sets `selectedStep` to that token's index
- The currently selected token span gets a highlighted background
- Render raw tokens in the text flow (don't replace spaces with ␣ in the flowing text — only use the display symbols in the CandidateList)
- **EOS token:** When `event.done` is true, render the token text (e.g., `<|endoftext|>`) with a distinct `.eos-token` style — red text, light red background, small pill badge. If the token is somehow empty, show `⏹` as a fallback. This is clickable like any other token to inspect its candidates.

**Token span rendering (exact JSX):**

```tsx
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
    {gt.done ? gt.token || "⏹" : gt.token}
  </span>
))}
```

**StepCandidates panel:**
- Shown below the generated text when a token is selected (`selectedStep !== null`)
- Uses the shared `CandidateList` component with the selected step's candidates
- Pass `highlightToken` set to the actual chosen token for that step, so the audience can see which candidate was selected

**Button states:**
- "Generate" is disabled while generating
- "Stop" is only visible/enabled while generating

#### Styling

Add styles for the new elements to `App.css`:
- Tab bar: horizontal flex, each tab is a clickable element, active tab has a bottom border or background highlight
- Generated text: displayed in a monospace block with `white-space: pre-wrap`
- Token spans: inline, with a subtle hover effect and a highlighted state when selected
- `.eos-token`: red text (`#ef4444`), light red background (`#fef2f2`), red border (`#fecaca`), smaller font size (`0.75rem`), bold, small pill shape with padding and border-radius. Hover darkens the background.
- Step candidates panel: reuses the same `.candidate-list` styles from the shared `CandidateList`

### File: `frontend/src/App.css`

Extend (do not replace) the existing styles with new classes for the tab bar, generation view, token spans, and step candidates panel.

---

## Verification

1. Run `dotnet build` — must succeed with zero errors
2. Run `cd frontend && npx tsc --noEmit` — no type errors
3. Open the frontend — tab bar should show "Token Prediction" and "Generation Loop"
4. "Token Prediction" tab works exactly as before (Phase 1 — autocomplete with click-to-insert)
5. Switch to "Generation Loop" tab
6. Default prompt "Once upon a time" is pre-filled
7. Click "Generate" — tokens stream in one by one, building up text
8. Click "Stop" mid-generation — streaming stops
9. Click on any generated token — the step's top candidates appear below using the same CandidateList as Phase 1
10. Verify `/api/generate` appears in Scalar API docs at `/scalar/v1`
