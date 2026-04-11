# Phase 2: Autoregressive Generation Loop

## Goal

Add a streaming `POST /api/generate` endpoint and a generation UI that shows text building up token-by-token. The server runs a prediction loop — repeatedly predicting the next token and selecting the top result — until a stop token (like `<|endoftext|>`) is detected. Each step's token and candidates are streamed to the client, including the final EOS token so the audience can see exactly why generation stopped.

The key conceptual point: **generation is just Phase 1 (single token prediction) run in a loop.** The spec reflects this by first refactoring Phase 1's prediction logic into a server-side abstraction, then building Phase 2 as a loop over that same abstraction.

### Important: Token Selection and EOS Detection

We do **not** use Ollama's `Response` field to determine the chosen token. Instead, we always pick the token ourselves from the logprob candidates. This gives us control over sampling (greedy vs temperature-based) and avoids several Ollama quirks:

- Ollama's `Done` flag is always `true` for `NumPredict = 1` requests — useless for stop detection.
- Ollama suppresses special tokens in the `Response` field, returning an empty string for EOS.
- Ollama's built-in temperature sampling doesn't always match the logprob distribution.

**Stop detection** is based on whether the chosen token matches the `<|...|>` pattern (e.g., `<|endoftext|>`). When Ollama returns no logprobs at all (empty candidates), synthesize a `<|endoftext|>` entry so the frontend always shows why generation stopped.

**SSE serialization**: The `/api/generate` endpoint manually serializes JSON via `JsonSerializer.Serialize(...)`. Unlike `Results.Ok()` (which auto-camelCases), manual serialization preserves PascalCase by default. You **must** pass `JsonSerializerOptions.Web` to get camelCase matching the frontend TypeScript interfaces.

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
        string prompt, int topN = 10, double temperature = 0,
        CancellationToken cancellationToken = default)
    {
        var request = new GenerateRequest
        {
            Model = "phi4-mini",
            Prompt = prompt,
            Raw = true,
            Stream = false,
            Logprobs = true,
            TopLogprobs = topN,
            Options = new RequestOptions { NumPredict = 1 }
        };

        GenerateResponseStream? lastResponse = null;
        await foreach (var response in client.GenerateAsync(request, cancellationToken))
            if (response is not null)
                lastResponse = response;

        var candidates = ExtractCandidates(lastResponse);

        // When Ollama returns no logprobs (empty candidates), the model is signaling EOS.
        // Synthesize an <|endoftext|> entry so the frontend always shows why generation stopped.
        if (candidates.Count == 0)
            candidates = [new TokenCandidate("<|endoftext|>", 0, 1.0)];

        // We pick the token ourselves from the candidate list.
        // Temperature=0 → greedy (always top candidate).
        // Temperature>0 → re-weight logprobs and sample from the distribution.
        var token = temperature > 0
            ? SampleWithTemperature(candidates, temperature)
            : candidates[0].Token;

        var isEos = IsSpecialToken(token.Trim());
        return new TokenPredictionResult(
            token,
            candidates,
            isEos);
    }

    private static string SampleWithTemperature(List<TokenCandidate> candidates, double temperature)
    {
        // Divide logprobs by temperature, then softmax to get adjusted probabilities
        var scaled = candidates.Select(c => c.LogProbability / temperature).ToList();
        var maxScaled = scaled.Max();
        var exps = scaled.Select(s => Math.Exp(s - maxScaled)).ToList(); // subtract max for numerical stability
        var sum = exps.Sum();
        var probs = exps.Select(e => e / sum).ToList();

        // Sample from the distribution
        var roll = Random.Shared.NextDouble();
        var cumulative = 0.0;
        for (var i = 0; i < probs.Count; i++)
        {
            cumulative += probs[i];
            if (roll <= cumulative)
                return candidates[i].Token;
        }

        return candidates[^1].Token;
    }

    private static bool IsSpecialToken(string token) =>
        token.StartsWith("<|") && token.EndsWith("|>");

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

Add after the existing `builder.AddOllamaApiClient("phi4-mini")` line (already in the baseline):

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
record GenerateStreamRequest(string Prompt, int MaxTokens = 200, int TopN = 5, double Temperature = 0);
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
            currentPrompt, request.TopN, request.Temperature, context.RequestAborted);

        // IMPORTANT: Must use JsonSerializerOptions.Web for camelCase output.
        // Manual JsonSerializer.Serialize() preserves PascalCase by default,
        // unlike Results.Ok() which auto-camelCases. Without this, the frontend
        // gets "Token"/"Probability" instead of "token"/"probability".
        var payload = JsonSerializer.Serialize(new
        {
            token = result.Token,
            done = result.Done,
            candidates = result.Candidates
        }, JsonSerializerOptions.Web);

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
const [temperature, setTemperature] = useState(0);
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
├── TemperatureControl (slider 0–2, step 0.1, with label showing value and hint text)
├── ButtonRow ("Generate" button + "Stop" button)
├── GeneratedText (the prompt + generated tokens displayed inline, each token is a clickable span)
└── StepCandidates (when a token span is clicked, show its top candidates via CandidateList)
```

**Temperature slider:**
- Range 0–2, step 0.1, default 0
- Label shows current value and a hint: 0 = "(greedy)", ≤0.5 = "(focused)", ≤1 = "(balanced)", >1 = "(creative)"
- Disabled while generating

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
- Temperature control: label with flex layout, monospace value, muted hint text, full-width range slider with blue accent color, disabled state at 50% opacity
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
6. Default prompt "Once upon a time" is pre-filled, temperature slider at 0.0 (greedy)
7. Click "Generate" — tokens stream in one by one, building up text
8. At temperature 0, running the same prompt twice produces identical output (greedy/deterministic)
9. Click "Stop" mid-generation — streaming stops
10. Click on any generated token — the step's top candidates appear below, with the chosen token highlighted
11. The highlighted token should always be the #1 candidate when temperature is 0
12. Generation ends with a visible `<|endoftext|>` red badge (not an empty/invisible stop)
13. Slide temperature to 1.0, regenerate — output should vary between runs
14. Verify `/api/generate` appears in Scalar API docs at `/scalar/v1`
