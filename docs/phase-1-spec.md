# Phase 1: Next Token Prediction

## Goal

Add a `POST /api/predict` endpoint to the backend and a token prediction UI to the frontend. Given a text prompt, the endpoint returns the model's top N predicted next tokens with their probabilities.

---

## Backend

### File: `AgentsDemystified.Server/Program.cs`

Add the following after the existing `var api = app.MapGroup("/api");` line.

#### Request/Response Records

```csharp
record PredictRequest(string Prompt, int TopN = 10);

record TokenCandidate(string Token, double LogProbability, double Probability);

record PredictResponse(string PredictedToken, List<TokenCandidate> Candidates);
```

#### Endpoint

```csharp
api.MapPost("/predict", async (PredictRequest request, IOllamaApiClient client) =>
{
    var generateRequest = new GenerateRequest
    {
        Model = "phi4-mini",
        Prompt = request.Prompt,
        Raw = true,
        Stream = false,
        Logprobs = true,
        TopLogprobs = request.TopN,
        Options = new RequestOptions { NumPredict = 1 }
    };

    GenerateResponseStream? lastResponse = null;
    await foreach (var response in client.GenerateAsync(generateRequest))
    {
        if (response is not null)
            lastResponse = response;
    }

    var firstLogprob = lastResponse?.Logprobs?.FirstOrDefault();
    var candidates = firstLogprob?.TopLogprobs?
        .Where(lp => lp.Token is not null)
        .Select(lp => new TokenCandidate(
            lp.Token!,
            lp.LogProbability ?? 0,
            Math.Exp(lp.LogProbability ?? double.NegativeInfinity)))
        .OrderByDescending(c => c.Probability)
        .ToList() ?? [];

    return Results.Ok(new PredictResponse(
        lastResponse?.Response?.Trim() ?? "",
        candidates));
})
.WithName("PredictNextToken")
.WithDescription("Predict the next token for a given prompt and return top N candidates with probabilities");
```

#### Required Usings

Add at the top of Program.cs:

```csharp
using OllamaSharp;
using OllamaSharp.Models;
```

`IOllamaApiClient` is already registered via `builder.AddOllamaApiClient("phi4-mini")`. No additional DI setup needed.

### Key OllamaSharp Types (reference)

All in namespace `OllamaSharp.Models`:

- `GenerateRequest` — `Model` (string), `Prompt` (string), `Raw` (bool?), `Stream` (bool, default true), `Logprobs` (bool?), `TopLogprobs` (int?), `Options` (RequestOptions?)
- `RequestOptions` — `NumPredict` (int?), `Temperature` (float?), etc.
- `GenerateResponseStream` — `Response` (string), `Done` (bool), `Logprobs` (IEnumerable&lt;Logprob&gt;?)
- `Logprob` — `Token` (string?), `LogProbability` (double?), `TopLogprobs` (IEnumerable&lt;Logprob&gt;?)

`IOllamaApiClient.GenerateAsync(GenerateRequest)` returns `IAsyncEnumerable<GenerateResponseStream?>`.

---

## Frontend

### File: `frontend/src/App.tsx`

Replace the entire contents of App.tsx with the token prediction UI.

#### Component Structure

```
App
├── Header ("Agents Demystified — Next Token Prediction")
├── PromptInput (textarea, no button — predictions fire automatically)
└── CandidateList (ranked bar chart of top N candidates, shown below textarea like an autocomplete dropdown)
```

#### Behavior

- **No button.** Predictions fire automatically as the user types.
- Debounce the API call by 300ms so it doesn't fire on every keystroke.
- While a prediction is loading, show a subtle loading indicator (e.g. a small spinner or "predicting..." text near the candidate list, NOT a disabled button).
- If the prompt is empty or whitespace-only, clear the results and don't call the API.
- Use `AbortController` to cancel in-flight requests when the user types again before the previous request completes.

#### State

```typescript
interface TokenCandidate {
  token: string;
  logProbability: number;
  probability: number;
}

interface PredictResponse {
  predictedToken: string;
  candidates: TokenCandidate[];
}

// Component state:
const [prompt, setPrompt] = useState("The capital of France is");
const [result, setResult] = useState<PredictResponse | null>(null);
const [loading, setLoading] = useState(false);
```

#### Debounced API Call

Use a `useEffect` that watches `prompt` and debounces the fetch by 300ms. Cancel previous in-flight requests with `AbortController`.

```typescript
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
```

#### Results Display

Show the candidate list directly below the textarea, like an autocomplete suggestion list.

For each candidate in `result.candidates`:

- Show the token text, but **replace invisible whitespace characters with visible symbols** so they're not blank:
  - Space `" "` → `"␣"`
  - Newline `"\n"` → `"↵"`
  - Tab `"\t"` → `"⇥"`
  - Carriage return `"\r"` → `"⏎"`
  - If the token is entirely whitespace but doesn't match the above, show it as a JS-style escape (e.g. `"\\n"`)
  - Use a helper function like:
    ```typescript
    function displayToken(token: string): string {
      return token
        .replace(/ /g, "␣")
        .replace(/\n/g, "↵")
        .replace(/\t/g, "⇥")
        .replace(/\r/g, "⏎");
    }
    ```
  - Render the display token text inside the candidate row, NOT the raw token
- Show probability as a percentage: `(candidate.probability * 100).toFixed(1) + "%"`
- Render a horizontal bar with width proportional to probability relative to the top candidate: `width: (candidate.probability / result.candidates[0].probability * 100) + "%"`
- Highlight the first candidate (the predicted token) with a distinct background color

When `loading` is true and there are no results yet, show "Predicting..." below the textarea.

#### Clicking a Candidate

When the user clicks a candidate row, append the **raw token** (not the display-escaped version) to the current prompt. This simulates selecting an autocomplete suggestion. After appending, the debounced prediction will automatically fire again for the new prompt, creating a fluid "keep picking the next token" experience.

#### Styling

Use inline styles or a simple CSS approach. The UI should be minimal and readable:

- Dark or light background, monospace font for tokens
- Bars in a color like `#3b82f6` (blue)
- Candidate list positioned directly below the textarea like an autocomplete dropdown
- Prompt textarea should be wide (100%) and ~3 rows tall

### File: `frontend/src/App.css`

Replace with minimal styles for the prediction UI. Keep it simple — the focus is on the data, not polish.

---

## Verification

1. Run `dotnet build` — must succeed with zero errors
2. Run `aspire start` — all resources (ollama, phi4-mini, server, webfrontend) reach Running/Healthy
3. Open the frontend URL from the Aspire dashboard
4. The default prompt "The capital of France is" should be pre-filled
5. Candidate list should appear automatically (no button click needed)
6. Results should show "Paris" (or similar) as the top candidate with high probability
7. Other candidates should appear with lower probabilities and shorter bars
8. Editing the prompt should trigger a new prediction after a short debounce delay
9. Verify endpoint appears in Scalar API docs at `/scalar/v1`
