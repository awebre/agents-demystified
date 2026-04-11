# Agents Demystified

A live demo that progressively reveals how AI agents work, starting from raw token prediction and building up to tool calling. Designed for conference presentations where an AI agent reads a spec and implements each phase on stage.

## Architecture

- **.NET 10 Aspire** orchestrates all services (AppHost)
- **Ollama** (phi3 model) for local LLM inference
- **ASP.NET Minimal API** backend (`AgentsDemystified.Server`)
- **Vite React** frontend (`frontend/`)
- **Scalar** for API docs at `/scalar/v1`

## Demo Phases

1. **Next Token Prediction** — autocomplete UI showing top N candidates with probabilities
2. **Autoregressive Generation** — Phase 1 in a loop with SSE streaming and EOS detection
3. **Chat Framing** — system prompt + chat template, split view showing the raw prompt
4. **Tool Calling** — custom tool schema, execution, results fed back into the loop

## Approach: Specs as Starting Lines

Each phase has a **prescriptive spec** in `docs/` committed to `main`. During the live demo, an AI agent reads the spec and implements it on stage. Specs are intentionally detailed with exact code snippets so the agent can reliably reproduce the implementation.

### Git Strategy

- **`main`** — specs only (the "starting line" for each demo phase)
- **`phase-N`** — stacked branches with implementations (`phase-1` off `main`, `phase-2` off `phase-1`, etc.)
- To reset for a demo: check out `main` and let the agent build from the spec

## Running

```bash
cd AgentsDemystified.AppHost
aspire run
```

The Aspire dashboard URL is printed at startup. The frontend proxies `/api`, `/scalar`, and `/openapi` to the backend automatically.

## Ollama Quirks

- **EOS tokens**: Ollama suppresses special tokens (like `<|endoftext|>`) in the `Response` field, returning an empty string. The actual token is available in logprobs. Code must recover it from the top logprob candidate.
- **`Done` flag**: Always `true` for `NumPredict = 1` requests. Stop detection must use token inspection (empty or special token pattern), not Ollama's `Done` flag.
