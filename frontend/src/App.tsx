import { useState, useEffect } from "react";
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

function displayToken(token: string): string {
  return token
    .replace(/ /g, "\u2423")
    .replace(/\n/g, "\u21B5")
    .replace(/\t/g, "\u21E5")
    .replace(/\r/g, "\u23CE");
}

function App() {
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

  const maxProbability =
    result && result.candidates.length > 0
      ? result.candidates[0].probability
      : 1;

  return (
    <div className="app">
      <header className="header">
        <h1>Agents Demystified &mdash; Next Token Prediction</h1>
      </header>

      <main className="main">
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

          {result &&
            result.candidates.map((candidate, index) => (
              <div
                key={index}
                className={`candidate-row ${index === 0 ? "top-candidate" : ""}`}
                onClick={() => setPrompt((prev) => prev + candidate.token)}
              >
                <span className="candidate-token">
                  {displayToken(candidate.token)}
                </span>
                <span className="candidate-prob">
                  {(candidate.probability * 100).toFixed(1)}%
                </span>
                <div className="candidate-bar-bg">
                  <div
                    className="candidate-bar"
                    style={{
                      width:
                        (candidate.probability / maxProbability) * 100 + "%",
                    }}
                  />
                </div>
              </div>
            ))}
        </div>
      </main>
    </div>
  );
}

export default App;
