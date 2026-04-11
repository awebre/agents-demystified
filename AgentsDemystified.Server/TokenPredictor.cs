using OllamaSharp;
using OllamaSharp.Models;

namespace AgentsDemystified.Server;

public record TokenCandidate(string Token, double LogProbability, double Probability);

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
        if (candidates.Count == 0) candidates = [new TokenCandidate("<|endoftext|>", 0, 1.0)];

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

    private static bool IsSpecialToken(string token)
    {
        return token.StartsWith("<|") && token.EndsWith("|>");
    }

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