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

        // Ollama suppresses special tokens in the Response field, returning "".
        // When that happens, grab the actual token from the top logprob candidate
        // so the frontend can display what the model really predicted (e.g. <|endoftext|>).
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
