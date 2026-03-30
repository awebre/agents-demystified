using OllamaSharp;
using OllamaSharp.Models;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);
builder.AddServiceDefaults();
builder.AddOllamaApiClient("phi4-mini");
builder.Services.AddProblemDetails();
builder.Services.AddOpenApi();

var app = builder.Build();
app.UseExceptionHandler();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.MapScalarApiReference();
}

var api = app.MapGroup("/api");

api.MapPost("/predict", async (PredictRequest request, IOllamaApiClient client) =>
{
    var generateRequest = new GenerateRequest
    {
        Model = "phi3",
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

app.MapDefaultEndpoints();
app.Run();

record PredictRequest(string Prompt, int TopN = 10);

record TokenCandidate(string Token, double LogProbability, double Probability);

record PredictResponse(string PredictedToken, List<TokenCandidate> Candidates);
