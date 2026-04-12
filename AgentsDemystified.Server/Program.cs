using System.Text.Json;
using AgentsDemystified.Server;
using OllamaSharp;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);
builder.AddServiceDefaults();
builder.AddOllamaApiClient("phi4-mini");
builder.Services.AddTransient<TokenPredictor>();
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

api.MapPost("/predict", async (PredictRequest request, TokenPredictor predictor) =>
    {
        var result = await predictor.PredictNextAsync(request.Prompt, request.TopN);
        return Results.Ok(new PredictResponse(result.Token, result.Candidates));
    })
    .WithName("PredictNextToken")
    .WithDescription("Predict the next token for a given prompt and return top N candidates with probabilities");

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

api.MapPost("/chat", async (ChatRequest request, TokenPredictor predictor, HttpContext context) =>
    {
        context.Response.ContentType = "text/event-stream";
        context.Response.Headers.CacheControl = "no-cache";
        context.Response.Headers.Connection = "keep-alive";

        // This is the entire "chat" logic — build a prompt string from messages
        var prompt = ChatTemplateBuilder.BuildPrompt(request.SystemPrompt, request.Messages);

        for (var i = 0; i < request.MaxTokens; i++)
        {
            context.RequestAborted.ThrowIfCancellationRequested();

            // Same prediction call as Phase 2 — nothing new here
            var result = await predictor.PredictNextAsync(
                prompt, request.TopN, request.Temperature, context.RequestAborted);

            var payload = JsonSerializer.Serialize(new
            {
                type = "token",
                token = result.Token,
                done = result.Done,
                candidates = result.Candidates
            }, JsonSerializerOptions.Web);

            await context.Response.WriteAsync($"data: {payload}\n\n");
            await context.Response.Body.FlushAsync();

            if (result.Done || string.IsNullOrEmpty(result.Token))
                break;

            prompt += result.Token;
        }
    })
    .WithName("ChatStream")
    .WithDescription("Chat with the model using structured messages — Phase 2's generation loop with a chat template");

app.MapDefaultEndpoints();
app.Run();

internal record PredictRequest(string Prompt, int TopN = 10);

internal record PredictResponse(string PredictedToken, List<TokenCandidate> Candidates);

internal record GenerateStreamRequest(string Prompt, int MaxTokens = 200, int TopN = 5, double Temperature = 0);

public record ChatMessage(string Role, string Content);

record ChatRequest(
    string SystemPrompt,
    List<ChatMessage> Messages,
    int MaxTokens = 200,
    int TopN = 5,
    double Temperature = 0);