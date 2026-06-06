using System.Diagnostics;
using System.Text.Json;
using AgentsDemystified.Server;
using OllamaSharp;
using Scalar.AspNetCore;

var agentActivitySource = new ActivitySource("AgentsDemystified.Server.Agent");

var builder = WebApplication.CreateBuilder(args);
builder.AddServiceDefaults();
builder.AddOllamaApiClient("phi4-mini");
builder.Services.AddTransient<TokenPredictor>();
builder.Services.AddSingleton(new ToolRegistry(
    Path.GetFullPath(Path.Combine(builder.Environment.ContentRootPath, ".."))));
builder.Services.AddProblemDetails();
builder.Services.AddOpenApi();
builder.Services.AddOpenTelemetry()
    .WithTracing(tracing => tracing.AddSource("AgentsDemystified.Server.Agent"));

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

api.MapGet("/tools", (ToolRegistry tools) =>
    {
        return Results.Ok(tools.GetTools());
    })
    .WithName("ListTools")
    .WithDescription("List available tools and their schemas");

api.MapPost("/agent", async (AgentRequest request, TokenPredictor predictor, ToolRegistry tools, ILogger<Program> log, HttpContext context) =>
    {
        context.Response.ContentType = "text/event-stream";
        context.Response.Headers.CacheControl = "no-cache";
        context.Response.Headers.Connection = "keep-alive";

        using var loopActivity = agentActivitySource.StartActivity("agent.loop");
        loopActivity?.SetTag("agent.messages.count", request.Messages.Count);
        loopActivity?.SetTag("agent.max_tokens", request.MaxTokens);
        loopActivity?.SetTag("agent.max_tool_calls", request.MaxToolCalls);
        loopActivity?.SetTag("agent.temperature", request.Temperature);
        loopActivity?.SetTag("agent.system_prompt", request.SystemPrompt);
        var lastUserMessage = request.Messages.LastOrDefault(m => m.Role == "user")?.Content ?? "";
        loopActivity?.SetTag("agent.user_message", lastUserMessage);
        loopActivity?.SetTag("agent.conversation",
            string.Join("\n", request.Messages.Select(m => $"[{m.Role}] {m.Content}")));

        log.LogInformation("agent.loop.start messages={MessageCount} maxTokens={MaxTokens} maxToolCalls={MaxToolCalls} temp={Temperature} userMessage={UserMessage}",
            request.Messages.Count, request.MaxTokens, request.MaxToolCalls, request.Temperature, lastUserMessage);

        // Inject tool definitions into the system prompt — this is how the model "learns" about tools
        var systemPromptWithTools = request.SystemPrompt + tools.GetToolPromptSection();
        var prompt = ChatTemplateBuilder.BuildPrompt(systemPromptWithTools, request.Messages);

        var toolCallCount = 0;

        // Guardrail: remember which tool calls we've already run this turn so we
        // can short-circuit duplicates. phi4-mini at temperature 0 tends to
        // re-issue the same read_file instead of answering; re-executing it
        // re-injects the full file contents, bloating the prompt until a single
        // Ollama call exceeds its timeout and the stream dies. Skipping the
        // re-execution keeps context bounded and nudges the model to answer.
        var executedCalls = new HashSet<string>();

        // THE AGENT LOOP — generate, detect tool call, execute, inject result, repeat
        while (toolCallCount <= request.MaxToolCalls)
        {
            using var iterActivity = agentActivitySource.StartActivity("agent.iteration");
            iterActivity?.SetTag("agent.iteration", toolCallCount);

            var generatedText = "";
            var tokenCount = 0;
            var hitClosingTag = false;
            var eosHit = false;

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
                {
                    eosHit = true;
                    break;
                }

                prompt += result.Token;
                generatedText += result.Token;
                tokenCount++;

                if (generatedText.Contains("</tool_call>"))
                {
                    hitClosingTag = true;
                    break;
                }
            }

            iterActivity?.SetTag("agent.tokens_generated", tokenCount);
            iterActivity?.SetTag("agent.hit_closing_tag", hitClosingTag);
            iterActivity?.SetTag("agent.eos_hit", eosHit);
            iterActivity?.SetTag("agent.generated_text", generatedText);

            log.LogInformation(
                "agent.iteration idx={Idx} tokens={Tokens} closingTag={ClosingTag} eos={Eos} text={Text}",
                toolCallCount, tokenCount, hitClosingTag, eosHit, generatedText);

            var toolCall = ParseToolCall(generatedText);
            if (toolCall is null)
            {
                iterActivity?.SetTag("agent.tool_call_parsed", false);
                log.LogInformation("agent.no_tool_call — loop exit");
                break;
            }

            toolCallCount++;
            var argsJson = toolCall.Arguments.GetRawText();
            var callKey = $"{toolCall.Name}:{argsJson}";
            var isDuplicate = !executedCalls.Add(callKey);

            iterActivity?.SetTag("agent.tool_call_parsed", true);
            iterActivity?.SetTag("agent.tool_call.name", toolCall.Name);
            iterActivity?.SetTag("agent.tool_call.arguments", argsJson);
            iterActivity?.SetTag("agent.tool_call.duplicate", isDuplicate);

            log.LogInformation("agent.tool_call name={Name} args={Args} duplicate={Duplicate}",
                toolCall.Name, argsJson, isDuplicate);

            await StreamEvent(context, new
            {
                type = "tool_call",
                name = toolCall.Name,
                arguments = toolCall.Arguments
            });

            // Duplicate guardrail: the model re-issued a call we already ran this
            // turn. Don't re-execute (that would re-inject the full result and
            // bloat the prompt). Feed back a short nudge instead so it answers.
            if (isDuplicate)
            {
                var nudge = JsonSerializer.Serialize(new
                {
                    note = $"You already called {toolCall.Name} with these arguments and received the result above. Do not call it again — answer the user's question now using what you already have."
                });

                await StreamEvent(context, new
                {
                    type = "tool_result",
                    name = toolCall.Name,
                    result = nudge
                });

                prompt +=
                    $"{ChatTemplateBuilder.EndToken}\n" +
                    $"{ChatTemplateBuilder.UserToken}\n<tool_result>{NeutralizeControlTokens(nudge)}</tool_result>\n{ChatTemplateBuilder.EndToken}\n" +
                    $"{ChatTemplateBuilder.AssistantToken}\n";
                continue;
            }

            using (var execActivity = agentActivitySource.StartActivity("tool.execute"))
            {
                execActivity?.SetTag("tool.name", toolCall.Name);
                execActivity?.SetTag("tool.arguments", argsJson);

                var toolResult = await tools.ExecuteAsync(toolCall.Name, toolCall.Arguments);

                var preview = toolResult.Length > 500 ? toolResult[..500] + "..." : toolResult;
                execActivity?.SetTag("tool.result_length", toolResult.Length);
                execActivity?.SetTag("tool.result_preview", preview);

                log.LogInformation("tool.execute name={Name} resultLength={Len} preview={Preview}",
                    toolCall.Name, toolResult.Length, preview);

                await StreamEvent(context, new
                {
                    type = "tool_result",
                    name = toolCall.Name,
                    result = toolResult
                });

                prompt +=
                    $"{ChatTemplateBuilder.EndToken}\n" +
                    $"{ChatTemplateBuilder.UserToken}\n<tool_result>{NeutralizeControlTokens(toolResult)}</tool_result>\n{ChatTemplateBuilder.EndToken}\n" +
                    $"{ChatTemplateBuilder.AssistantToken}\n";
            }
        }

        loopActivity?.SetTag("agent.tool_calls_executed", toolCallCount);
        log.LogInformation("agent.loop.done toolCalls={Count}", toolCallCount);

        await StreamEvent(context, new { type = "done" });
    })
    .WithName("AgentStream")
    .WithDescription("Agent loop — chat with tool calling. The capstone: predict → generate → chat → agent");

app.MapDefaultEndpoints();
app.Run();

// Neutralize control tokens inside tool-result content before it is injected
// into the prompt. A read file (or search hit) may itself contain the chat
// template delimiters (<|end|>, <|user|>, …) or tool-call tags (<tool_call>…),
// e.g. when the agent reads this project's own spec docs. Left raw, the
// tokenizer treats those embedded tokens as real turn boundaries / tool calls,
// corrupting the conversation structure and derailing the model. Rewriting them
// to look-alike angle brackets keeps the content human-readable while making it
// inert. (Only the prompt copy is sanitized; the copy streamed to the UI stays
// verbatim so the audience sees the true file contents.) This also closes a
// mild prompt-injection vector — tool output can no longer forge framing.
static string NeutralizeControlTokens(string content) => content
    .Replace("<|system|>", "‹system›")
    .Replace("<|user|>", "‹user›")
    .Replace("<|assistant|>", "‹assistant›")
    .Replace("<|end|>", "‹end›")
    .Replace("<tool_call>", "‹tool_call›")
    .Replace("</tool_call>", "‹/tool_call›")
    .Replace("<tool_result>", "‹tool_result›")
    .Replace("</tool_result>", "‹/tool_result›");

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
        var args = doc.RootElement.GetProperty("arguments").Clone();
        return new ToolCallParsed(name, args);
    }
    catch
    {
        return null;
    }
}

static async Task StreamEvent(HttpContext context, object data)
{
    var payload = JsonSerializer.Serialize(data, JsonSerializerOptions.Web);
    await context.Response.WriteAsync($"data: {payload}\n\n");
    await context.Response.Body.FlushAsync();
}

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

record AgentRequest(
    string SystemPrompt,
    List<ChatMessage> Messages,
    int MaxTokens = 500,
    int TopN = 5,
    double Temperature = 0,
    int MaxToolCalls = 5);

record ToolCallParsed(string Name, JsonElement Arguments);