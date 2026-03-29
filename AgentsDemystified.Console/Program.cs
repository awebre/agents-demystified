using Hex1b;
using Hex1b.Input;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using OllamaSharp;

var builder = Host.CreateApplicationBuilder(args);
builder.AddServiceDefaults();
builder.AddOllamaApiClient("phi3");

var host = builder.Build();
await host.StartAsync();

var ollama = host.Services.GetRequiredService<IOllamaApiClient>();

string[] modes =
[
    "1. Next Token Prediction",
    "2. Autoregressive Generation Loop",
    "3. Chat with Raw Prompt View",
    "4. Tool Calling"
];

using var app = new Hex1bApp(ctx =>
    ctx.Border(b =>
    [
        b.Text("Select a demo mode to explore how LLMs work under the hood:"),
        b.Text(""),
        b.List(modes)
            .OnItemActivated(e =>
            {
                // TODO: navigate to the selected mode's screen
            }),
        b.Text(""),
        b.Text("Press Enter to select, Ctrl+Q to quit"),
    ]).Title("Agents Demystified")
    .WithInputBindings(bindings =>
    {
        bindings.Ctrl().Key(Hex1bKey.Q).Action(ctx => ctx.RequestStop(), "Quit");
    })
);

await app.RunAsync();
await host.StopAsync();
