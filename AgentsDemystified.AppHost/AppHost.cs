var builder = DistributedApplication.CreateBuilder(args);

var ollama = builder.AddOllama("ollama")
    .WithDataVolume();

var phi4mini = ollama.AddModel("phi4-mini", "phi4-mini");

var server = builder.AddProject<Projects.AgentsDemystified_Server>("server")
    .WithReference(phi4mini)
    .WaitFor(phi4mini)
    .WithHttpHealthCheck("/health")
    .WithExternalHttpEndpoints();

builder.AddViteApp("webfrontend", "../frontend")
    .WithReference(server)
    .WaitFor(server);

builder.Build().Run();
