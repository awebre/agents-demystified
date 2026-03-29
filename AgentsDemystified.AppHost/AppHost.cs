var builder = DistributedApplication.CreateBuilder(args);

var ollama = builder.AddOllama("ollama")
    .WithDataVolume();

var phi3 = ollama.AddModel("phi3", "phi3");

var server = builder.AddProject<Projects.AgentsDemystified_Server>("server")
    .WithReference(phi3)
    .WaitFor(phi3)
    .WithHttpHealthCheck("/health")
    .WithExternalHttpEndpoints();

builder.AddViteApp("webfrontend", "../frontend")
    .WithReference(server)
    .WaitFor(server);

builder.Build().Run();
