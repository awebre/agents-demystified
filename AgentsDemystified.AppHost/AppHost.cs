var builder = DistributedApplication.CreateBuilder(args);

var ollama = builder.AddOllama("ollama")
    .WithDataVolume();

var phi3 = ollama.AddModel("phi3", "phi3");

builder.AddProject<Projects.AgentsDemystified_Console>("console")
    .WithReference(phi3)
    .WaitFor(phi3);

builder.Build().Run();
