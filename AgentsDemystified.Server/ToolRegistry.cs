using System.Text.Json;

namespace AgentsDemystified.Server;

public record ToolDefinition(string Name, string Description, Dictionary<string, string> Parameters);

public class ToolRegistry
{
    private readonly string projectRoot;

    public ToolRegistry(string projectRoot)
    {
        this.projectRoot = Path.GetFullPath(projectRoot);
    }

    public List<ToolDefinition> GetTools() =>
    [
        new("search_files",
            "Search for files in the project whose name or content matches the query. Returns matching file paths.",
            new() { ["query"] = "The search term to match against file names and contents" }),
        new("read_file",
            "Read the full contents of a file. Returns the file text.",
            new() { ["path"] = "File path relative to the project root" }),
    ];

    public string GetToolPromptSection()
    {
        var section = "\n\nYou have access to the following tools:\n\n";
        foreach (var tool in GetTools())
        {
            section += $"## {tool.Name}\n{tool.Description}\n";
            section += "Parameters: " + JsonSerializer.Serialize(tool.Parameters) + "\n\n";
        }
        // This section is intentionally MECHANICAL ONLY — it teaches the model
        // the tool-call SYNTAX, nothing about when or whether to use the tools.
        // Behavior steering (when to call, search→read chaining, "you can read
        // files") lives in the user-editable System Prompt. That separation is a
        // deliberate teaching beat: with a generic system prompt the model has
        // the tools but won't use them well, and the presenter fixes it live by
        // editing the System Prompt — demonstrating that tools are capability
        // and the system prompt is behavior.
        section += """
            To call a tool, output EXACTLY this format on its own line and NOTHING else in that turn — no prose before or after:
            <tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>

            Fill in every required argument with a concrete value; never emit empty arguments. After each tool call you will receive a <tool_result>...</tool_result> containing the output.
            """;
        return section;
    }

    public async Task<string> ExecuteAsync(string name, JsonElement arguments)
    {
        try
        {
            return name switch
            {
                "search_files" => await SearchFiles(GetStringArg(arguments, "query")),
                "read_file" => await ReadFile(GetStringArg(arguments, "path")),
                _ => JsonSerializer.Serialize(new { error = $"Unknown tool: {name}" })
            };
        }
        catch (Exception ex)
        {
            return JsonSerializer.Serialize(new { error = ex.Message });
        }
    }

    private static string GetStringArg(JsonElement arguments, string name)
    {
        if (arguments.ValueKind != JsonValueKind.Object || !arguments.TryGetProperty(name, out var prop))
            throw new ArgumentException($"Missing required argument '{name}'");
        return prop.ValueKind switch
        {
            JsonValueKind.String => prop.GetString() ?? string.Empty,
            _ => prop.GetRawText().Trim('"'),
        };
    }

    private Task<string> SearchFiles(string query)
    {
        // Broad, forgiving match: split the query into terms and match any file
        // whose path or content contains ANY term. This lets the model get away
        // with sloppy natural-language queries (e.g. "ChatTemplateBuilder special
        // tokens" still finds ChatTemplateBuilder.cs). Files are ranked by how
        // many distinct terms they match so the most relevant float to the top.
        var terms = query
            .Split([' ', '\t', '\n', ',', '.', ':', ';', '"', '\'', '(', ')', '[', ']', '{', '}', '/', '\\'],
                StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(t => t.ToLowerInvariant())
            .Distinct()
            .ToArray();

        if (terms.Length == 0)
            return Task.FromResult(JsonSerializer.Serialize(new { files = Array.Empty<string>() }));

        var scored = new List<(string Path, int Score)>();
        var searchDir = new DirectoryInfo(projectRoot);

        foreach (var file in searchDir.EnumerateFiles("*", SearchOption.AllDirectories))
        {
            var relativePath = Path.GetRelativePath(projectRoot, file.FullName);
            if (relativePath.Contains("/bin/") || relativePath.Contains("/obj/") ||
                relativePath.Contains("/node_modules/") || relativePath.StartsWith("."))
                continue;

            var haystack = relativePath.ToLowerInvariant();
            if (file.Length < 100_000)
            {
                try { haystack += "\n" + File.ReadAllText(file.FullName).ToLowerInvariant(); }
                catch { /* skip unreadable content, still match on path */ }
            }

            var score = terms.Count(haystack.Contains);
            if (score > 0)
                scored.Add((relativePath, score));
        }

        var files = scored
            .OrderByDescending(x => x.Score)
            .ThenBy(x => x.Path)
            .Select(x => x.Path)
            .Take(10);

        return Task.FromResult(JsonSerializer.Serialize(new { files }));
    }

    private Task<string> ReadFile(string path)
    {
        var fullPath = Path.GetFullPath(Path.Combine(projectRoot, path));
        if (!fullPath.StartsWith(projectRoot))
            return Task.FromResult(JsonSerializer.Serialize(new { error = "Path outside project directory" }));

        if (!File.Exists(fullPath))
            return Task.FromResult(JsonSerializer.Serialize(new { error = $"File not found: {path}" }));

        var content = File.ReadAllText(fullPath);
        if (content.Length > 5000)
            content = content[..5000] + "\n... (truncated)";

        return Task.FromResult(JsonSerializer.Serialize(new { path, content }));
    }
}
