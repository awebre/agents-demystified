namespace AgentsDemystified.Server;

public static class ChatTemplateBuilder
{
    public const string SystemToken = "<|system|>";
    public const string UserToken = "<|user|>";
    public const string AssistantToken = "<|assistant|>";
    public const string EndToken = "<|end|>";

    /// <summary>
    /// Assemble a chat prompt from a system prompt and message history.
    /// This is the entire "trick" of chat — it's just string concatenation
    /// with special tokens.
    /// </summary>
    public static string BuildPrompt(string systemPrompt, List<ChatMessage> messages)
    {
        var prompt = $"{SystemToken}\n{systemPrompt}\n{EndToken}\n";
        foreach (var msg in messages)
        {
            var roleToken = msg.Role == "user" ? UserToken : AssistantToken;
            prompt += $"{roleToken}\n{msg.Content}\n{EndToken}\n";
        }
        prompt += $"{AssistantToken}\n";
        return prompt;
    }
}
