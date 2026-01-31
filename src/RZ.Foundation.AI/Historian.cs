namespace RZ.Foundation.AI;

/// <summary>
/// The Historian class manages and maintains chat history for AI conversations.
/// It tracks chat entries, maintains conversation context, and provides historical
/// data for chat agents. This helps in maintaining conversation continuity and
/// context awareness across multiple interactions.
/// </summary>
/// <param name="chat">The chat function used for AI communication</param>
/// <param name="entries">Optional initial collection of chat messages to populate history</param>
[PublicAPI]
public class Historian(AiChatFunc chat, IEnumerable<ChatMessage>? entries = null)
{
    readonly List<ChatMessage> history = [..entries ?? []];

    public IReadOnlyList<ChatMessage> History => history;

    public AgentResponse Invoke(string message, ChatRole role = ChatRole.User)
        => Invoke(new ChatMessage.Content(role, message));

    public async AgentResponse Invoke(ChatMessage message) {
        history.Add(message);
        if (Fail(await chat(history), out var e, out var response, out var cost)) return e.Trace();

        history.AddRange(from r in response select r.Message);
        return (response, cost);
    }
}