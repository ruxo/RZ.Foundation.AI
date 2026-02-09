using RZ.Foundation.AI;

namespace UnitTests;

public sealed class LLMTests
{
    [Test, DisplayName("Tests that the LLM resolver returns the expected result for a valid input without tool call")]
    public async ValueTask ResolverWithoutToolCall() {
        var tools = ToolWrapper.FromType(typeof(AddTool));

        var resolver = LLM.CreateResolver(ChatNoCall, tools);

        // when call resolver
        var result = await resolver([new ChatMessage.Content(ChatRole.User, "Hello")]);

        // Then just get the mocked result
        await Assert.That(Success(result, out var r)).IsTrue();
        var (chat, cost) = r;
        await Assert.That(chat.Count).IsEqualTo(1);
        await Assert.That(chat[0].Message).IsTypeOf<ChatMessage.Content>()
                    .And.HasProperty(x => x.Message).IsEqualTo("Hi!");
        await Assert.That(cost).IsEqualTo(new ChatCost(1, 2));
    }

    [Test, DisplayName("Tests that the LLM resolver returns the expected result for a valid input with tool call")]
    public async ValueTask ResolverWithToolCall() {
        var tools = ToolWrapper.FromType(typeof(AddTool));
        var resolver = LLM.CreateResolver(ChatToolCall, tools);

        // When call resolver
        var result = await resolver([new ChatMessage.Content(ChatRole.User, "Hello")]);

        // Then just get the mocked result
        await Assert.That(Success(result, out var r)).IsTrue();
        var (chat, _) = r;
        await Assert.That(chat.Count).IsEqualTo(3);
        await Assert.That(chat[0].Message).IsTypeOf<ChatMessage.ToolCall>()
                    .And.Member(x => x.Requests.Count, req => req.IsEqualTo(1))
                    .And.Member(x => x.Requests[0].Function, func => func.IsEqualTo(nameof(AddTool.DoSomething)));
        await Assert.That(chat[1].Message).IsTypeOf<ChatMessage.ToolResult>()
                    .And.Member(x => x.Response.Id, id => id.IsEqualTo("123"))
                    .And.Member(x => x.Response.Response, json => json.Satisfies(j => j!.ToString() == "done"));
        await Assert.That(chat[2].Message).IsTypeOf<ChatMessage.Content>()
                    .And.HasProperty(x => x.Message).IsEqualTo("Got result");
    }

    static ValueTask<Outcome<(IReadOnlyList<ChatEntry>, ChatCost)>> ChatNoCall(IEnumerable<ChatMessage> messages) {
        var now = new DateTimeOffset(2026, 2, 1, 0, 0, 0, TimeSpan.Zero);
        ChatEntry[] entry = [new(now, new ChatMessage.Content(ChatRole.Agent, "Hi!"), Admin: null, ChatCost.Zero)];
        return new((entry, new ChatCost(1, 2)));
    }

    static ValueTask<Outcome<(IReadOnlyList<ChatEntry>, ChatCost)>> ChatToolCall(IEnumerable<ChatMessage> messages) {
        var now = new DateTimeOffset(2026, 2, 1, 0, 0, 0, TimeSpan.Zero);
        ChatMessage message = messages.Any(m => m is ChatMessage.ToolResult)
                                  ? new ChatMessage.Content(ChatRole.Agent, "Got result")
                                  : new ChatMessage.ToolCall([new("123", nameof(AddTool.DoSomething), Arguments: null)]);

        ChatEntry[] entry = [
            new(now, message,
                Admin: null, ChatCost.Zero)
        ];
        return new((entry, ChatCost.Zero));
    }

    static class AddTool
    {
        [AiToolName]
        public static string DoSomething() => "done";
    }
}