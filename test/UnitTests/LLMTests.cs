using System.Text.Json;
using System.Text.Json.Nodes;
using RZ.Foundation.AI;
using RZ.Foundation.Types;

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

    [Test, DisplayName("A tool call missing a required parameter becomes an { error } tool result the model can act on")]
    public async ValueTask MissingParameterBecomesErrorResult() {
        var tools = new PlanTools();
        var chat = new ScriptedChat(ToolCallOf("c1", "needs_name", JsonNode.Parse("""{ "other": "x" }""")),
                                    Text("Which name?"));
        var resolver = LLM.CreateResolver(chat.Chat, ToolWrapper.From(tools));

        var result = await resolver([new ChatMessage.Content(ChatRole.User, "Hello")]);

        await Assert.That(Success(result, out var r)).IsTrue();
        var (entries, _) = r;
        await Assert.That(entries.Count).IsEqualTo(3);
        await Assert.That(entries[1].Message).IsTypeOf<ChatMessage.ToolResult>()
                    .And.Member(x => x.Response.Id, id => id.IsEqualTo("c1"))
                    .And.Member(x => x.Response.Response, json => json.Satisfies(j => IsErrorObject(j, "Missing parameter: name")));
        await Assert.That(entries[2].Message).IsTypeOf<ChatMessage.Content>()
                    .And.HasProperty(x => x.Message).IsEqualTo("Which name?");
    }

    [Test, DisplayName("A tool call naming an unknown tool becomes an { error } tool result the model can act on")]
    public async ValueTask UnknownToolBecomesErrorResult() {
        var tools = new PlanTools();
        var chat = new ScriptedChat(ToolCallOf("c1", "no_such_tool"), Text("Sorry"));
        var resolver = LLM.CreateResolver(chat.Chat, ToolWrapper.From(tools));

        var result = await resolver([new ChatMessage.Content(ChatRole.User, "Hello")]);

        await Assert.That(Success(result, out var r)).IsTrue();
        var (entries, _) = r;
        await Assert.That(entries.Count).IsEqualTo(3);
        await Assert.That(entries[1].Message).IsTypeOf<ChatMessage.ToolResult>()
                    .And.Member(x => x.Response.Id, id => id.IsEqualTo("c1"))
                    .And.Member(x => x.Response.Response, json => json.Satisfies(j => IsErrorObject(j, "Unknown tool: no_such_tool")));
    }

    [Test, DisplayName("The resolver keeps running tool calls until the model answers in text")]
    public async ValueTask ResolverLoopsUntilText() {
        var tools = new PlanTools();
        var chat = new ScriptedChat(ToolCallOf("a1", "tool_a"), ToolCallOf("b1", "tool_b"), Text("All done"));
        var resolver = LLM.CreateResolver(chat.Chat, ToolWrapper.From(tools));

        var result = await resolver([new ChatMessage.Content(ChatRole.User, "Hello")]);

        await Assert.That(Success(result, out var r)).IsTrue();
        var (entries, _) = r;
        await Assert.That(entries.Count).IsEqualTo(5);
        await Assert.That(entries[0].Message).IsTypeOf<ChatMessage.ToolCall>()
                    .And.Member(x => x.Requests[0].Id, id => id.IsEqualTo("a1"));
        await Assert.That(entries[1].Message).IsTypeOf<ChatMessage.ToolResult>()
                    .And.Member(x => x.Response.Id, id => id.IsEqualTo("a1"));
        await Assert.That(entries[2].Message).IsTypeOf<ChatMessage.ToolCall>()
                    .And.Member(x => x.Requests[0].Id, id => id.IsEqualTo("b1"));
        await Assert.That(entries[3].Message).IsTypeOf<ChatMessage.ToolResult>()
                    .And.Member(x => x.Response.Id, id => id.IsEqualTo("b1"));
        await Assert.That(entries[4].Message).IsTypeOf<ChatMessage.Content>()
                    .And.HasProperty(x => x.Message).IsEqualTo("All done");
        await Assert.That(tools.ACount).IsEqualTo(1);
        await Assert.That(tools.BCount).IsEqualTo(1);
        await Assert.That(chat.Calls.Count).IsEqualTo(3);
        await Assert.That(chat.Calls[2].Length).IsEqualTo(5);
    }

    [Test, DisplayName("The resolver returns the sum of every chat's cost across the rounds")]
    public async ValueTask ResolverSumsCostAcrossRounds() {
        var tools = new PlanTools();
        var chat = new ScriptedChat([ToolCallOf("a1", "tool_a"), ToolCallOf("b1", "tool_b"), Text("All done")],
                                    n => new ChatCost(n + 1, 10 * (n + 1)));
        var resolver = LLM.CreateResolver(chat.Chat, ToolWrapper.From(tools));

        var result = await resolver([new ChatMessage.Content(ChatRole.User, "Hello")]);

        await Assert.That(Success(result, out var r)).IsTrue();
        await Assert.That(r.Cost).IsEqualTo(new ChatCost(6, 60));
    }

    [Test, DisplayName("MAX_FAILED_ROUNDS consecutive all-error rounds fail the resolver with tool-loop")]
    public async ValueTask ConsecutiveErrorRoundsFailWithToolLoop() {
        var tools = new PlanTools();
        var errorRounds = Enumerable.Range(1, LLM.MAX_FAILED_ROUNDS)
                                    .Select(i => ToolCallOf($"c{i}", "needs_name", JsonNode.Parse("""{ "other": "x" }""")));
        var chat = new ScriptedChat([..errorRounds, Text("Never reached")]);
        var resolver = LLM.CreateResolver(chat.Chat, ToolWrapper.From(tools));

        var result = await resolver([new ChatMessage.Content(ChatRole.User, "Hello")]);

        await Assert.That(Fail(result, out var e)).IsTrue();
        await Assert.That(e!.Code).IsEqualTo("tool-loop");
        await Assert.That(e.Message).IsEqualTo("Tool `needs_name` failed 5 times in a row: Missing parameter: name");
        await Assert.That(JsonNode.DeepEquals(JsonNode.Parse(e.Data!), JsonNode.Parse("""{ "tool": "needs_name", "error": "Missing parameter: name" }""")))
                    .IsTrue();
        await Assert.That(chat.Calls.Count).IsEqualTo(LLM.MAX_FAILED_ROUNDS);
    }

    [Test, DisplayName("A round with a successful tool result resets the consecutive-error count")]
    public async ValueTask SuccessfulRoundResetsErrorCount() {
        var tools = new PlanTools();
        ChatMessage ErrorRound(int i) => ToolCallOf($"e{i}", "needs_name", JsonNode.Parse("""{ "other": "x" }"""));
        var chat = new ScriptedChat([
            ..Enumerable.Range(1, 4).Select(ErrorRound),
            ToolCallOf("a1", "tool_a"),
            ..Enumerable.Range(5, 4).Select(ErrorRound),
            Text("Done")
        ]);
        var resolver = LLM.CreateResolver(chat.Chat, ToolWrapper.From(tools));

        var result = await resolver([new ChatMessage.Content(ChatRole.User, "Hello")]);

        await Assert.That(Success(result, out var r)).IsTrue();
        await Assert.That(r.Chat.Count).IsEqualTo(19);
        await Assert.That(r.Chat[^1].Message).IsTypeOf<ChatMessage.Content>()
                    .And.HasProperty(x => x.Message).IsEqualTo("Done");
    }

    [Test, DisplayName("A model still calling a tool after MAX_ROUNDS rounds fails the resolver with tool-loop")]
    public async ValueTask ExceedingMaxRoundsFailsWithToolLoop() {
        var tools = new PlanTools();
        var rounds = Enumerable.Range(1, LLM.MAX_ROUNDS)
                               .Select(i => i % 2 == 1
                                                ? ToolCallOf($"a{i}", "tool_a")
                                                : ToolCallOf($"e{i}", "needs_name", JsonNode.Parse("""{ "other": "x" }""")));
        var chat = new ScriptedChat([..rounds, ToolCallOf("e21", "needs_name", JsonNode.Parse("""{ "other": "x" }""")), Text("Never reached")]);
        var resolver = LLM.CreateResolver(chat.Chat, ToolWrapper.From(tools));

        var result = await resolver([new ChatMessage.Content(ChatRole.User, "Hello")]);

        await Assert.That(Fail(result, out var e)).IsTrue();
        await Assert.That(e!.Code).IsEqualTo("tool-loop");
        await Assert.That(e.Message).IsEqualTo("Tool calls exceeded 20 rounds in one reply; last tool `needs_name`");
        await Assert.That(JsonNode.DeepEquals(JsonNode.Parse(e.Data!), JsonNode.Parse("""{ "tool": "needs_name", "error": "Missing parameter: name" }""")))
                    .IsTrue();
        await Assert.That(chat.Calls.Count).IsEqualTo(LLM.MAX_ROUNDS + 1);
    }

    [Test, DisplayName("MAX_ROUNDS rounds of successful tool calls followed by text succeed: the cap bites only at a 21st round")]
    public async ValueTask MaxRoundsThenTextSucceeds() {
        var tools = new PlanTools();
        var rounds = Enumerable.Range(1, LLM.MAX_ROUNDS).Select(i => ToolCallOf($"a{i}", "tool_a"));
        var chat = new ScriptedChat([..rounds, Text("Done")]);
        var resolver = LLM.CreateResolver(chat.Chat, ToolWrapper.From(tools));

        var result = await resolver([new ChatMessage.Content(ChatRole.User, "Hello")]);

        await Assert.That(Success(result, out var r)).IsTrue();
        await Assert.That(r.Chat.Count).IsEqualTo(2 * LLM.MAX_ROUNDS + 1);
        await Assert.That(tools.ACount).IsEqualTo(LLM.MAX_ROUNDS);
    }

    [Test, DisplayName("A tool whose own code throws fails the resolver, not with tool-loop")]
    public async ValueTask ThrowingToolFailsTheChat() {
        var tools = new PlanTools();
        var chat = new ScriptedChat(ToolCallOf("x1", "explode"), Text("Never reached"));
        var resolver = LLM.CreateResolver(chat.Chat, ToolWrapper.From(tools));

        var result = await resolver([new ChatMessage.Content(ChatRole.User, "Hello")]);

        await Assert.That(Fail(result, out var e)).IsTrue();
        await Assert.That(e!.Code).IsNotEqualTo(LLM.TOOL_LOOP);
        await Assert.That(chat.Calls.Count).IsEqualTo(1);
    }

    [Test, DisplayName("The tool budget and the tool-loop code are public constants")]
    public async ValueTask BudgetConstants() {
        static object? PublicConstant(string name)
            => typeof(LLM).GetField(name) is { IsPublic: true, IsLiteral: true } f ? f.GetRawConstantValue() : null;

        await Assert.That(PublicConstant(nameof(LLM.MAX_ROUNDS))).IsEqualTo(20);
        await Assert.That(PublicConstant(nameof(LLM.MAX_FAILED_ROUNDS))).IsEqualTo(5);
        await Assert.That(PublicConstant(nameof(LLM.TOOL_LOOP))).IsEqualTo("tool-loop");
    }

    static bool IsErrorObject(JsonNode? json, string expected)
        => json is JsonObject o && o.Count == 1 && o["error"]?.GetValue<string>() == expected;

    static ChatMessage ToolCallOf(string id, string function, JsonNode? arguments = null)
        => new ChatMessage.ToolCall([new(id, function, arguments)]);

    static ChatMessage Text(string message)
        => new ChatMessage.Content(ChatRole.Agent, message);

    /// <summary>
    /// A fake chat that answers call n with the n-th scripted reply, costing <c>cost(n)</c>, and records what each call received.
    /// </summary>
    sealed class ScriptedChat(IReadOnlyList<ChatMessage> replies, Func<int, ChatCost>? cost = null)
    {
        public ScriptedChat(params ChatMessage[] replies) : this((IReadOnlyList<ChatMessage>)replies) { }

        public List<ChatMessage[]> Calls { get; } = [];

        public ValueTask<Outcome<(IReadOnlyList<ChatEntry>, ChatCost)>> Chat(IEnumerable<ChatMessage> messages) {
            var n = Calls.Count;
            Calls.Add(messages.ToArray());
            if (n >= replies.Count){
                Outcome<(IReadOnlyList<ChatEntry>, ChatCost)> unscripted = new ErrorInfo("test-unscripted", $"Unscripted chat call {n + 1}");
                return new(unscripted);
            }
            var now = new DateTimeOffset(2026, 2, 1, 0, 0, 0, TimeSpan.Zero);
            ChatEntry[] entry = [new(now, replies[n], Admin: null, ChatCost.Zero)];
            return new((entry, cost?.Invoke(n) ?? ChatCost.Zero));
        }
    }

    sealed class PlanTools
    {
        int aCount, bCount;

        public int ACount => aCount;
        public int BCount => bCount;

        [AiToolName("tool_a")]
        public string A() {
            Interlocked.Increment(ref aCount);
            return "a done";
        }

        [AiToolName("tool_b")]
        public string B() {
            Interlocked.Increment(ref bCount);
            return "b done";
        }

        [AiToolName("needs_name")]
        public string NeedsName(string name) => $"Hello {name}";

        [AiToolName("explode")]
        public string Explode() => throw new InvalidOperationException("boom");
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