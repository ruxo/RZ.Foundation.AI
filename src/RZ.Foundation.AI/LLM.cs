using System.Text.Json;
using System.Text.Json.Nodes;

namespace RZ.Foundation.AI;

public static class LLM
{
    public const decimal USD2Satang = 38_00m;

    /// <summary>Most rounds of tool calls the resolver runs for one reply. A round is one chat reply that carries tool calls.</summary>
    public const int MAX_ROUNDS = 20;

    /// <summary>Most rounds in a row whose tool results are all <c>{ error }</c> results.</summary>
    public const int MAX_FAILED_ROUNDS = 5;

    /// <summary>Error code of a resolver that ran out of its tool budget.</summary>
    public const string TOOL_LOOP = "tool-loop";

    [Pure]
    public static ChatCost CalcCost(CostStructure rate, int input, int output, int thought) {
        var inputRate = input >= rate.InputThreshold ? rate.HighText : rate.Text;
        var inputCost = USD2Satang * inputRate.CalcInput(input);
        var outputRate = output >= rate.InputThreshold ? rate.HighText : rate.Text;
        var outputCost = USD2Satang * (outputRate.CalcOutput(output) + rate.Thought.CalcOutput(thought));
        return new(inputCost, outputCost);
    }

    public static AiChatFunc CreateResolver(AiChatFunc chat, IReadOnlyList<ToolWrapper> wrappers, TimeProvider? clock = null)
        => async messages => {
            var m = Seq(messages);
            var collected = new List<ChatEntry>();
            var cost = ChatCost.Zero;
            var loop = ToolLoop.Start;

            while (true){
                if (Fail(await chat(m), out var e, out var responses, out var chatCost)) return e.Trace("Chat failed");
                collected.AddRange(responses);
                cost += chatCost;

                var requests = (from r in responses
                                let tc = r.Message as ChatMessage.ToolCall
                                where tc is not null
                                from request in tc.Requests
                                select request).ToArray();
                if (requests.Length == 0)
                    return (collected.ToArray(), cost);

                if (loop.Rounds == MAX_ROUNDS) return loop.Exceeded(requests[^1].Function);

                var result = await Task.WhenAll(from request in requests
                                                select Call(wrappers, request).AsTask());
                if (Fail(result.MakeList(), out e, out var toolResults)) return e;

                loop = loop.Next(requests[^1].Function, toolResults);
                if (loop.FailedRounds == MAX_FAILED_ROUNDS) return loop.Failed();

                var now = (clock ?? TimeProvider.System).GetUtcNow();
                collected.AddRange(toolResults.Map(tr => new ChatEntry(now, tr.Result, Admin: null, ChatCost.Zero)));
                m = m.Concat(responses.Map(x => x.Message)).Concat(toolResults.Map(tr => (ChatMessage)tr.Result));
            }
        };

    /// <summary>
    /// A tool call's result. <paramref name="Error"/> is the message of a model-caused failure, which <paramref name="Result"/> carries as
    /// <c>{ "error": … }</c>, or null when the tool ran.
    /// </summary>
    readonly record struct ToolOutcome(ChatMessage.ToolResult Result, string? Error);

    readonly record struct ToolLoop(int Rounds, int FailedRounds, string LastTool, string? LastError)
    {
        public static readonly ToolLoop Start = new(0, 0, string.Empty, LastError: null);

        [Pure]
        public ToolLoop Next(string lastTool, IReadOnlyList<ToolOutcome> results)
            => new(Rounds + 1,
                   results.All(r => r.Error is not null) ? FailedRounds + 1 : 0,
                   lastTool,
                   results[^1].Error);

        [Pure]
        public ErrorInfo Failed()
            => Error($"Tool `{LastTool}` failed {MAX_FAILED_ROUNDS} times in a row: {LastError}", LastTool, LastError);

        [Pure]
        public ErrorInfo Exceeded(string tool)
            => Error($"Tool calls exceeded {MAX_ROUNDS} rounds in one reply; last tool `{tool}`", tool, LastError);

        static ErrorInfo Error(string message, string tool, string? lastError)
            => new(TOOL_LOOP, message, data: new JsonObject { ["tool"] = tool, ["error"] = lastError ?? message }.ToJsonString());
    }

    static async ValueTask<Outcome<ToolOutcome>> Call(IReadOnlyList<ToolWrapper> wrappers, ToolRequest callInfo) {
        if (!IfSome(wrappers.TryFirst(t => t.Definition.Name == callInfo.Function), out var tool))
            return ErrorResult(callInfo.Id, $"Unknown tool: {callInfo.Function}");

        if (Fail(tool.ParseParameters(callInfo.Arguments), out var e, out var parameters)) return ErrorResult(callInfo.Id, e.Message);
        if (FailButNotFound(await tool.Call(parameters), out e, out var result)) return e.Trace();

        if (result is null)
            return new ErrorInfo(InvalidResponse, $"Tool {callInfo.Function} must not return null");
        return new ToolOutcome(new ChatMessage.ToolResult(new(callInfo.Id, JsonSerializer.SerializeToNode(result)!)), Error: null);
    }

    [Pure]
    static ToolOutcome ErrorResult(string callId, string message)
        => new(new ChatMessage.ToolResult(new(callId, new JsonObject { ["error"] = message })), message);
}
