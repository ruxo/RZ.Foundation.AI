using System.Text.Json;

namespace RZ.Foundation.AI;

public static class LLM
{
    public const decimal USD2Satang = 38_00m;

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
            if (Fail(await chat(m), out var e, out var responses, out var cost)) return e;
            var toolCalls = (from r in responses
                             let tc = r.Message as ChatMessage.ToolCall
                             where tc is not null
                             select tc).ToArray();
            if (toolCalls.Length == 0)
                return (responses, cost);

            var result = await Task.WhenAll(from tc in toolCalls
                                            from request in tc.Requests
                                            select Call(wrappers, request).AsTask());
            if (Fail(result.MakeList(), out e, out var toolResults)) return e;

            var now = (clock ?? TimeProvider.System).GetUtcNow();
            var toolResponse = toolResults.Map(tr => new ChatEntry(now, tr, Admin: null, ChatCost.Zero));
            m = m.Concat(responses.Map(x => x.Message)).Concat(toolResults);

            if (Fail(await chat(m), out e, out var responses2, out var cost2)) return e.Trace("Chat failed");

            return (responses.Concat(toolResponse).Concat(responses2).ToArray(), cost + cost2);
        };

    static async ValueTask<Outcome<ChatMessage.ToolResult>> Call(IReadOnlyList<ToolWrapper> wrappers, ToolRequest callInfo) {
        if (!IfSome(wrappers.TryFirst(t => t.Definition.Name == callInfo.Function), out var tool))
            return new ErrorInfo(InvalidRequest, $"Unknown tool: {callInfo.Function}");

        if (Fail(tool.ParseParameters(callInfo.Arguments), out var e, out var parameters)) return e.Trace();
        if (FailButNotFound(await tool.Call(parameters), out e, out var result)) return e.Trace();

        if (result is null)
            return new ErrorInfo(InvalidResponse, $"Tool {callInfo.Function} must not return null");
        return new ChatMessage.ToolResult(new(callInfo.Id, JsonSerializer.SerializeToNode(result)!));
    }
}