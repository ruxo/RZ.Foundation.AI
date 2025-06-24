using System.Text.Json;

namespace RZ.Foundation.AI;

[PublicAPI]
public static class LLM
{
    public const decimal USD2Satang = 38_00m;

    [System.Diagnostics.Contracts.Pure]
    public static ChatCost CalcCost(CostStructure rate, int input, int output, int thought) {
        var inputRate = input >= rate.InputThreshold? rate.HighText : rate.Text;
        var inputCost = USD2Satang * inputRate.CalcInput(input);
        var outputRate = output >= rate.InputThreshold? rate.HighText : rate.Text;
        var outputCost = USD2Satang * (outputRate.CalcOutput(output) + rate.Thought.CalcOutput(thought));
        return new(inputCost, outputCost);
    }

    public static AiChatFunc CreateResolver(AiChatFunc chat, IReadOnlyList<ToolWrapper> wrappers, TimeProvider? clock = null)
        => async messages => {
            var m = Seq(messages);
            var (responses, cost) = await chat(m);
            var toolCalls = (from r in responses
                             let tc = r.Message as ChatMessage.ToolCall
                             where tc is not null
                             select tc).ToArray();
            if (toolCalls.Length == 0)
                return (responses, cost);

            var toolResults = await Task.WhenAll(from tc in toolCalls
                                                 from request in tc.Requests
                                                 select Call(wrappers, request));
            var now = (clock ?? TimeProvider.System).GetUtcNow();
            var toolResponse = toolResults.Map(tr => new ChatEntry(now, tr, Admin: null, ChatCost.Zero));
            m = m.Concat(responses.Map(x => x.Message)).Concat(toolResults);
            var (responses2, cost2) = await chat(m);
            return (responses.Concat(toolResponse).Concat(responses2).ToArray(), cost + cost2);
        };

    static async Task<ChatMessage.ToolResult> Call(IReadOnlyList<ToolWrapper> wrappers, ToolRequest callInfo) {
        var tool = wrappers.First(t => t.Definition.Name == callInfo.Function);
        var parameters = tool.ParseParameters(callInfo.Arguments);
        var result = await tool.Call(parameters);
        if (result is null)
            throw new InvalidOperationException($"Tool {callInfo.Function} must not return null");
        return new ChatMessage.ToolResult(new(callInfo.Id, JsonSerializer.SerializeToNode(result)!));
    }
}