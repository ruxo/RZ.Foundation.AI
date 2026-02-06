using System.Text.Json;
using FluentAssertions;
using RZ.Foundation.AI;

namespace FoundationAITests;

public sealed class StaticToolTests
{
    readonly IReadOnlyList<ToolWrapper> wrappers = ToolWrapper.FromType(typeof(AddTool));

    static class AddTool
    {
        [AiToolName("add_numbers")]
        public static string Add(int a, int b)
            => $"{a} + {b} = {a + b}";
    }

    [Fact]
    public void CheckDefinition() {
        wrappers.Count.Should().Be(1);
        wrappers[0].Definition.Should().BeEquivalentTo(new ToolDefinition("add_numbers", Description: null, [
            new("a", Description: null, ToolParameterType.Number, null),
            new("b", Description: null, ToolParameterType.Number, null)
        ]));
        wrappers[0].Tool.Should().BeNull();
        wrappers[0].Method.Should().BeSameAs(typeof(AddTool).GetMethod(nameof(AddTool.Add))!);
    }

    [Fact]
    public async Task CheckCallTool() {
        var parameters = wrappers[0].ParseParameters(JsonSerializer.SerializeToNode(new { a = 1, b = 2 })).Unwrap();
        var result = await ThrowIfError(wrappers[0].Call(parameters));

        result.Should().Be("1 + 2 = 3");
    }
}