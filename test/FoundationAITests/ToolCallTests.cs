using System.Text.Json;
using FluentAssertions;
using RZ.Foundation.AI;

namespace FoundationAITests;

public class ToolCallTests
{
    [Fact]
    public async Task ScanClassForToolInstance() {
        var instance = new ComputingUnit(42);
        var tools = ToolWrapper.From(instance);

        // then, definitions are correct
        tools.Count.Should().Be(1);
        tools[0].Definition.Should().BeEquivalentTo(new ToolDefinition("Compute", Description: null, [
            new("x", Description: null, ToolParameterType.Number, null)
        ]));
        tools[0].Tool.Should().Be(instance);
        tools[0].Method.Should().BeSameAs(typeof(ComputingUnit).GetMethod(nameof(ComputingUnit.Compute))!);

        // and when parse parameters
        var parameters = tools[0].ParseParameters(JsonSerializer.SerializeToNode(new { x = 1 }));
        parameters.Length.Should().Be(1);
        parameters[0].Should().Be(1);

        // and when call the method
        var result = await tools[0].Call(parameters);
        result.Should().Be("Compute value of 1 = 43");
    }

    sealed class ComputingUnit(int secret)
    {
        [AiToolName]
        public string Compute(int x)
            => $"Compute value of {x} = {x + secret}";
    }
}

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
        var parameters = wrappers[0].ParseParameters(JsonSerializer.SerializeToNode(new { a = 1, b = 2 }));
        var result = await wrappers[0].Call(parameters);

        result.Should().Be("1 + 2 = 3");
    }
}