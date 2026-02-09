using System.Text.Json;
using System.Text.Json.Nodes;
using JetBrains.Annotations;
using RZ.Foundation.AI;

namespace UnitTests;

[UsedImplicitly]
public sealed class ToolCallTests
{
    [Test]
    public async ValueTask ParseParametersWithNullValues() {
        // given
        var tools = ToolWrapper.FromType(typeof(TestTools));
        var theTool = tools.Single();

        // when parse parameters with null values
        var p = theTool.ParseParameters(JsonNode.Parse("""{"name": null, "age": null}"""));

        // then
        await Assert.That(Success(p, out var parameters)).IsTrue();
        await Assert.That(parameters).IsNotNull();
        await Assert.That(parameters.Length).IsEqualTo(2);
        await Assert.That(parameters[0]).IsNull();
        await Assert.That(parameters[1]).IsNull();
    }

    [Test]
    public async ValueTask ParseParametersWithNonNullValues() {
        // given
        var tools = ToolWrapper.FromType(typeof(TestTools));
        var theTool = tools.Single();

        // when parse parameters with null values
        var p = theTool.ParseParameters(JsonNode.Parse("""{"name": "John", "age": 123}"""));

        // then
        await Assert.That(Success(p, out var parameters)).IsTrue();
        await Assert.That(parameters).IsNotNull();
        await Assert.That(parameters.Length).IsEqualTo(2);
        await Assert.That(parameters[0]).IsEqualTo("John");
        await Assert.That(parameters[1]).IsEqualTo(123);
    }

    static class TestTools
    {
        [AiToolName]
        public static ValueTask<string> Record(string? name, int? age)
            => new($"Recorded: Name={name}, Age={age}");
    }

    [Test]
    public async ValueTask ScanClassForToolInstance() {
        var instance = new ComputingUnit(42);
        var tools = ToolWrapper.From(instance).OrderBy(x => x.Definition.Name).ToArray();

        // then, definitions are correct
        await Assert.That(tools.Count).IsEqualTo(2);
        await Assert.That(tools[0].Definition).IsEquivalentTo(new ToolDefinition(nameof(ComputingUnit.Compute), Description: null, [
            new("x", Description: null, ToolParameterType.Number, null)
        ]));
        await Assert.That(tools[0].Tool).IsEqualTo(instance);
        await Assert.That(tools[0].Method).IsEqualTo(typeof(ComputingUnit).GetMethod(nameof(ComputingUnit.Compute))!);

        await Assert.That(tools[1].Definition).IsEquivalentTo(new ToolDefinition(nameof(ComputingUnit.ComputeNull), Description: null, []));
        await Assert.That(tools[1].Tool).IsEqualTo(instance);
        await Assert.That(tools[1].Method).IsEqualTo(typeof(ComputingUnit).GetMethod(nameof(ComputingUnit.ComputeNull))!);

        // and when parse parameters
        var parameters = tools[0].ParseParameters(JsonSerializer.SerializeToNode(new { x = 1 })).Unwrap();
        await Assert.That(parameters.Length).IsEqualTo(1);
        await Assert.That(parameters[0]).IsEqualTo(1);

        // and when call the method
        var result = await ThrowIfError(tools[0].Call(parameters));
        await Assert.That(result).IsEqualTo("Compute value of 1 = 43");

        await Assert.That(Fail(await tools[1].Call([]), out var e)).IsTrue();
        await Assert.That(e?.Code).IsEqualTo(StandardErrorCodes.ValidationFailed);
        Console.WriteLine(e?.ToString());
    }

    sealed class ComputingUnit(int secret)
    {
        [AiToolName]
        public string Compute(int x)
            => $"Compute value of {x} = {x + secret}";

        [AiToolName]
        public string? ComputeNull() => null;
    }
}

[UsedImplicitly]
public sealed class StaticToolTests
{
    readonly IReadOnlyList<ToolWrapper> wrappers = ToolWrapper.FromType(typeof(AddTool));

    static class AddTool
    {
        [AiToolName("add_numbers")]
        public static string Add(int a, int b)
            => $"{a} + {b} = {a + b}";
    }

    [Test]
    public async ValueTask CheckDefinition() {
        await Assert.That(wrappers.Count).IsEqualTo(1);
        await Assert.That(wrappers[0].Definition).IsEquivalentTo(new ToolDefinition("add_numbers", Description: null, [
            new("a", Description: null, ToolParameterType.Number, null),
            new("b", Description: null, ToolParameterType.Number, null)
        ]));
        await Assert.That(wrappers[0].Tool).IsNull();
        await Assert.That(wrappers[0].Method).IsEqualTo(typeof(AddTool).GetMethod(nameof(AddTool.Add))!);
    }

    [Test]
    public async ValueTask CheckCallTool() {
        var parameters = wrappers[0].ParseParameters(JsonSerializer.SerializeToNode(new { a = 1, b = 2 })).Unwrap();
        var result = await ThrowIfError(wrappers[0].Call(parameters));

        await Assert.That(result).IsEqualTo("1 + 2 = 3");
    }
}
