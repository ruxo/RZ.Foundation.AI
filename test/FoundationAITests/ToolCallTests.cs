using System.Text.Json;
using JetBrains.Annotations;
using RZ.Foundation.AI;

namespace FoundationAITests;

[UsedImplicitly(ImplicitUseTargetFlags.Members)]
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