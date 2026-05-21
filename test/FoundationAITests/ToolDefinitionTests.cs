using System.ComponentModel;
using System.Text.Json;
using JetBrains.Annotations;
using RZ.Foundation.AI;
using RZ.Foundation.Extensions;
using TD = RZ.Foundation.AI.ToolDefinition;

namespace UnitTests;

public sealed class LangChainAgentTests
{
    #region Tests ToolDefinition creation

    [Test]
    [TUnit.Core.DisplayName("Generate a tool definition with a method (Simple)")]
    public ValueTask GenerateToolDef() => TestToolDef("GetGreeting", new TD("test", "Get hello world", []));

    [Test]
    [TUnit.Core.DisplayName("Generate a tool definition with a method with parameters")]
    public ValueTask FromASingleParameter() => TestToolDef("GreetWithName", new TD("test", null, [new("name", null, ToolParameterType.String, null)]));

    [Test]
    [TUnit.Core.DisplayName("Generate a tool definition with a method with optional parameters")]
    public ValueTask FromOptionalParameters() => TestToolDef("Remember", new TD("test", "Remember user", [
        new("name", null, ToolParameterType.String, Some((object)"Someone")),
        new("id", "User ID", ToolParameterType.Number, None)
    ]));

    static ValueTask TestToolDef(string methodName, TD expected) {
        var method = typeof(TestTool).GetMethod(methodName) ?? throw new Exception();
        var result = TD.From("test", method);
        return TestToolDef(result, expected);
    }

    static async ValueTask TestToolDef(TD result, TD expected) {
        await Assert.That(result.Name).IsEqualTo(expected.Name);
        await Assert.That(result.Description).IsEqualTo(expected.Description);

        foreach (var p in expected.Parameters){
            var matched = result.Parameters.TryFirst(x => x.Name == p.Name).ToNullable();
            await Assert.That(matched).IsNotNull().Because($"but {p.Name} is missing from {result}");

            await Assert.That(matched.Value.Type).IsEqualTo(p.Type);
            await Assert.That(matched.Value.Description).IsEqualTo(p.Description);
            await Assert.That(matched.Value.DefaultValue).IsEqualTo(p.DefaultValue)
                        .Because($"but property \"{p.Name}\" expected [{p.DefaultValue}] has a different result's value [{matched.Value.DefaultValue}]");
        }
    }

    #endregion

    #region Test ToSchema method

    [Test]
    [TUnit.Core.DisplayName("Transform ToolDefinition to JSON schema with a mandatory parameters method")]
    public async ValueTask TransformToSchemaWithMandatory() {
        var source = new TD("test", null, [new("name", null, ToolParameterType.String, null)]);

        var result = source.ToJsonSchema();

        var expected = JsonSerializer.SerializeToNode(new {
            name = "test",
            description = (string?)null,
            parameters = new {
                type = "object",
                properties = new {
                    name = new { type = "string", description = (string?)null }
                },
                required = new[] { "name" }
            }
        })!;
        await Assert.That(result.ToJsonString()).IsEqualTo(expected.ToJsonString());
    }

    [Test]
    [TUnit.Core.DisplayName("Transform ToolDefinition to JSON schema with a optional parameters method")]
    public async ValueTask TransformToSchema() {
        var source = new TD("test", "Remember user", [
            new("name", null, ToolParameterType.String, Some((object)"Someone")),
            new("id", "User ID", ToolParameterType.Number, None)
        ]);

        var result = source.ToJsonSchema();

        var expected = JsonSerializer.SerializeToNode(new {
            name = "test",
            description = (string?)"Remember user",
            parameters = new {
                type = "object",
                properties = new {
                    name = new { type = "string", description = (string?)null },
                    id = new { type = "number", description = (string?)"User ID" }
                },
                required = Array.Empty<string>()
            }
        })!;
        await Assert.That(result.ToJsonString()).IsEqualTo(expected.ToJsonString());
    }

    #endregion

    [UsedImplicitly]
    sealed record Person(int Id, string Name);

    [UsedImplicitly(ImplicitUseTargetFlags.Members)]
    sealed class TestTool
    {
        [AiToolName("get_greeting")]
        [Description("Get hello world")]
        public string GetGreeting() => "Hello";

        [AiToolName("greet_with_name")]
        public string GreetWithName(string name) => $"Hello {name}";

        [AiToolName("remember")]
        [Description("Remember user")]
        public string Remember(string? name = "Someone", [Description("User ID")] int? id = null) => "Remembered";

        [AiToolName("get_greeting_async")]
        public Task<string> GetGreetingAsync() => Task.FromResult("Hello");

        [AiToolName("get_person")]
        public Task<Person> GetPerson(int id, string? name = null) => Task.FromResult(new Person(id, name ?? "John"));

        public string Dummy() => throw new NotSupportedException();
    }
}