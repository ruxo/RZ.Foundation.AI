using System.ComponentModel;
using System.Text.Json;
using FluentAssertions;
using JetBrains.Annotations;
using RZ.Foundation.AI;
using TD = RZ.Foundation.AI.ToolDefinition;

namespace UnitTests;

public sealed class LangChainAgentTests
{
    #region Tests ToolDefinition creation

    [Fact(DisplayName = "Generate a tool definition with a method (Simple)")]
    public void GenerateToolDef() => TestToolDef("GetGreeting", new TD("test", "Get hello world", []));

    [Fact(DisplayName = "Generate a tool definition with a method with parameters")]
    public void FromASingleParameter() => TestToolDef("GreetWithName", new TD("test", null, [new("name", null, ToolParameterType.String, null)]));

    [Fact(DisplayName = "Generate a tool definition with a method with optional parameters")]
    public void FromOptionalParameters() => TestToolDef("Remember", new TD("test", "Remember user", [
        new("name", null, ToolParameterType.String, Some((object)"Someone")),
        new("id", "User ID", ToolParameterType.Number, None)
    ]));

    static void TestToolDef(string methodName, TD expected) {
        var method = typeof(TestTool).GetMethod(methodName) ?? throw new Exception();
        var result = TD.From("test", method);
        TestToolDef(result, expected);
    }

    static void TestToolDef(TD result, TD expected) {
        result.Name.Should().Be(expected.Name);
        result.Description.Should().Be(expected.Description);
        foreach (var p in expected.Parameters){
            var matched = result.Parameters.FirstOrDefault(x => x.Name == p.Name);
            matched.Should().NotBeNull($"but {p.Name} is missing from {result}");

            matched.Type.Should().Be(p.Type);
            matched.Description.Should().Be(p.Description);
            (matched.DefaultValue == p.DefaultValue).Should().BeTrue($"but property \"{p.Name}\" expected [{p.DefaultValue}] has a different result's value [{matched.DefaultValue}]");
        }
    }

    #endregion

    #region Test ToSchema method

    [Fact(DisplayName = "Transform ToolDefinition to JSON schema with a mandatory parameters method")]
    public void TransformToSchemaWithMandatory() {
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
        result.ToJsonString().Should().BeEquivalentTo(expected.ToJsonString());
    }

    [Fact(DisplayName = "Transform ToolDefinition to JSON schema with a optional parameters method")]
    public void TransformToSchema() {
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
        result.ToJsonString().Should().BeEquivalentTo(expected.ToJsonString());
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