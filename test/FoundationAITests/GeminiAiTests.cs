using FluentAssertions;
using RZ.Foundation.AI;
// ReSharper disable InconsistentNaming

namespace FoundationAITests;

public class GeminiAiTests
{
    const string GeminiAiKey = "(API key)";
    static readonly HttpClient http = new();

    [Fact(Explicit = true)]
    public async Task SimpleChat() {
        var chat = new GeminiAi(GeminiAiKey, http).CreateModel(GeminiAi.gemini_15_flash_8b);

        var (response, cost) = await chat([new ChatMessage.Content(ChatRole.User, "Hello")]);

        TestContext.Current.TestOutputHelper!.WriteLine($"Cost: {cost}");
        cost.Input.Should().BeGreaterThan(0m);
        cost.Output.Should().BeGreaterThan(0m);
        response.Count.Should().Be(1, $"but {response}");
        response[0].Cost.Should().Be(cost);

        var content = (ChatMessage.Content)response[0].Message;
        content.Role.Should().Be(ChatRole.Agent);
    }
}