using FluentAssertions;
using RZ.Foundation.AI;
// ReSharper disable InconsistentNaming

namespace FoundationAITests;

public class GeminiAiTests
{
    const string GeminiAiKey = "(API key from https://aistudio.google.com/api-keys)";
    const bool RunTests = false;

    static readonly HttpClient http = new();

    [Fact]
    public async Task SimpleChat() {
        if (!RunTests) Assert.Skip("Skipping test");

        var chat = new GeminiAi(GeminiAiKey, http).CreateModel(GeminiAi.GEMINI_20_FLASH_LITE);

        var (response, cost) = await ThrowIfError(chat([new ChatMessage.Content(ChatRole.User, "Hello")]));

        TestContext.Current.TestOutputHelper!.WriteLine($"Cost: {cost}");
        cost.Input.Should().BeGreaterThan(0m);
        cost.Output.Should().BeGreaterThan(0m);
        response.Count.Should().Be(1, $"but {response}");
        response[0].Cost.Should().Be(cost);

        var content = (ChatMessage.Content)response[0].Message;
        content.Role.Should().Be(ChatRole.Agent);
    }
}