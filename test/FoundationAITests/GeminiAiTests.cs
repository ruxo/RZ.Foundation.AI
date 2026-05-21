using JetBrains.Annotations;
using RZ.Foundation.AI;
// ReSharper disable InconsistentNaming

namespace FoundationAITests;

[UsedImplicitly(ImplicitUseTargetFlags.Members)]
public class GeminiAiTests
{
    const string GeminiAiKey = "(API key from https://aistudio.google.com/api-keys)";
    const bool RunTests = false;

    static readonly HttpClient http = new();

    [Test]
    public async ValueTask SimpleChat() {
        Skip.When(!RunTests, "Skipped because `RunTests=false`");

        var chat = new GeminiAi(GeminiAiKey, http).CreateModel(GeminiAi.GEMINI_20_FLASH_LITE);

        var (response, cost) = await ThrowIfError(chat([new ChatMessage.Content(ChatRole.User, "Hello")]));

        Console.WriteLine($"Cost: {cost}");

        await Assert.That(cost.Input).IsGreaterThan(0m);
        await Assert.That(cost.Output).IsGreaterThan(0m);
        await Assert.That(response.Count).IsEqualTo(1).Because($"but {response}");
        await Assert.That(response[0].Cost).IsEqualTo(cost);

        var content = (ChatMessage.Content)response[0].Message;
        await Assert.That(content.Role).IsEqualTo(ChatRole.Agent);
    }
}