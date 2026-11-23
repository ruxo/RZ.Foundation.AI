using OpenAI.Chat;
using RZ.Foundation.AI;

namespace UnitTests;

public sealed class OpenAiCacheWriteTests
{
    [Test]
    [Arguments(10000, 0, 0)]
    [Arguments(10000, 0, 8000)]
    [Arguments(10000, 0, 10000)]
    [Arguments(10000, 2000, 4000)]
    [Arguments(272000, 10000, 8000)]
    [Arguments(272001, 10000, 8000)]
    public async ValueTask RawCacheWritesArePricedWithoutCountingTokensTwice(int input, int cached, int writes) {
        var body = BinaryData.FromObjectAsJson(new { usage = new { prompt_tokens_details = new { cache_write_tokens = writes } } });
        var cost = OpenAi.CalcCost(OpenAi.GetRate(OpenAi.GPT_6_LUNA), Usage(input, cached), body);
        var high = input > 272000;
        var ordinaryRate = high ? 0.2m : 0.1m;
        var cachedRate = high ? 0.02m : 0.01m;
        var outputRate = high ? 0.75m : 0.5m;
        var expected = ((input - cached - writes) * ordinaryRate + cached * cachedRate + writes * ordinaryRate * 1.25m) / 1000000m;
        await Assert.That(cost.Input).IsEqualTo(new TokenCost(input, expected));
        await Assert.That(cost.Output).IsEqualTo(new TokenCost(200, 200 * outputRate / 1000000m));
        await Assert.That(cost.Other).IsEqualTo(new TokenCost(300, 300 * outputRate / 1000000m));
        await Assert.That(cost.Total.Tokens).IsEqualTo(input + 500L);
    }

    [Test]
    [Arguments("{}")]
    [Arguments("{\"usage\":{}}")]
    [Arguments("{\"usage\":{\"prompt_tokens_details\":null}}")]
    [Arguments("{\"usage\":{\"prompt_tokens_details\":{}}}")]
    [Arguments("{\"usage\":{\"prompt_tokens_details\":{\"cache_write_tokens\":null}}}")]
    public async ValueTask LegacyUsageWithoutWriteMetadataKeepsItsExistingPrice(string json) {
        var usage = Usage(10000, 2000);
        await Assert.That(OpenAi.CalcCost(OpenAi.GetRate(OpenAi.GPT_6_LUNA), usage, BinaryData.FromString(json)))
                    .IsEqualTo(OpenAi.CalcCost(OpenAi.GetRate(OpenAi.GPT_6_LUNA), usage));
    }

    [Test]
    public async ValueTask ExplicitZeroRatesStillPreserveAllUsage() {
        var body = BinaryData.FromString("{\"usage\":{\"prompt_tokens_details\":{\"cache_write_tokens\":4000}}}");
        var cost = OpenAi.CalcCost(CostStructure.Simple(0, 0), Usage(10000, 2000), body);
        await Assert.That(cost.Input).IsEqualTo(new TokenCost(10000, 0));
        await Assert.That(cost.Total).IsEqualTo(new TokenCost(10500, 0));
    }

    [Test]
    [Arguments(-1)]
    [Arguments(9000)]
    public async ValueTask InvalidWriteSubsetsAreRejected(int writes) {
        var body = BinaryData.FromObjectAsJson(new { usage = new { prompt_tokens_details = new { cache_write_tokens = writes } } });
        var rejected = false;
        try { OpenAi.CalcCost(OpenAi.GetRate(OpenAi.GPT_6_LUNA), Usage(10000, 2000), body); }
        catch (ArgumentOutOfRangeException) { rejected = true; }
        await Assert.That(rejected).IsTrue();
    }

    static ChatTokenUsage Usage(int input, int cached)
        => OpenAIChatModelFactory.ChatTokenUsage(outputTokenCount: 500, inputTokenCount: input, totalTokenCount: input + 500,
            outputTokenDetails: OpenAIChatModelFactory.ChatOutputTokenUsageDetails(reasoningTokenCount: 300),
            inputTokenDetails: OpenAIChatModelFactory.ChatInputTokenUsageDetails(audioTokenCount: 0, cachedTokenCount: cached));
}
