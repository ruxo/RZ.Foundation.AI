using System.Net;
using System.Text;
using System.Text.Json;
using OpenAI.Chat;
using RZ.Foundation.AI;

namespace UnitTests;

public sealed class CostTests
{
    [Test]
    public async ValueTask SnapshotsAddTokensAndUsdWithoutRepricing() {
        var a = new ChatCost(new(0, 1.25m), new(500, 0), new(7, 0.03m));
        var b = new ChatCost(new(20, 0), new(0, 0.75m), new(3, 0.02m));
        var sum = a + b;
        await Assert.That(sum).IsEqualTo(new ChatCost(new(20, 1.25m), new(500, 0.75m), new(10, 0.05m)));
        await Assert.That(sum.Total).IsEqualTo(new TokenCost(530, 2.05m));
        await Assert.That(a + ChatCost.Zero).IsEqualTo(a);
        await Assert.That(JsonSerializer.Deserialize<ChatCost>(JsonSerializer.Serialize(sum))).IsEqualTo(sum);
        await Assert.That(JsonDocument.Parse(JsonSerializer.Serialize(sum)).RootElement.TryGetProperty("Total", out _)).IsFalse();
    }

    [Test]
    public async ValueTask FlatAndThoughtPricesProduceUsdSnapshots() {
        var flat = LLM.CalcCost(CostStructure.Simple(2.5m, 15m), 1000, 200, 0);
        await Assert.That(flat).IsEqualTo(new ChatCost(new(1000, 0.0025m), new(200, 0.003m), TokenCost.Zero));
        var thought = LLM.CalcCost(CostStructure.WithThought(0.15m, 0.6m, 3.5m), 1000, 200, 300);
        await Assert.That(thought).IsEqualTo(new ChatCost(new(1000, 0.00015m), new(200, 0.00012m), new(300, 0.00105m)));
        await Assert.That(thought.Total).IsEqualTo(new TokenCost(1500, 0.00132m));
    }

    [Test]
    public async ValueTask BothLongContextModelsUseInputForEveryRateAndExcludeExactThreshold() {
        foreach (var (model, inputRate, outputRate) in new[] { (OpenAi.GPT_54, 2.5m, 15m), (OpenAi.GPT_55, 5m, 30m) }) {
            foreach (var input in new[] { 271999, 272000, 272001, 300000 }) {
                // Even a large output must not select the tier. Reasoning is part of total output.
                const int output = 300000;
                var usage = OpenAIChatModelFactory.ChatTokenUsage(outputTokenCount: output, inputTokenCount: input,
                    totalTokenCount: input + output, outputTokenDetails: OpenAIChatModelFactory.ChatOutputTokenUsageDetails(reasoningTokenCount: 300));
                var cost = OpenAi.CalcCost(OpenAi.GetRate(model), usage);
                var high = input > 272000;
                await Assert.That(cost.Input).IsEqualTo(new TokenCost(input, input * inputRate * (high ? 2m : 1m) / 1000000m));
                await Assert.That(cost.Output).IsEqualTo(new TokenCost(output - 300, (output - 300) * outputRate * (high ? 1.5m : 1m) / 1000000m));
                await Assert.That(cost.Other).IsEqualTo(new TokenCost(300, 300 * outputRate * (high ? 1.5m : 1m) / 1000000m));
                await Assert.That(cost.Total.Tokens).IsEqualTo((long)input + output);
            }
            // RA-315 reproduction: small output still uses the higher rate with long input.
            var longCost = LLM.CalcCost(OpenAi.GetRate(model), 300000, 1000, 0);
            await Assert.That(longCost.Output.Usd).IsEqualTo(1000 * outputRate * 1.5m / 1000000m);
        }
    }

    [Test]
    public async ValueTask OpenAiMissingPricesKeepUsageAndAllowExplicitRates() {
        var usage = OpenAIChatModelFactory.ChatTokenUsage(outputTokenCount: 500, inputTokenCount: 1000,
            totalTokenCount: 1500, outputTokenDetails: OpenAIChatModelFactory.ChatOutputTokenUsageDetails(reasoningTokenCount: 300));
        var cost = OpenAi.CalcCost(OpenAi.GetRate("unpriced-model"), usage);
        await Assert.That(cost).IsEqualTo(new ChatCost(new(1000, 0), new(200, 0), new(300, 0)));
        var priced = OpenAi.CalcCost(CostStructure.Simple(2m, 10m), usage);
        await Assert.That(priced).IsEqualTo(new ChatCost(new(1000, 0.002m), new(200, 0.002m), new(300, 0.003m)));
        var noDetails = OpenAIChatModelFactory.ChatTokenUsage(outputTokenCount: 200, inputTokenCount: 1000, totalTokenCount: 1200);
        await Assert.That(OpenAi.CalcCost(CostStructure.Simple(2m, 10m), noDetails).Other).IsEqualTo(TokenCost.Zero);
    }

    [Test]
    public async ValueTask TokenCountsAreNotLimitedToInt32() {
        var cost = LLM.CalcCost(CostStructure.Simple(1, 2, 3), 3000000000L, 4000000000L, 5000000000L);
        await Assert.That(cost.Total).IsEqualTo(new TokenCost(12000000000L, 26000m));
    }

    [Test]
    [Arguments(false)]
    [Arguments(true)]
    public async ValueTask GeminiAppliesInputPricingAndAttributesCostToFirstEntry(bool discounted) {
        using var http = new HttpClient(new GeminiResponse());
        var rate = CostStructure.WithThought(0.15m, 0.6m, 3.5m, cachedInput: discounted ? 0.015m : null);
        var chat = new GeminiAi("offline-test-key", http).CreateModel("offline-model", rate);
        var result = await chat([new RZ.Foundation.AI.ChatMessage.Content(ChatRole.User, "Hello")]);
        await Assert.That(Success(result, out var response)).IsTrue();
        var expected = new ChatCost(new(1000, discounted ? 0.000042m : 0.00015m), new(200, 0.00012m), new(300, 0.00105m));
        await Assert.That(response.Cost).IsEqualTo(expected);
        await Assert.That(response.Chat.Count).IsEqualTo(2);
        await Assert.That(response.Chat[0].Cost).IsEqualTo(expected);
        await Assert.That(response.Chat[1].Cost).IsEqualTo(ChatCost.Zero);
    }

    sealed class GeminiResponse : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) {
                Content = new StringContent("""
                    {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"},{"text":" again"}]},"finishReason":"STOP"}],
                     "usageMetadata":{"promptTokenCount":1000,"cachedContentTokenCount":800,"candidatesTokenCount":200,"thoughtsTokenCount":300,"totalTokenCount":1500}}
                    """, Encoding.UTF8, "application/json")
            });
    }
}
