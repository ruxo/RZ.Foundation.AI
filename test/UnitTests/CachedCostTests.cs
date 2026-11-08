using System.Text.Json;
using OpenAI.Chat;
using RZ.Foundation.AI;

namespace UnitTests;

public sealed class CachedCostTests
{
    [Test]
    public async ValueTask ReportedCacheUsageChangesUsdWithoutChangingInputTokens() {
        foreach (var cached in new[] { 0, 8000, 10000 }) {
            var cost = OpenAi.CalcCost(OpenAi.GetRate(OpenAi.GPT_54), Usage(10000, cached));
            await Assert.That(cost.Input).IsEqualTo(new TokenCost(10000, ((10000 - cached) * 2.5m + cached * 0.25m) / 1000000m));
            await Assert.That(cost.Output).IsEqualTo(new TokenCost(200, 0.003m));
            await Assert.That(cost.Other).IsEqualTo(new TokenCost(300, 0.0045m));
            await Assert.That(cost.Total.Tokens).IsEqualTo(10500L);
        }
    }

    [Test]
    public async ValueTask CachedRatesUseTheTierSelectedByTotalInput() {
        foreach (var (model, ordinary, cachedRate) in new[] { (OpenAi.GPT_54, 2.5m, 0.25m), (OpenAi.GPT_55, 5m, 0.5m) }) {
            foreach (var input in new[] { 271999, 272000, 272001 }) {
                foreach (var cached in new[] { 8000, input }) {
                    var cost = OpenAi.CalcCost(OpenAi.GetRate(model), Usage(input, cached));
                    var multiplier = input > 272000 ? 2m : 1m;
                    await Assert.That(cost.Input).IsEqualTo(new TokenCost(input, ((input - cached) * ordinary + cached * cachedRate) * multiplier / 1000000m));
                }
            }
        }
    }

    [Test]
    public async ValueTask ExistingModelsHaveExplicitCachedPrices() {
        foreach (var (model, price) in new[] {
            (OpenAi.GPT_5_NANO, 0.005m), (OpenAi.GPT_41_NANO, 0.025m), (OpenAi.GPT_4O_MINI, 0.075m),
            (OpenAi.GPT_54_NANO, 0.02m), (OpenAi.GPT_41_MINI, 0.10m), (OpenAi.GPT_5_MINI, 0.025m),
            (OpenAi.GPT_54_MINI, 0.075m), (OpenAi.GPT_41, 0.50m), (OpenAi.GPT_5, 0.125m),
            (OpenAi.GPT_51, 0.125m), (OpenAi.GPT_4O, 1.25m), (OpenAi.GPT_54, 0.25m), (OpenAi.GPT_55, 0.50m)
        }) {
            await Assert.That(OpenAi.CalcCost(OpenAi.GetRate(model), Usage(10000, 10000)).Input)
                        .IsEqualTo(new TokenCost(10000, price * 0.01m));
        }
    }

    [Test]
    public async ValueTask CustomRatesFallbackAndZeroPricesRetainUsage() {
        var usage = Usage(10000, 8000);
        await Assert.That(OpenAi.CalcCost(CostStructure.Simple(2, 10), usage).Input).IsEqualTo(new TokenCost(10000, 0.02m));
        await Assert.That(OpenAi.CalcCost(CostStructure.Simple(2, 10, cachedInput: 0.2m), usage).Input).IsEqualTo(new TokenCost(10000, 0.0056m));
        await Assert.That(OpenAi.CalcCost(CostStructure.Simple(2, 10, cachedInput: 0), usage).Input).IsEqualTo(new TokenCost(10000, 0.004m));
        await Assert.That(OpenAi.CalcCost(OpenAi.GetRate("unpriced-model"), usage).Input).IsEqualTo(new TokenCost(10000, 0));
    }

    [Test]
    [Arguments("", 20000)]
    [Arguments(", \"CachedInputUsdPerMillion\": 0", 4000)]
    [Arguments(", \"CachedInputUsdPerMillion\": 0.2", 5600)]
    public async ValueTask JsonCachedRatePreservesOmissionAndExplicitPrices(string cachedPrice, int expectedMillionths) {
        var json = """{"InputUsdPerMillion":2,"OutputUsdPerMillion":10,"OtherUsdPerMillion":3""" + cachedPrice + "}";
        var rate = JsonSerializer.Deserialize<ChatRate>(json);
        var cost = OpenAi.CalcCost(new FlatCostStructure(rate), Usage(10000, 8000));
        await Assert.That(cost.Input).IsEqualTo(new TokenCost(10000, expectedMillionths / 1000000m));
    }

    [Test]
    public async ValueTask ExplicitZeroCachedRateSurvivesJsonRoundTripWithDefaultValuesOmitted() {
        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web) {
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingDefault
        };
        var rate = new ChatRate(2, 10, 3, 0);
        var json = JsonSerializer.Serialize(rate, options);
        await Assert.That(JsonSerializer.Deserialize<ChatRate>(json, options)).IsEqualTo(rate);
    }

    [Test]
    public async ValueTask NegativeValuesAreRejectedByConstructorsAndDeserialization() {
        foreach (var create in new Action[] {
            () => _ = new TokenCost(-1, 0), () => _ = new TokenCost(0, -1),
            () => _ = new ChatRate(-1, 0, 0, 0), () => _ = new ChatRate(0, -1, 0, 0),
            () => _ = new ChatRate(0, 0, -1, 0), () => _ = new ChatRate(0, 0, 0, -1),
            () => _ = new TieredCostStructure(-1, default, default),
            () => _ = JsonSerializer.Deserialize<TokenCost>("""{"Tokens":-1,"Usd":0}"""),
            () => _ = JsonSerializer.Deserialize<ChatRate>("""{"InputUsdPerMillion":0,"OutputUsdPerMillion":0,"OtherUsdPerMillion":0,"CachedInputUsdPerMillion":-1}""")
        }) {
            await Assert.That(Rejects<ArgumentOutOfRangeException>(create)).IsTrue();
        }
        await Assert.That(typeof(TokenCost).GetProperties().All(p => p.SetMethod is null)).IsTrue();
        await Assert.That(typeof(ChatRate).GetProperties().All(p => p.SetMethod is null)).IsTrue();
        await Assert.That(default(TokenCost)).IsEqualTo(TokenCost.Zero);
        var rate = new ChatRate(2, 10, 3, 0.2m);
        await Assert.That(JsonSerializer.Deserialize<ChatRate>(JsonSerializer.Serialize(rate))).IsEqualTo(rate);
    }

    [Test]
    public async ValueTask InvalidUsageIsRejectedAtTheCalculatorBoundary() {
        var rate = CostStructure.Simple(0, 0);
        foreach (var calculate in new Action[] {
            () => _ = LLM.CalcCost(rate, -1, 0, 0), () => _ = LLM.CalcCost(rate, 0, -1, 0),
            () => _ = LLM.CalcCost(rate, 0, 0, -1), () => _ = LLM.CalcCost(rate, 1, 0, 0, -1),
            () => _ = LLM.CalcCost(rate, 1, 0, 0, 2)
        }) {
            await Assert.That(Rejects<ArgumentOutOfRangeException>(calculate)).IsTrue();
        }
    }

    [Test]
    [Arguments(nameof(TokenCost.Tokens))]
    [Arguments(nameof(TokenCost.Usd))]
    public ValueTask TokenCostRejectsMissingJsonField(string field)
        => AssertMissingFieldRejected(new TokenCost(10, 0.02m), field);

    [Test]
    [Arguments(nameof(ChatCost.Input))]
    [Arguments(nameof(ChatCost.Output))]
    [Arguments(nameof(ChatCost.Other))]
    public ValueTask ChatCostRejectsMissingJsonBucket(string field)
        => AssertMissingFieldRejected(ChatCost.Zero, field);

    [Test]
    [Arguments(nameof(ChatRate.InputUsdPerMillion))]
    [Arguments(nameof(ChatRate.OutputUsdPerMillion))]
    [Arguments(nameof(ChatRate.OtherUsdPerMillion))]
    public ValueTask ChatRateRejectsMissingJsonRate(string field)
        => AssertMissingFieldRejected(new ChatRate(2, 10, 3, 0.2m), field);

    [Test]
    public async ValueTask RequiredZeroFieldsSurviveDefaultValueOmission() {
        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web) {
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingDefault
        };
        var cost = ChatCost.Zero;
        var rate = default(ChatRate);
        await Assert.That(JsonSerializer.Deserialize<ChatCost>(JsonSerializer.Serialize(cost, options), options)).IsEqualTo(cost);
        await Assert.That(JsonSerializer.Deserialize<ChatRate>(JsonSerializer.Serialize(rate, options), options)).IsEqualTo(rate);
    }

    static async ValueTask AssertMissingFieldRejected<T>(T value, string field) {
        var json = JsonSerializer.SerializeToNode(value)!.AsObject();
        json.Remove(field);
        await Assert.That(Rejects<JsonException>(() => JsonSerializer.Deserialize<T>(json))).IsTrue();
    }

    static ChatTokenUsage Usage(int input, int cached)
        => OpenAIChatModelFactory.ChatTokenUsage(outputTokenCount: 500, inputTokenCount: input, totalTokenCount: input + 500,
            outputTokenDetails: OpenAIChatModelFactory.ChatOutputTokenUsageDetails(reasoningTokenCount: 300),
            inputTokenDetails: OpenAIChatModelFactory.ChatInputTokenUsageDetails(audioTokenCount: 0, cachedTokenCount: cached));

    static bool Rejects<TException>(Action action) where TException : Exception {
        try { action(); return false; }
        catch (TException) { return true; }
    }
}
