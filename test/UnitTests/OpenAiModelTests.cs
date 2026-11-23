using OpenAI.Chat;
using RZ.Foundation.AI;

namespace UnitTests;

public sealed class OpenAiModelTests
{
    [Test]
    public async ValueTask LunaPricesAllBucketsAtTheLongContextBoundary() {
        foreach (var input in new long[] { 272000, 272001 }) {
            var cost = LLM.CalcCost(OpenAi.GetRate(OpenAi.GPT_6_LUNA), input, 200, 300, 10000);
            var high = input > 272000;
            await Assert.That(cost.Input).IsEqualTo(new TokenCost(input, ((input - 10000) * (high ? 0.2m : 0.1m) + 10000 * (high ? 0.02m : 0.01m)) / 1000000m));
            await Assert.That(cost.Output).IsEqualTo(new TokenCost(200, 200 * (high ? 0.75m : 0.5m) / 1000000m));
            await Assert.That(cost.Other).IsEqualTo(new TokenCost(300, 300 * (high ? 0.75m : 0.5m) / 1000000m));
        }
    }

    [Test]
    public async ValueTask RetiredModelsAreAbsentFromTheSelectableCatalogue() {
        foreach (var model in new[] { "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1", "gpt-51", "gpt-5.4-nano" })
            await Assert.That(OpenAi.SupportedModels.Contains(model)).IsFalse();
        foreach (var model in new[] { "gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6", "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.2", "gpt-6.2-astra", "gpt-6.2-sol", "chat-latest" })
            await Assert.That(OpenAi.SupportedModels.Contains(model)).IsTrue().Because(model);
    }

    [Test]
    public async ValueTask ModelsAreOrderedByUncachedInputAndOutputPrice() {
        var prices = OpenAi.SupportedModels.Select(model => {
            var rate = OpenAi.GetRate(model).RateFor(0);
            return rate.InputUsdPerMillion + rate.OutputUsdPerMillion;
        }).ToArray();
        await Assert.That(prices.SequenceEqual(prices.Order())).IsTrue();
    }

    [Test]
    public async ValueTask LunaAndSolDefaultToNonReasoningToolsAndRefuseExplicitReasoning() {
#pragma warning disable OPENAI001
        foreach (var model in new[] { OpenAi.GPT_6_LUNA, OpenAi.GPT_6_SOL }) {
            var options = WithTool();
            await Assert.That(OpenAi.ConfigureModelOptions(model, options)).IsNull();
            await Assert.That(options.ReasoningEffortLevel!.Value.ToString()).IsEqualTo("none");
            options.ReasoningEffortLevel = new ChatReasoningEffortLevel("high");
            await Assert.That(OpenAi.ConfigureModelOptions(model, options)).IsNotNull();
            await Assert.That(options.ReasoningEffortLevel.Value.ToString()).IsEqualTo("high");
        }
#pragma warning restore OPENAI001
    }

    [Test]
    public async ValueTask ResponsesOnlyToolModelsStillAllowTextCompletions() {
        foreach (var model in new[] { OpenAi.GPT_62_ASTRA, OpenAi.GPT_62_SOL, OpenAi.GPT_6_ASTRA, OpenAi.GPT_61_SOL, OpenAi.GPT_61_TERRA }) {
            await Assert.That(OpenAi.ConfigureModelOptions(model, WithTool())).IsNotNull();
            await Assert.That(OpenAi.ConfigureModelOptions(model, new ChatCompletionOptions())).IsNull();
        }
    }

    static ChatCompletionOptions WithTool() {
        var options = new ChatCompletionOptions();
        options.Tools.Add(ChatTool.CreateFunctionTool("lookup"));
        return options;
    }
}
