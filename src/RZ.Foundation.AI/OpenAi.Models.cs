using OpenAI.Chat;

namespace RZ.Foundation.AI;

public partial class OpenAi
{
    public const string GPT_62_ASTRA = "gpt-6.2-astra";
    public const string GPT_62_SOL = "gpt-6.2-sol";
    public const string GPT_6_ASTRA = "gpt-6-astra";
    public const string GPT_61_SOL = "gpt-6.1-sol";
    public const string GPT_61_TERRA = "gpt-6.1-terra";
    public const string GPT_6_SOL = "gpt-6-sol";
    public const string GPT_6_LUNA = "gpt-6-luna";
    public const string GPT_56_SOL = "gpt-5.6-sol";
    public const string GPT_56 = "gpt-5.6";
    public const string GPT_56_TERRA = "gpt-5.6-terra";
    public const string GPT_56_LUNA = "gpt-5.6-luna";
    public const string GPT_55 = "gpt-5.5";
    public const string GPT_54 = "gpt-5.4";
    public const string GPT_54_MINI = "gpt-5.4-mini";
    public const string GPT_52 = "gpt-5.2";
    public const string CHAT_LATEST = "chat-latest";

    [Obsolete("This model is retired or deprecated; use a current SupportedModels entry.")]
    public const string GPT_54_NANO = "gpt-5.4-nano";
    [Obsolete("This model is retired or deprecated; use a current SupportedModels entry.")]
    public const string GPT_51 = "gpt-51";
    [Obsolete("This model is retired or deprecated; use a current SupportedModels entry.")]
    public const string GPT_5 = "gpt-5";
    [Obsolete("This model is retired or deprecated; use a current SupportedModels entry.")]
    public const string GPT_5_MINI = "gpt-5-mini";
    [Obsolete("This model is retired or deprecated; use a current SupportedModels entry.")]
    public const string GPT_5_NANO = "gpt-5-nano";
    [Obsolete("This model is retired or deprecated; use a current SupportedModels entry.")]
    public const string GPT_41_MINI = "gpt-4.1-mini";
    [Obsolete("This model is retired or deprecated; use a current SupportedModels entry.")]
    public const string GPT_41_NANO = "gpt-4.1-nano";
    [Obsolete("This model is retired or deprecated; use a current SupportedModels entry.")]
    public const string GPT_41 = "gpt-4.1";
    [Obsolete("This model is retired or deprecated; use a current SupportedModels entry.")]
    public const string GPT_4O_MINI = "gpt-4o-mini";
    [Obsolete("This model is retired or deprecated; use a current SupportedModels entry.")]
    public const string GPT_4O = "gpt-4o";

    // Sort by the short-context rate for equal uncached input and output token counts.
    static readonly Map<string, CostStructure> RateTable = LanguageExt.Prelude.Map(
        (GPT_6_LUNA, new TieredCostStructure(272_000, new(0.1m, 0.5m, 0.5m, 0.01m), new(0.2m, 0.75m, 0.75m, 0.02m))),
        (GPT_56_LUNA, new TieredCostStructure(272_000, new(0.2m, 1.2m, 1.2m, 0.02m), new(0.4m, 1.8m, 1.8m, 0.04m))),
        (GPT_54_MINI, CostStructure.Simple(0.75m, 4.5m, cachedInput: 0.075m)),
        (GPT_61_TERRA, new TieredCostStructure(272_000, new(1m, 5m, 5m, 0.05m), new(2m, 7.5m, 7.5m, 0.1m))),
        (GPT_6_SOL, new TieredCostStructure(272_000, new(2m, 10m, 10m, 0.2m), new(4m, 15m, 15m, 0.4m))),
        (GPT_61_SOL, new TieredCostStructure(272_000, new(2m, 10m, 10m, 0.1m), new(4m, 15m, 15m, 0.2m))),
        (GPT_62_SOL, new TieredCostStructure(272_000, new(2m, 10m, 10m, 0.1m), new(4m, 15m, 15m, 0.2m))),
        (GPT_56_TERRA, new TieredCostStructure(272_000, new(2m, 12m, 12m, 0.2m), new(4m, 18m, 18m, 0.4m))),
        (GPT_52, CostStructure.Simple(1.75m, 14m, cachedInput: 0.175m)),
        (GPT_54, new TieredCostStructure(272_000, new(2.5m, 15m, 15m, 0.25m), new(5m, 22.5m, 22.5m, 0.5m))),
        (GPT_56, new TieredCostStructure(272_000, new(4m, 20m, 20m, 0.4m), new(8m, 30m, 30m, 0.8m))),
        (GPT_56_SOL, new TieredCostStructure(272_000, new(4m, 20m, 20m, 0.4m), new(8m, 30m, 30m, 0.8m))),
        (CHAT_LATEST, CostStructure.Simple(5m, 30m, cachedInput: 0.5m)),
        (GPT_55, new TieredCostStructure(272_000, new(5m, 30m, 30m, 0.5m), new(10m, 45m, 45m, 1m))),
        (GPT_62_ASTRA, new TieredCostStructure(272_000, new(8m, 40m, 40m, 0.4m), new(16m, 60m, 60m, 0.8m))),
        (GPT_6_ASTRA, new TieredCostStructure(272_000, new(10m, 50m, 50m, 1m), new(20m, 75m, 75m, 2m)))
    );

    public static readonly IReadOnlyList<string> SupportedModels = RateTable.Keys
        .OrderBy(model => {
            var rate = GetRate(model).RateFor(0);
            return rate.InputUsdPerMillion + rate.OutputUsdPerMillion;
        })
        .ToArray().AsReadOnly();

    internal static ErrorInfo? ConfigureModelOptions(string model, ChatCompletionOptions options) {
        if (options.Tools.Count == 0) return null;
        if (model is GPT_62_ASTRA or GPT_62_SOL or GPT_6_ASTRA or GPT_61_SOL or GPT_61_TERRA)
            return new ErrorInfo(InvalidRequest, "This model requires the Responses API for tool calling.");
        if (model is GPT_6_SOL or GPT_6_LUNA){
#pragma warning disable OPENAI001
            if (options.ReasoningEffortLevel is { } effort && effort != new ChatReasoningEffortLevel("none"))
                return new ErrorInfo(InvalidRequest, "Chat Completions tool calling requires reasoning effort none for this model.");
            options.ReasoningEffortLevel = new ChatReasoningEffortLevel("none");
#pragma warning restore OPENAI001
        }
        return null;
    }
}
