using System.Text.Json.Serialization;

namespace RZ.Foundation.AI;

/// <summary>A captured token count and USD amount. Neither value is inferred from the other.</summary>
[JsonConverter(typeof(TokenCostJsonConverter))]
public readonly record struct TokenCost
{
    public long Tokens { get; }
    public decimal Usd { get; }

    public TokenCost(long Tokens, decimal Usd) {
        ArgumentOutOfRangeException.ThrowIfNegative(Tokens);
        ArgumentOutOfRangeException.ThrowIfNegative(Usd);
        this.Tokens = Tokens;
        this.Usd = Usd;
    }

    public static readonly TokenCost Zero = new(0, 0);

    public static TokenCost operator +(TokenCost a, TokenCost b)
        => new(a.Tokens + b.Tokens, a.Usd + b.Usd);
}

[method: JsonConstructor]
public readonly record struct ChatCost(
    [property: JsonRequired, JsonIgnore(Condition = JsonIgnoreCondition.Never)] TokenCost Input,
    [property: JsonRequired, JsonIgnore(Condition = JsonIgnoreCondition.Never)] TokenCost Output,
    [property: JsonRequired, JsonIgnore(Condition = JsonIgnoreCondition.Never)] TokenCost Other)
{
    public static readonly ChatCost Zero = new(TokenCost.Zero, TokenCost.Zero, TokenCost.Zero);

    [JsonIgnore]
    public TokenCost Total => Input + Output + Other;

    public static ChatCost operator +(ChatCost a, ChatCost b)
        => new(a.Input + b.Input, a.Output + b.Output, a.Other + b.Other);
}

[JsonConverter(typeof(ChatRateJsonConverter))]
public readonly record struct ChatRate
{
    public decimal InputUsdPerMillion { get; }
    public decimal OutputUsdPerMillion { get; }
    public decimal OtherUsdPerMillion { get; }
    public decimal CachedInputUsdPerMillion { get; }

    public ChatRate(decimal InputUsdPerMillion, decimal OutputUsdPerMillion, decimal OtherUsdPerMillion)
        : this(InputUsdPerMillion, OutputUsdPerMillion, OtherUsdPerMillion, InputUsdPerMillion) { }

    public ChatRate(decimal InputUsdPerMillion, decimal OutputUsdPerMillion, decimal OtherUsdPerMillion,
                    decimal CachedInputUsdPerMillion) {
        ArgumentOutOfRangeException.ThrowIfNegative(InputUsdPerMillion);
        ArgumentOutOfRangeException.ThrowIfNegative(OutputUsdPerMillion);
        ArgumentOutOfRangeException.ThrowIfNegative(OtherUsdPerMillion);
        ArgumentOutOfRangeException.ThrowIfNegative(CachedInputUsdPerMillion);
        this.InputUsdPerMillion = InputUsdPerMillion;
        this.OutputUsdPerMillion = OutputUsdPerMillion;
        this.OtherUsdPerMillion = OtherUsdPerMillion;
        this.CachedInputUsdPerMillion = CachedInputUsdPerMillion;
    }
}

public abstract class CostStructure
{
    private protected CostStructure() { }

    public abstract ChatRate RateFor(long inputTokens);

    [Pure]
    public static CostStructure Simple(decimal input, decimal output, decimal? other = null, decimal? cachedInput = null)
        => new FlatCostStructure(new(input, output, other ?? output, cachedInput ?? input));

    [Pure]
    public static CostStructure WithThought(decimal input, decimal output, decimal thought, decimal? cachedInput = null)
        => Simple(input, output, thought, cachedInput);
}

public sealed class FlatCostStructure(ChatRate rate) : CostStructure
{
    public ChatRate Rate { get; } = rate;

    public override ChatRate RateFor(long inputTokens) => Rate;
}

public sealed class TieredCostStructure : CostStructure
{
    public long InputThresholdTokens { get; }
    public ChatRate Standard { get; }
    public ChatRate High { get; }

    public TieredCostStructure(long inputThresholdTokens, ChatRate standard, ChatRate high) {
        ArgumentOutOfRangeException.ThrowIfNegative(inputThresholdTokens);
        InputThresholdTokens = inputThresholdTokens;
        Standard = standard;
        High = high;
    }

    public override ChatRate RateFor(long inputTokens) => inputTokens > InputThresholdTokens ? High : Standard;
}
