using System.Text.Json;
using System.Text.Json.Serialization;

namespace RZ.Foundation.AI;

internal sealed class TokenCostJsonConverter : JsonConverter<TokenCost>
{
    public override TokenCost Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) {
        var fields = JsonSerializer.Deserialize<Fields>(ref reader, options)
                     ?? throw new JsonException("Expected a TokenCost object.");
        return new(fields.Tokens, fields.Usd);
    }

    public override void Write(Utf8JsonWriter writer, TokenCost value, JsonSerializerOptions options)
        => JsonSerializer.Serialize(writer, new Fields(value.Tokens, value.Usd), options);

    sealed record Fields(
        [property: JsonRequired, JsonIgnore(Condition = JsonIgnoreCondition.Never)] long Tokens,
        [property: JsonRequired, JsonIgnore(Condition = JsonIgnoreCondition.Never)] decimal Usd);
}
