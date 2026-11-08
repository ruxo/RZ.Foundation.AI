using System.Text.Json;
using System.Text.Json.Serialization;

namespace RZ.Foundation.AI;

internal sealed class ChatRateJsonConverter : JsonConverter<ChatRate>
{
    public override ChatRate Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) {
        var fields = JsonSerializer.Deserialize<Fields>(ref reader, options)
                     ?? throw new JsonException("Expected a ChatRate object.");
        return new(fields.InputUsdPerMillion, fields.OutputUsdPerMillion, fields.OtherUsdPerMillion,
                   fields.CachedInputUsdPerMillion);
    }

    public override void Write(Utf8JsonWriter writer, ChatRate value, JsonSerializerOptions options)
        => JsonSerializer.Serialize(writer, new Fields(value.InputUsdPerMillion, value.OutputUsdPerMillion, value.OtherUsdPerMillion) {
            CachedInputUsdPerMillion = value.CachedInputUsdPerMillion
        }, options);

    sealed record Fields(
        [property: JsonRequired, JsonIgnore(Condition = JsonIgnoreCondition.Never)] decimal InputUsdPerMillion,
        [property: JsonRequired, JsonIgnore(Condition = JsonIgnoreCondition.Never)] decimal OutputUsdPerMillion,
        [property: JsonRequired, JsonIgnore(Condition = JsonIgnoreCondition.Never)] decimal OtherUsdPerMillion)
    {
        decimal? cachedInput;

        [JsonIgnore(Condition = JsonIgnoreCondition.Never)]
        public decimal CachedInputUsdPerMillion {
            get => cachedInput ?? InputUsdPerMillion;
            init => cachedInput = value;
        }
    }
}
