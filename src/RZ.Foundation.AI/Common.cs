global using AgentResponse = System.Threading.Tasks.Task<(System.Collections.Generic.IReadOnlyList<RZ.Foundation.AI.ChatEntry> Chat, RZ.Foundation.AI.ChatCost Cost)>;
using System.ComponentModel;
using System.Reflection;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using RZ.Foundation.Types;
using PureAttribute = System.Diagnostics.Contracts.PureAttribute;

namespace RZ.Foundation.AI;

public readonly record struct AgentCommonParameters
{
    public float? Temperature { get; init; }
    public float? TopP { get; init; }

    public static readonly AgentCommonParameters Focused = new(){ Temperature = 0.01f, TopP = 0.1f };
}

public enum ChatRole
{
    System,
    Tool,
    ToolResponse,
    Agent, User, Developer,

    // Special roles
    Marker, Admin, ToolOutput
}

/// <summary>
///
/// </summary>
/// <param name="Input">Cost of input tokens per one million</param>
/// <param name="Output">Cost of output tokens per one million</param>
public readonly record struct ChatCost(decimal Input, decimal Output)
{
    public static readonly ChatCost Zero = new(0, 0);

    /// <summary>
    /// Return price per million tokens
    /// </summary>
    public decimal CalcInput(int tokens) => Input * tokens / 1_000_000m;

    /// <summary>
    /// Return price per million tokens
    /// </summary>
    public decimal CalcOutput(int tokens) => Output * tokens / 1_000_000m;

    [JsonIgnore]
    public decimal Total => Input + Output;

    public static ChatCost operator +(ChatCost a, ChatCost b)
        => new(a.Input + b.Input, a.Output + b.Output);
}

public readonly record struct ToolRequest(string Id, string Function, JsonNode? Arguments);
public readonly record struct ToolResponse(string Id, JsonNode Response);

[JsonPolymorphic(TypeDiscriminatorPropertyName = "type")]
[JsonDerivedType(typeof(Text), "text")]
[JsonDerivedType(typeof(Image), "image")]
[JsonDerivedType(typeof(Audio), "audio")]
[JsonDerivedType(typeof(File), "file")]
[JsonDerivedType(typeof(ImageUri), "image-uri")]
[JsonDerivedType(typeof(AudioUri), "audio-uri")]
[JsonDerivedType(typeof(FileUri), "file-uri")]
public abstract record ContentType
{
    public sealed record Text(string Content) : ContentType;
    public sealed record Image(byte[] Content, string MediaType) : ContentType;
    public sealed record Audio(byte[] Content, string MediaType) : ContentType;
    public sealed record File(byte[] Content, string MediaType, string FileName) : ContentType;

    public sealed record ImageUri(WebRequestData Request) : ContentType;
    public sealed record AudioUri(WebRequestData Request) : ContentType;
    public sealed record FileUri(WebRequestData Request, string FileName) : ContentType;
}

[JsonPolymorphic(TypeDiscriminatorPropertyName = "type")]
[JsonDerivedType(typeof(Content), "content")]
[JsonDerivedType(typeof(MultiContent), "multi-content")]
[JsonDerivedType(typeof(ToolCall), "tool-call")]
[JsonDerivedType(typeof(ToolResult), "tool-result")]
public abstract record ChatMessage
{
    public sealed record Content(ChatRole Role, string Message) : ChatMessage;
    public sealed record MultiContent(ChatRole Role, IReadOnlyList<ContentType> Message) : ChatMessage;
    public sealed record ToolCall(IReadOnlyList<ToolRequest> Requests) : ChatMessage;
    public sealed record ToolResult(ToolResponse Response) : ChatMessage;
}

public sealed record ChatEntry(DateTimeOffset Timestamp, ChatMessage Message, string? Admin, ChatCost Cost);

public delegate AgentResponse AiChatFunc(IEnumerable<ChatMessage> messages);

public readonly record struct CostStructure(ChatCost Text, int InputThreshold, ChatCost HighText, ChatCost Thought)
{
    [Pure]
    public static CostStructure Simple(decimal input, decimal output)
        => new(ChatCost.Zero, 0, new ChatCost(input, output), ChatCost.Zero);

    [Pure]
    public static CostStructure WithThought(decimal input, decimal output, decimal thought)
        => new(ChatCost.Zero, 0, new ChatCost(input, output), new(0, thought));
}

[JsonPolymorphic(TypeDiscriminatorPropertyName = "JsonType")]
[JsonDerivedType(typeof(StringType), "string")]
[JsonDerivedType(typeof(NumberType), "number")]
[JsonDerivedType(typeof(BooleanType), "boolean")]
[JsonDerivedType(typeof(EnumType), "enum")]
public abstract record ToolParameterType
{
    public sealed record StringType : ToolParameterType
    {
        public override string JsonType => "string";
    }
    public sealed record NumberType : ToolParameterType
    {
        public override string JsonType => "number";
    }
    public sealed record BooleanType : ToolParameterType
    {
        public override string JsonType => "boolean";
    }
    public sealed record EnumType(IReadOnlyList<string> Literals) : ToolParameterType
    {
        public override string JsonType => "string";
    }

    public static readonly ToolParameterType String = new StringType();
    public static readonly ToolParameterType Number = new NumberType();
    public static readonly ToolParameterType Boolean = new BooleanType();
    public static ToolParameterType Enum(IEnumerable<string> literals) => new EnumType(literals as IReadOnlyList<string> ?? literals.AsArray());

    public static ToolParameterType From(Type t) {
    rerun:
        if (t == typeof(string)) return String;
        if (t == typeof(int) || t == typeof(double) || t == typeof(decimal)) return Number;
        if (t == typeof(bool)) return Boolean;
        if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(Nullable<>)){
            t = t.GenericTypeArguments[0];
            goto rerun;
        }
        if (!t.IsEnum) throw new NotSupportedException($"Type {t.Name} is not supported for tool parameters");
        var literals = System.Enum.GetValues(t).Cast<object>().Map(x => x.ToString()!);
        return Enum(literals);
    }

    [JsonIgnore]
    public abstract string JsonType { get; }
}

public readonly record struct ToolParameter(string Name, string? Description, ToolParameterType Type, Option<object>? DefaultValue)
{
    public bool IsOptional => DefaultValue is not null;
}

public readonly record struct ToolDefinition(string Name, string? Description, IReadOnlyList<ToolParameter> Parameters)
{
    [Pure]
    public static ToolDefinition From(string name, MethodInfo method) {
        var description = method.GetCustomAttribute<DescriptionAttribute>()?.Description;
        var @params = (from p in method.GetParameters()
                       let paramDesc = p.GetCustomAttribute<DescriptionAttribute>()?.Description
                       select new ToolParameter(p.Name ?? throw new InvalidOperationException($"Parameter {p.Name} has no name"),
                                                paramDesc,
                                                ToolParameterType.From(p.ParameterType),
                                                DefaultValue: p.HasDefaultValue ? (Option<object>?)Optional(p.DefaultValue) : null)
                      ).ToArray();
        return new(name, description, @params);
    }

    [Pure]
    public object ToJson()
        => new {
            name = Name,
            description = Description,
            parameters = Parameters.Map(p => new {
                name = p.Name,
                description = p.Description,
                type = p.Type.JsonType,
                isOptional = p.IsOptional,
                @enum = p.Type is ToolParameterType.EnumType et ? et.Literals : null
            }).ToArray()
        };

    [Pure]
    public JsonObject ToJsonSchema() {
        var properties = new JsonObject();
        foreach (var p in Parameters){
            var type = p.Type.JsonType;
            var prop = new JsonObject {
                ["type"] = type,
                ["description"] = p.Description
            };
            properties[p.Name] = prop;
        }
        var required = new JsonArray((from p in Parameters
                                      where p.DefaultValue is null
                                      select (JsonNode)p.Name
                                     ).ToArray());

        return new JsonObject {
            ["name"] = Name,
            ["description"] = Description,
            ["parameters"] = new JsonObject {
                ["type"] = "object",
                ["properties"] = properties,
                ["required"] = required
            }
        };
    }
}

/// <summary>
/// An attribute used to mark methods as AI tools, allowing them to be discovered and registered
/// as available tools for AI agents. When applied to a method, it designates that method as
/// an AI-callable tool with a specified name.
/// </summary>
/// <remarks>
/// When no name is provided, the method name will be used as the tool name.
/// The marked method must have parameters that can be converted to supported tool parameter types:
/// - string
/// - number (int, double, decimal)
/// - boolean
/// - enum
/// </remarks>
/// <param name="name">Optional custom name for the tool. If null, the method name will be used.</param>
[PublicAPI, AttributeUsage(AttributeTargets.Method)]
public class AiToolNameAttribute(string? name = null) : Attribute
{
    public string? Name => name;
}