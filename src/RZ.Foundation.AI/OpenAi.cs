using System.Text.Json.Nodes;
using OpenAI.Chat;

namespace RZ.Foundation.AI;

public delegate AgentResponse OpenAiFunc(IEnumerable<ChatMessage> messages, ChatCompletionOptions options);

[PublicAPI]
public class OpenAi(string apiKey, TimeProvider? clock = null)
{
    public const string GPT_4O_MINI = "gpt-4o-mini";
    public const string GPT_41_MINI = "gpt-4.1-mini";
    public const string GPT_41_NANO = "gpt-4.1-nano";
    public const string GPT_4O = "gpt-4o";
    public const string GPT_41 = "gpt-4.1";

    static readonly Map<string, CostStructure> RateTable = LanguageExt.Prelude.Map(
            (GPT_4O_MINI, CostStructure.Simple(0.15m, 0.6m)),
            (GPT_41_MINI, CostStructure.Simple(0.4m, 1.6m)),
            (GPT_41_NANO, CostStructure.Simple(0.1m, 0.4m)),
            (GPT_4O, CostStructure.Simple(2.5m, 10m)),
            (GPT_41, CostStructure.Simple(2m, 8m))
        );

    public static bool IsModelSupported(string model)
        => RateTable.ContainsKey(model);

    public AiChatFunc CreateModel(string model, IReadOnlyList<ToolDefinition>? tools = null, AgentCommonParameters? cp = null)
        => CreateModelInternal(model, tools?.Select(ToChatTool) ?? [], cp);

    public OpenAiFunc CreateNative(string model, HttpClient? http = null) {
        http ??= new HttpClient();
        var rate = RateTable[model];
        var ai = new ChatClient(model, apiKey);

        return async (messages, options) => {
            if (Fail(await messages.ChooseAsync((x, _, _) => ConvertChatMessageToMessage(http, x)).MakeList(), out var e, out var history)
             || Fail(await TryCatch(ai.CompleteChatAsync(history, options)), out e, out var completion)) return e;

            var usage = completion.Value.Usage;
            var cost = LLM.CalcCost(rate, usage.InputTokenCount, usage.OutputTokenCount, 0);
            var entryIndex = 0;
            if (Fail(ReadResult(clock ?? TimeProvider.System, GetCost, completion), out e, out var result)) return e;
            return (result, cost);

            ChatCost GetCost() => entryIndex++ == 0 ? cost : ChatCost.Zero;
        };
    }

    AiChatFunc CreateModelInternal(string model, IEnumerable<ChatTool> tools, AgentCommonParameters? cp = null) {
        var options = new ChatCompletionOptions {
            Temperature = cp?.Temperature,
            TopP = cp?.TopP
        };
        tools.Iter(options.Tools.Add);
        var ai = CreateNative(model);

        return messages => ai(messages, options);
    }

    #region Response transformation

    static Outcome<IReadOnlyList<ChatEntry>> ReadResult(TimeProvider clock, Func<ChatCost> getCost, ChatCompletion completion) {
        var now = clock.GetLocalNow();
        return completion.FinishReason switch {
            ChatFinishReason.Stop => SuccessOutcome(ReadOnly(from part in completion.Content
                                                             where part.Kind == ChatMessageContentPartKind.Text
                                                             select new ChatEntry(now, new ChatMessage.Content(ChatRole.Agent, part.Text), Admin: null, getCost()))),

            ChatFinishReason.ToolCalls => ToolCallResult(completion, getCost, now),

            ChatFinishReason.Length        => new ErrorInfo(ServiceError, "Incomplete model output due to MaxTokens parameter or token limit exceeded."),
            ChatFinishReason.ContentFilter => new ErrorInfo(ServiceError, "Omitted content due to a content filter flag."),
            ChatFinishReason.FunctionCall  => new ErrorInfo(ServiceError, "Deprecated in favor of tool calls."),

            _ => new ErrorInfo(Unhandled, $"Unknown finish reason: {completion.FinishReason}")
        };

        static Outcome<IReadOnlyList<ChatEntry>> ToolCallResult(ChatCompletion completion, Func<ChatCost> getCost, DateTimeOffset now) {
            var ops = completion.ToolCalls
                                .Map(toolCall => from arguments in Parse(toolCall.FunctionArguments)
                                                 select new ToolRequest(toolCall.Id, toolCall.FunctionName, arguments))
                                .MakeList();
            if (Fail(ops, out var e, out var results)) return e;

            return SuccessOutcome(ReadOnly(new ChatEntry(now, new ChatMessage.ToolCall(results), Admin: null, getCost())));
        }

        static Outcome<JsonNode> Parse(BinaryData data)
            => JsonNode.Parse(data) is { } v ? SuccessOutcome(v) : ErrorInfo.NotFound;
    }

    static async ValueTask<Outcome<OpenAI.Chat.ChatMessage>> ConvertChatMessageToMessage(HttpClient http, ChatMessage cm)
        => cm switch {
            ChatMessage.Content entry      => ToChatMessage(entry),
            ChatMessage.MultiContent entry => await ToChatMessage(http, entry),
            ChatMessage.ToolCall tc        => ToChatMessage(tc),
            ChatMessage.ToolResult tr      => ToolChatMessage(tr),

            _ => throw new NotSupportedException($"Unknown message type: {cm.GetType().Name}")
        };

    static Outcome<OpenAI.Chat.ChatMessage> ToChatMessage(ChatMessage.Content entry)
        => entry.Role switch {
            ChatRole.Agent  => new AssistantChatMessage(entry.Message),
            ChatRole.System => new SystemChatMessage(entry.Message),
            ChatRole.User   => new UserChatMessage(entry.Message),
#pragma warning disable OPENAI001
            ChatRole.Developer => new DeveloperChatMessage(entry.Message),
#pragma warning restore OPENAI001

            ChatRole.Admin or ChatRole.Marker or ChatRole.ToolOutput => ErrorInfo.NotFound,

            ChatRole.Tool or ChatRole.ToolResponse => new ErrorInfo(Unhandled, "Tool roles are not expected here!"),

            _ => new ErrorInfo(Unhandled, $"Unknown role: {entry.Role}")
        };

    static async ValueTask<Outcome<OpenAI.Chat.ChatMessage>> ToChatMessage(HttpClient http, ChatMessage.MultiContent entry) {
        if (Fail(await entry.Messages.MapAsync(async (x, _, _) => await CreatePart(http, x)).MakeList(), out var e, out var parts)) return e;

        return entry.Role switch {
            ChatRole.Agent  => new AssistantChatMessage(parts),
            ChatRole.System => new SystemChatMessage(parts),
            ChatRole.User   => new UserChatMessage(parts),
#pragma warning disable OPENAI001
            ChatRole.Developer => new DeveloperChatMessage(parts),
#pragma warning restore OPENAI001

            ChatRole.Admin or ChatRole.Marker or ChatRole.ToolOutput => ErrorInfo.NotFound,

            ChatRole.Tool or ChatRole.ToolResponse => new ErrorInfo(Unhandled, "Tool roles are not expected here!"),

            _ => new ErrorInfo(Unhandled, $"Unknown role: {entry.Role}")
        };
    }

    static async ValueTask<Outcome<ChatMessageContentPart>> CreatePart(HttpClient http, ContentType ct)
        => ct switch {
            ContentType.Text m  => ChatMessageContentPart.CreateTextPart(m.Data),
            ContentType.Image m => CreateImagePart((m.MediaType, m.Data)),
            ContentType.Audio m => CreateAudioPart((m.MediaType, m.Data)),
            ContentType.File m  => CreateFilePart(m.FileName, (m.MediaType, m.Data)),

            ContentType.ImageUri m => await (from data in m.Request.Retrieve(http) select CreateImagePart(data)),
            ContentType.AudioUri m => await (from data in m.Request.Retrieve(http) select CreateAudioPart(data)),
            ContentType.FileUri m  => await (from data in m.Request.Retrieve(http) select CreateFilePart(m.FileName, data)),

            _ => new ErrorInfo(Unhandled, $"Unknown content type: {ct.GetType().Name}")
        };

    static ChatMessageContentPart CreateImagePart((string MediaType, byte[] Data) data)
        => ChatMessageContentPart.CreateImagePart(BinaryData.FromBytes(data.Data), data.MediaType);

    static ChatMessageContentPart CreateAudioPart((string MediaType, byte[] Data) data)
#pragma warning disable OPENAI001
        => ChatMessageContentPart.CreateInputAudioPart(BinaryData.FromBytes(data.Data), new ChatInputAudioFormat(data.MediaType));
#pragma warning restore OPENAI001

    static ChatMessageContentPart CreateFilePart(string fileName, (string MediaType, byte[] Data) data)
#pragma warning disable OPENAI001
        => ChatMessageContentPart.CreateFilePart(BinaryData.FromBytes(data.Data), data.MediaType, fileName);
#pragma warning restore OPENAI001

    static AssistantChatMessage ToChatMessage(ChatMessage.ToolCall tc)
        => new(from t in tc.Requests
               select ChatToolCall.CreateFunctionToolCall(t.Id, t.Function, BinaryData.FromObjectAsJson(t.Arguments)));

    static ToolChatMessage ToolChatMessage(ChatMessage.ToolResult tr)
        => new(tr.Response.Id, tr.Response.ToString());

    #endregion

    public static ChatTool ToChatTool(ToolDefinition toolDef) {
        if (toolDef.Parameters.Count == 0)
            return ChatTool.CreateFunctionTool(toolDef.Name, toolDef.Description);

        var (strict, jsonParameters) = ToJsonSchema(toolDef.Parameters);
        var parameters = BinaryData.FromObjectAsJson(jsonParameters);
        return ChatTool.CreateFunctionTool(toolDef.Name, toolDef.Description, parameters, strict);
    }

    static KeyValuePair<string, JsonNode?> ToJsonSchema(ToolParameter parameter) {
        var value = new JsonObject { ["type"] = parameter.Type.Kind };
        if (parameter.Description is not null)
            value["description"] = parameter.Description;
        if (parameter.Type is ToolParameterType.EnumType e)
            value["enum"] = new JsonArray(e.Literals.Map(x => JsonValue.Create(x)).ToArray());
        return new KeyValuePair<string, JsonNode?>(parameter.Name, value);
    }

    static (bool IsStrict, JsonObject Parameters) ToJsonSchema(IReadOnlyList<ToolParameter> parameters) {
        var required = (from p in parameters where !p.IsOptional select (JsonNode)p.Name).ToArray();
        var strict = required.Length == parameters.Count;
        var result = new JsonObject {
            ["type"] = "object",
            ["properties"] = new JsonObject(parameters.Map(ToJsonSchema)),
            ["additionalProperties"] = false
        };
        if (required.Length > 0)
            result["required"] = new JsonArray(required);
        return (strict, result);
    }
}