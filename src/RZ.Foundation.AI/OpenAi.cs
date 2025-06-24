// ReSharper disable InconsistentNaming

using System.Text.Json.Nodes;
using OpenAI.Chat;
using RZ.Foundation.Types;

namespace RZ.Foundation.AI;

public delegate AgentResponse OpenAiFunc(IEnumerable<ChatMessage> messages, ChatCompletionOptions options);

[PublicAPI]
public class OpenAi(string apiKey, TimeProvider? clock = null)
{
    public const string gpt_4o_mini = "gpt-4o-mini";
    public const string gpt_41_mini = "gpt-4.1-mini";
    public const string gpt_41_nano = "gpt-4.1-nano";
    public const string gpt_4o = "gpt-4o";
    public const string gpt_41 = "gpt-4.1";

    static readonly Map<string, CostStructure> RateTable = LanguageExt.Prelude.Map(
            (gpt_4o_mini, CostStructure.Simple(0.15m, 0.6m)),
            (gpt_41_mini, CostStructure.Simple(0.4m, 1.6m)),
            (gpt_41_nano, CostStructure.Simple(0.1m, 0.4m)),
            (gpt_4o, CostStructure.Simple(2.5m, 10m)),
            (gpt_41, CostStructure.Simple(2m, 8m))
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
            var history = await messages.ChooseAsync(Convert).ToArrayAsync();
            var completion = await ai.CompleteChatAsync(history, options);
            var usage = completion.Value.Usage;
            var cost = LLM.CalcCost(rate, usage.InputTokenCount, usage.OutputTokenCount, 0);
            var entryIndex = 0;
            var result = ReadResult(clock ?? TimeProvider.System, getCost, completion).ToArray();
            return (result, cost);

            ChatCost getCost() => entryIndex++ == 0? cost : ChatCost.Zero;
        };

        async ValueTask<Option<OpenAI.Chat.ChatMessage>> Convert(ChatMessage cm)
            => Optional(await ConvertChatMessageToMessage(http, cm));
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

    static IEnumerable<ChatEntry> ReadResult(TimeProvider clock, Func<ChatCost> getCost, ChatCompletion completion) {
        var now = clock.GetLocalNow();
        return completion.FinishReason switch {
            ChatFinishReason.Stop => from part in completion.Content
                                     where part.Kind == ChatMessageContentPartKind.Text
                                     select new ChatEntry(now, new ChatMessage.Content(ChatRole.Agent, part.Text), Admin: null, getCost()),

            ChatFinishReason.ToolCalls => [
                new ChatEntry(now,
                              new ChatMessage.ToolCall((from toolCall in completion.ToolCalls
                                                        let arguments = JsonNode.Parse(toolCall.FunctionArguments)
                                                        select new ToolRequest(toolCall.Id, toolCall.FunctionName, arguments)).ToArray()), Admin: null, getCost())
            ],

            ChatFinishReason.Length        => throw new ErrorInfoException(StandardErrorCodes.ServiceError, "Incomplete model output due to MaxTokens parameter or token limit exceeded."),
            ChatFinishReason.ContentFilter => throw new ErrorInfoException(StandardErrorCodes.ServiceError, "Omitted content due to a content filter flag."),
            ChatFinishReason.FunctionCall  => throw new ErrorInfoException(StandardErrorCodes.ServiceError, "Deprecated in favor of tool calls."),

            _ => throw new NotSupportedException($"Unknown finish reason: {completion.FinishReason}")
        };
    }

    static async ValueTask<OpenAI.Chat.ChatMessage?> ConvertChatMessageToMessage(HttpClient http, ChatMessage cm)
        => cm switch {
            ChatMessage.Content entry      => ToChatMessage(entry),
            ChatMessage.MultiContent entry => await ToChatMessage(http, entry),
            ChatMessage.ToolCall tc        => ToChatMessage(tc),
            ChatMessage.ToolResult tr      => ToolChatMessage(tr),

            _ => throw new NotSupportedException($"Unknown message type: {cm.GetType().Name}")
        };

    static OpenAI.Chat.ChatMessage? ToChatMessage(ChatMessage.Content entry)
        => entry.Role switch {
            ChatRole.Agent     => new AssistantChatMessage(entry.Message),
            ChatRole.System    => new SystemChatMessage(entry.Message),
            ChatRole.User      => new UserChatMessage(entry.Message),
            ChatRole.Developer => new DeveloperChatMessage(entry.Message),

            ChatRole.Admin or ChatRole.Marker or ChatRole.ToolOutput => null,

            ChatRole.Tool or ChatRole.ToolResponse => throw new Exception("Tool roles are not expected here!"),

            _ => throw new NotSupportedException($"Unknown role: {entry.Role}")
        };

    static async Task<OpenAI.Chat.ChatMessage?> ToChatMessage(HttpClient http, ChatMessage.MultiContent entry) {
        var parts = await entry.Message.MapAsync(async x => await CreatePart(http, x)).ToArrayAsync();
        return entry.Role switch {
            ChatRole.Agent     => new AssistantChatMessage(parts),
            ChatRole.System    => new SystemChatMessage(parts),
            ChatRole.User      => new UserChatMessage(parts),
            ChatRole.Developer => new DeveloperChatMessage(parts),

            ChatRole.Admin or ChatRole.Marker or ChatRole.ToolOutput => null,

            ChatRole.Tool or ChatRole.ToolResponse => throw new Exception("Tool roles are not expected here!"),

            _ => throw new NotSupportedException($"Unknown role: {entry.Role}")
        };
    }

    static async ValueTask<ChatMessageContentPart> CreatePart(HttpClient http, ContentType ct)
        => ct switch {
            ContentType.Text m  => ChatMessageContentPart.CreateTextPart(m.Content),
            ContentType.Image m => CreateImagePart((m.MediaType, m.Content)),
            ContentType.Audio m => CreateAudioPart((m.MediaType, m.Content)),
            ContentType.File m  => CreateFilePart(m.FileName, (m.MediaType, m.Content)),

            ContentType.ImageUri m => CreateImagePart(await m.Request.Retrieve(http)),
            ContentType.AudioUri m => CreateAudioPart(await m.Request.Retrieve(http)),
            ContentType.FileUri m  => CreateFilePart(m.FileName, await m.Request.Retrieve(http)),

            _ => throw new NotSupportedException($"Unknown content type: {ct.GetType().Name}")
        };

    static ChatMessageContentPart CreateImagePart((string MediaType, byte[] Data) data)
        => ChatMessageContentPart.CreateImagePart(BinaryData.FromBytes(data.Data), data.MediaType);

    static ChatMessageContentPart CreateAudioPart((string MediaType, byte[] Data) data)
        => ChatMessageContentPart.CreateInputAudioPart(BinaryData.FromBytes(data.Data), new ChatInputAudioFormat(data.MediaType));

    static ChatMessageContentPart CreateFilePart(string fileName, (string MediaType, byte[] Data) data)
        => ChatMessageContentPart.CreateFilePart(BinaryData.FromBytes(data.Data), data.MediaType, fileName);

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
        var value = new JsonObject { ["type"] = parameter.Type.JsonType };
        if (parameter.Description is not null)
            value["description"] = parameter.Description;
        if (parameter.Type is ToolParameterType.EnumType e)
            value["enum"] = new JsonArray(e.Literals.Map(x => JsonValue.Create(x)).ToArray());
        return new KeyValuePair<string, JsonNode?>(parameter.Name, value);
    }

    static (bool IsStrict, JsonObject Parameters) ToJsonSchema(IReadOnlyList<ToolParameter> parameters) {
        var required = (from p in parameters where !p.IsOptional select (JsonNode) p.Name).ToArray();
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