// ReSharper disable InconsistentNaming

using GenerativeAI;
using GenerativeAI.Types;
using Microsoft.Extensions.Logging;

namespace RZ.Foundation.AI;

public delegate AgentResponse GeminiAiFunc(string model, GenerateContentRequest request);

[PublicAPI]
public class GeminiAi(string apiKey, HttpClient? http = null, ILogger? logger = null, TimeProvider? clock = null)
    : BaseModel(new GoogleAIPlatformAdapter(apiKey), http ?? new HttpClient(), logger)
{
    public const string GEMINI_20_FLASH_LITE = "gemini-2.0-flash-lite";
    public const string GEMINI_20_FLASH = "gemini-2.0-flash";
    public const string GEMINI_25_FLASH = "gemini-2.5-flash-preview-04-17";

    static readonly Map<string, CostStructure> RateTable = LanguageExt.Prelude.Map(
            (GEMINI_20_FLASH_LITE, CostStructure.Simple(0.075m, 0.3m)),
            (GEMINI_20_FLASH, CostStructure.Simple(0.15m, 0.6m)),
            (GEMINI_25_FLASH, CostStructure.WithThought(0.15m, 0.6m, 3.5m))
        );

    public static bool IsModelSupported(string model)
        => RateTable.ContainsKey(model);

    public AiChatFunc CreateModel(string model, in AgentCommonParameters? cp = null) {
        var effectiveModel = model == "gemini-2.5-flash" ? GEMINI_25_FLASH : model;
        var thinkingConfig = effectiveModel == GEMINI_25_FLASH ? new ThinkingConfig { IncludeThoughts = false } : null;
        var config = new GenerationConfig {
            Temperature = cp?.Temperature,
            TopP = cp?.TopP,
            ThinkingConfig = thinkingConfig
        };
        var aiCreationResult = TryCatch(CreateNative);

        return async mList => {
            if (Fail(aiCreationResult, out var e, out var ai)) return e;
            if (Fail(ToGeminiContent(mList), out e, out var tuple)) return e;
            var (systemContent, history) = tuple;
            var request = new GenerateContentRequest(history, systemInstruction: systemContent, generationConfig: config);
            return await ai(effectiveModel, request).ConfigureAwait(false);
        };
    }

    public GeminiAiFunc CreateNative()
        => async (model, request) => {
            var effectiveModel = model == "gemini-2.5-flash" ? GEMINI_25_FLASH : model;
            var rate = RateTable[effectiveModel];

            var result = await GenerateContentAsync(effectiveModel, request).ConfigureAwait(false);
            if (result.UsageMetadata is not { } meta) return new ErrorInfo(Unhandled, "No usage metadata");
            if (result.Candidates is null) return new ErrorInfo(Unhandled, "No candidates");

            var cost = LLM.CalcCost(rate, meta.PromptTokenCount, meta.CandidatesTokenCount, meta.ThoughtsTokenCount);
            var entryIndex = 0;
            var now = (clock ?? TimeProvider.System).GetLocalNow();
            ErrorInfo? e;
            var entriesResult = result.Candidates.SelectMany(h => {
                if (h.Content is not { } content) return [new ErrorInfo(InvalidRequest, "No content")];
                if (content.Role is not { } contentRole) return [new ErrorInfo(InvalidRequest, "No role")];
                if (Fail(ToChatRole(contentRole), out e, out var role)) return [e];

                return h.Content.Parts.Map(p => {
                    if (p.Text is not { } text) return FailedOutcome<ChatEntry>(new(InvalidRequest, "No text"));
                    var message = new ChatMessage.Content(role, text);
                    return new ChatEntry(now, message, Admin: null, getCost());
                });
            });
            if (Fail(entriesResult.MakeList(), out e, out var entries)) return e;
            return (entries, cost);

            ChatCost getCost() => entryIndex++ == 0 ? cost : ChatCost.Zero;
        };

    public static Outcome<(Content? System, List<Content> Content)> ToGeminiContent(IEnumerable<ChatMessage> content) {
        var messages = LanguageExt.Prelude.toList(content);
        var (system, historyList) = messages.Case switch {
            ChatMessage m                                => GetSystemMessage(m) is { } message ? (message, Lst<ChatMessage>.Empty) : (null, messages),
            (ChatMessage m, IEnumerable<ChatMessage> xs) => GetSystemMessage(m) is { } message ? (message, LanguageExt.Prelude.toList(xs)) : (null, messages),
            _                                            => (null, Lst<ChatMessage>.Empty)
        };
        var systemContent = system is not null ? new Content(system, Roles.System) : null;
        if (Fail(historyList.Choose(x => Optional(ConvertChatEntryToContent(x))).MakeMutableList(), out var e, out var history)) return e;

        return (systemContent, history);
    }

    static string? GetSystemMessage(ChatMessage message)
        => message is ChatMessage.Content { Role: ChatRole.System } c ? c.Message : null;

    static Outcome<ChatRole> ToChatRole(string role)
        => role switch {
            Roles.Model    => ChatRole.Agent,
            Roles.System   => ChatRole.System,
            Roles.User     => ChatRole.User,
            Roles.Function => ChatRole.ToolResponse,
            _              => new ErrorInfo(Unhandled, $"Unknown role: {role}")
        };

    static Outcome<Content> ConvertChatEntryToContent(ChatMessage entry)
        => entry is ChatMessage.Content c
               ? c.Role switch {
                   ChatRole.Agent                      => new Content(c.Message, Roles.Model),
                   ChatRole.System                     => new Content(c.Message, Roles.System),
                   ChatRole.Developer or ChatRole.User => new Content(c.Message, Roles.User),
                   ChatRole.ToolResponse               => new Content(c.Message, Roles.Function),
                   ChatRole.Admin or ChatRole.Marker   => ErrorInfo.NotFound,

                   ChatRole.Tool or ChatRole.ToolOutput => new ErrorInfo(Unhandled, "Tool roles are not expected here!"),

                   _ => new ErrorInfo(Unhandled, $"Unknown role: {c.Role}")
               }
               : new ErrorInfo(Unhandled, "Tool messages are not yet supported");
}