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
    public const string gemini_15_flash_8b = "gemini-1.5-flash-8b";
    public const string gemini_20_flash_lite = "gemini-2.0-flash-lite";
    public const string gemini_20_flash = "gemini-2.0-flash";
    public const string gemini_25_flash = "gemini-2.5-flash-preview-04-17";

    static readonly Map<string, CostStructure> RateTable = LanguageExt.Prelude.Map(
            (gemini_15_flash_8b, new CostStructure(new(0.0375m, 0.015m), 128 * 1024, new(0.075m, 0.3m), ChatCost.Zero)),
            (gemini_20_flash_lite, CostStructure.Simple(0.075m, 0.3m)),
            (gemini_20_flash, CostStructure.Simple(0.15m, 0.6m)),
            (gemini_25_flash, CostStructure.WithThought(0.15m, 0.6m, 3.5m))
        );

    public static bool IsModelSupported(string model)
        => RateTable.ContainsKey(model);

    public AiChatFunc CreateModel(string model, in AgentCommonParameters? cp = null) {
        var effectiveModel = model == "gemini-2.5-flash"? gemini_25_flash : model;
        var thinkingConfig = effectiveModel == gemini_25_flash? new ThinkingConfig{IncludeThoughts = false} : null;
        var config = new GenerationConfig {
            Temperature = cp?.Temperature,
            TopP = cp?.TopP,
            ThinkingConfig = thinkingConfig
        };
        var ai = CreateNative();

        return mList => {
            var (systemContent, history) = ToGeminiContent(mList);
            var request = new GenerateContentRequest(history, systemInstruction: systemContent, generationConfig: config);
            return ai(effectiveModel, request);
        };
    }

    public GeminiAiFunc CreateNative()
        => async (model, request) => {
            var effectiveModel = model == "gemini-2.5-flash" ? gemini_25_flash : model;
            var rate = RateTable[effectiveModel];

            var result = await GenerateContentAsync(effectiveModel, request);
            var meta = result.UsageMetadata ?? throw new Exception("No usage metadata");
            var cost = LLM.CalcCost(rate, meta.PromptTokenCount, meta.CandidatesTokenCount, meta.ThoughtsTokenCount);
            var entryIndex = 0;
            var entries = from h in result.Candidates
                          let content = h.Content ?? throw new Exception("No content")
                          from p in content.Parts
                          let role = ToChatRole(content.Role ?? throw new Exception("No role"))
                          let now = (clock ?? TimeProvider.System).GetLocalNow()
                          let message = new ChatMessage.Content(role, p.Text ?? throw new Exception("No text"))
                          select new ChatEntry(now, message, Admin: null, getCost());
            return (entries.ToArray(), cost);

            ChatCost getCost() => entryIndex++ == 0 ? cost : ChatCost.Zero;
        };

    public static (Content? System, List<Content> Content) ToGeminiContent(IEnumerable<ChatMessage> content) {
        var messages = LanguageExt.Prelude.toList(content);
        var (system, historyList) = messages.Case switch {
            ChatMessage m                                => GetSystemMessage(m) is { } message ? (message, Lst<ChatMessage>.Empty) : (null, messages),
            (ChatMessage m, IEnumerable<ChatMessage> xs) => GetSystemMessage(m) is { } message ? (message, LanguageExt.Prelude.toList(xs)) : (null, messages),
            _                                            => (null, Lst<ChatMessage>.Empty)
        };
        var systemContent = system is not null ? new Content(system, Roles.System) : null;
        var history = historyList.Choose(x => Optional(ConvertChatEntryToContent(x))).ToList();
        return (systemContent, history);
    }

    static string? GetSystemMessage(ChatMessage message)
        => message is ChatMessage.Content { Role: ChatRole.System } c ? c.Message : null;

    static ChatRole ToChatRole(string role)
        => role switch {
            Roles.Model    => ChatRole.Agent,
            Roles.System   => ChatRole.System,
            Roles.User     => ChatRole.User,
            Roles.Function => ChatRole.ToolResponse,
            _              => throw new NotSupportedException($"Unknown role: {role}")
        };

    static Content? ConvertChatEntryToContent(ChatMessage entry)
        => entry is ChatMessage.Content c
               ? c.Role switch {
                   ChatRole.Agent                      => new(c.Message, Roles.Model),
                   ChatRole.System                     => new(c.Message, Roles.System),
                   ChatRole.Developer or ChatRole.User => new(c.Message, Roles.User),
                   ChatRole.ToolResponse                => new(c.Message, Roles.Function),
                   ChatRole.Admin or ChatRole.Marker   => null,

                   ChatRole.Tool or ChatRole.ToolOutput => throw new Exception("Tool roles are not expected here!"),

                   _ => throw new NotSupportedException($"Unknown role: {c.Role}")
               }
               : throw new NotSupportedException("Tool messages are not yet supported");
}