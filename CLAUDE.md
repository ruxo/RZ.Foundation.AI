# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`RZ.Foundation.AI` is a .NET library providing a unified interface for multiple AI services (OpenAI, Gemini) with cost tracking and tool/function calling support.

## Commands

```bash
# Build
dotnet build
dotnet build -c Release

# Run all tests
dotnet test

# Run specific test project
dotnet test test/UnitTests/UnitTests.csproj
dotnet test test/FoundationAITests/FoundationAITests.csproj

# Run a single test by name
dotnet test --filter "LLMTests.ResolverWithToolCall"

# Package as NuGet
dotnet pack -c Release
# or use the PowerShell build script (also moves output to dest)
./build.ps1 d:\nuget
```

**Target framework:** `net10.0` (SDK 10.0.103 via `global.json`)

## Architecture

### Core Abstractions (`src/RZ.Foundation.AI/`)

**`AiChatFunc`** (defined in `Common.cs`) is the central delegate type:
```csharp
delegate AgentResponse AiChatFunc(IEnumerable<ChatMessage> messages);
// AgentResponse = ValueTask<Outcome<(IReadOnlyList<ChatEntry>, ChatCost)>>
```
Everything—OpenAI, Gemini, and the LLM resolver—is just an `AiChatFunc`.

**`LLM.CreateResolver`** (`LLM.cs`) wraps an `AiChatFunc` with automatic tool execution: it calls the underlying chat, detects `ChatMessage.ToolCall` responses, executes the tools in parallel via `ToolWrapper`, then calls chat again with tool results appended.

**`ToolWrapper`** (`ToolWrapper.cs`) discovers tools via reflection from a type. Mark methods with `[AiToolName]` (optionally with a custom name). Supported parameter types: `string`, `int`/`double`/`decimal`, `bool`, `enum`, and `Nullable<T>` of those. Use `[Description]` on methods and parameters for AI documentation.

**Message types** (`Common.cs`):
- `ChatMessage.Content` — plain text with a `ChatRole`
- `ChatMessage.MultiContent` — multi-modal (images, audio, files, URIs)
- `ChatMessage.ToolCall` — AI requesting tool invocations
- `ChatMessage.ToolResult` — result of a tool call

**`ChatEntry`** wraps a `ChatMessage` with timestamp, admin metadata, and `ChatCost`.

**Cost tracking:** `ChatCost` holds input/output costs in Satang (1 USD = 3800 Satang). `LLM.CalcCost` uses `CostStructure` rate tables with optional tiered pricing (`InputThreshold`) and thought-token costs.

### AI Service Wrappers

- **`OpenAi`** (`OpenAi.cs`): Wraps the `OpenAI` SDK. Call `new OpenAi(apiKey).CreateModel(OpenAi.gpt_4o_mini)` to get an `AiChatFunc`. Supports gpt-4o, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano variants.
- **`GeminiAi`** (`GeminiAi.cs`): Wraps `Google_GenerativeAI` SDK, inherits from `BaseModel`. Supports gemini-2.0-flash-lite, gemini-2.0-flash, gemini-2.5-flash-preview (with extended thinking).

### Error Handling

Uses `Outcome<T>` from `RZ.Foundation.Validation` throughout — no exceptions in the happy path. Helper methods `Fail(...)`, `IfSome(...)`, `Success(...)` come from global usings. Return errors via `new ErrorInfo(code, message)`.

## Test Structure

- **`test/UnitTests/`** (TUnit): Pure unit tests with mocked `AiChatFunc` delegates — no API keys needed. Run these for local development.
- **`test/FoundationAITests/`** (xUnit v3): Integration tests hitting real APIs. Tests are skipped by default (`RunTests = false`); requires API keys at runtime.

## Key Patterns

- `ToolWrapper.FromType(typeof(MyToolClass))` to register all `[AiToolName]`-marked static methods as tools.
- `AgentCommonParameters.Focused` preset (`Temperature = 0.01f, TopP = 0.1f`) for deterministic outputs.
- `Historian.cs` manages multi-turn conversation context automatically.
- `SharedHttp.cs` provides a lazy singleton `HttpClient`.
