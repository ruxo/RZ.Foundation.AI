using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace RZ.Foundation.AI;

[PublicAPI]
public readonly record struct ToolWrapper(ToolDefinition Definition, object? Tool, MethodInfo Method)
{
    /// <summary>
    /// Creates tool wrappers from all methods of the given object that have AI tool names.
    /// </summary>
    /// <param name="tool">The object containing the methods to wrap.</param>
    /// <param name="nameGetter">Optional function to resolve tool names. If null, uses <see cref="AiToolNameAttribute"/> or method name.</param>
    /// <returns>A list of tool wrappers for the object's methods.</returns>
    [Pure]
    public static IReadOnlyList<ToolWrapper> From(object tool, Func<MethodInfo, string?>? nameGetter = null)
        => From(tool,
                tool.GetType().GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.FlattenHierarchy),
                nameGetter);

    /// <summary>
    /// Creates tool wrappers from all static methods of the specified type that have AI tool names.
    /// </summary>
    /// <param name="t">The type containing the static methods to wrap.</param>
    /// <param name="nameGetter">Optional function to resolve tool names. If null, uses <see cref="AiToolNameAttribute"/> or method name.</param>
    /// <returns>A list of tool wrappers for the type's static methods.</returns>
    [Pure]
    public static IReadOnlyList<ToolWrapper> FromType(Type t, Func<MethodInfo, string?>? nameGetter = null)
        => From(null,
                t.GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.FlattenHierarchy),
                nameGetter);

    /// <summary>
    /// Creates tool wrappers from the static methods of a specified type.
    /// </summary>
    /// <param name="nameGetter">Function that resolves tool names. By default uses <see cref="AiToolNameAttribute"/> or the method name.</param>
    /// <typeparam name="T">The type containing the static methods to wrap.</typeparam>
    /// <returns>A list of tool wrappers for the type's static methods.</returns>
    [Pure]
    public static IReadOnlyList<ToolWrapper> FromType<T>(Func<MethodInfo, string?>? nameGetter = null)
        => FromType(typeof(T), nameGetter);

    /// <summary>
    /// Creates tool wrappers from a specified list of methods that have AI tool names.
    /// </summary>
    /// <param name="tool">The object instance for non-static methods, or null for static methods.</param>
    /// <param name="methods">The list of methods to create wrappers for.</param>
    /// <param name="nameGetter">Optional function to resolve tool names. If null, uses <see cref="AiToolNameAttribute"/> or method name.</param>
    /// <returns>A list of tool wrappers for the specified methods.</returns>
    [Pure]
    public static IReadOnlyList<ToolWrapper> From(object? tool, IReadOnlyList<MethodInfo> methods, Func<MethodInfo, string?>? nameGetter = null)
        => (from method in methods
            let name = (nameGetter ?? GetAiToolName)(method)
            where name is not null
            let toolDef = ToolDefinition.From(name, method)
            select new ToolWrapper(toolDef, method.IsStatic ? null : tool, method)
           ).ToArray();

    /// <summary>
    /// Retrieves the AI tool name for a method, either from its AiToolNameAttribute or the method name itself.
    /// </summary>
    /// <param name="method">The method to get the AI tool name for.</param>
    /// <returns>The AI tool name if available, otherwise null.</returns>
    [Pure]
    public static string? GetAiToolName(MethodInfo method)
        => method.GetCustomAttribute<AiToolNameAttribute>() is { } attr ? attr.Name ?? method.Name : null;

    /// <summary>
    /// Parses the JSON payload into an array of parameter values matching the method's signature.
    /// </summary>
    /// <param name="payload">The JSON payload containing parameter values.</param>
    /// <returns>An array of parameter values ready to be passed to the method.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when an enum value is invalid.</exception>
    /// <exception cref="ErrorInfoException">Thrown when a required parameter is missing.</exception>
    [Pure]
    public Outcome<object?[]> ParseParameters(JsonNode? payload) {
        if (Success(ParsePayload(payload).Map(x => x.CheckNotFound()).MakeMutableList(x => x.ToNullable()), out var values, out var e))
            return SuccessOutcome(values.ToArray());
        else
            return e;
    }

    IEnumerable<Outcome<object>> ParsePayload(JsonNode? payload) {
        if ((payload as JsonObject) is not { } pObj) yield break;

        foreach (var pinfo in Method.GetParameters()){
            if (IfSome(Definition.Parameters.TrySingle(x => x.Name == pinfo.Name), out var p)){
                if (!pObj.ContainsKey(p.Name) || FailButNotFound(GetValue(pinfo.ParameterType, pObj[p.Name]!), out var e, out var v))
                    if (p.DefaultValue is null)
                        yield return new ErrorInfo(InvalidRequest, $"Missing parameter: {p.Name}");
                    else
                        yield return WrapOutcome(p.DefaultValue.Value.ToNullable());
                else{
                    var pvalue = e?.IsNotFound() == true ? null : v;
                    if (p.Type is ToolParameterType.EnumType)
                        if (pvalue is not null && Enum.TryParse(pinfo.ParameterType, pvalue.ToString(), out var ev))
                            yield return ev;
                        else{
                            yield return new ErrorInfo(InvalidRequest, $"Invalid enum value for parameter '{p.Name}': {pvalue}");
                            yield break;
                        }
                    else
                        yield return WrapOutcome(pvalue);
                }
            }
            else{
                yield return new ErrorInfo(InvalidRequest, $"Invalid parameter: {pinfo.Name}");
                yield break;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Outcome<object> WrapOutcome(object? v)
        => v is null ? FailedOutcome<object>(ErrorInfo.NotFound) : SuccessOutcome(v);

    /// <summary>
    /// Invokes the wrapped method with the provided arguments and handles async/sync return values.
    /// </summary>
    /// <param name="args">The arguments to pass to the method.</param>
    /// <returns>The result of the method invocation.</returns>
    public async ValueTask<Outcome<object>> Call(object?[] args) {
        object? result;
        try{
            result = Method.Invoke(Tool, args);
        }
        catch (Exception e){
            return ErrorFrom.Exception(e)
                            .Trace($"Failed calling method {Method.DeclaringType?.Name}.{Method.Name} with args: {Seq(args)}");
        }
        return result is null
                   ? FailedOutcome<object>(new ErrorInfo(ValidationFailed, $"Tool {Method.DeclaringType?.Name}.{Method.Name} returned null"))
                   : result switch {
                       // TODO: catch lambda functions
                       Task t when t.GetType().IsGenericType        => await Expression.Lambda<Func<ValueTask<Outcome<object>>>>(Expression.Call(typeof(ToolWrapper).GetMethod(nameof(ConvertTask), BindingFlags.Static | BindingFlags.NonPublic)!.MakeGenericMethod(t.GetType().GetGenericArguments()[0]), Expression.Constant(t))).Compile()(),
                       ValueTask vt when vt.GetType().IsGenericType => await Expression.Lambda<Func<ValueTask<Outcome<object>>>>(Expression.Call(typeof(ToolWrapper).GetMethod(nameof(ConvertValueTask), BindingFlags.Static | BindingFlags.NonPublic)!.MakeGenericMethod(vt.GetType().GetGenericArguments()[0]), Expression.Constant(vt))).Compile()(),

                       _ => result
                   };
    }

    static async ValueTask<Outcome<object>> ConvertTask<T>(Task<T> task) {
        try{
            return await task is { } v ? v : ErrorInfo.NotFound;
        }
        catch (Exception e){
            return ErrorFrom.Exception(e);
        }
    }

    static async ValueTask<object> ConvertValueTask<T>(ValueTask<T> task) {
        try{
            return await task is { } v ? v : ErrorInfo.NotFound;
        }
        catch (Exception e){
            return ErrorFrom.Exception(e);
        }
    }

    static Outcome<object> GetValue(Type propType, JsonNode? jv)
        => jv is null
               ? ErrorInfo.NotFound
               : jv.GetValueKind() switch {
                   JsonValueKind.String                      => jv.GetValue<string>(),
                   JsonValueKind.Number                      => GetNumberValue(propType, jv),
                   JsonValueKind.True or JsonValueKind.False => jv.GetValue<bool>(),

                   _ => new ErrorInfo(Unhandled, $"Unsupported value kind: {jv.GetValueKind()}")
               };

    static Outcome<object> GetNumberValue(Type t, JsonNode jv) {
        var propType = t.IsGenericType && t.GetGenericTypeDefinition() == typeof(Nullable<>) ? t.GenericTypeArguments[0] : t;
        return propType == typeof(int)       ? jv.GetValue<int>()
               : propType == typeof(double)  ? jv.GetValue<double>()
               : propType == typeof(decimal) ? jv.GetValue<decimal>()
                                               : new ErrorInfo(Unhandled, $"Unsupported type: {t.Name}");
    }
}