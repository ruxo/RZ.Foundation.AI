using System.Linq.Expressions;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using RZ.Foundation.Types;
using PureAttribute = System.Diagnostics.Contracts.PureAttribute;

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
            select new ToolWrapper(toolDef, method.IsStatic? null : tool, method)
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
    public object?[] ParseParameters(JsonNode? payload) {
        var pdef = Definition.Parameters;
        var parameters = from pinfo in Method.GetParameters()
                         let p = pdef.Single(x => x.Name == pinfo.Name)
                         let valueFromParameters = from parameter in Optional(payload![p.Name])
                                                   from pvalue in Optional(GetValue(pinfo.ParameterType, parameter.AsValue()))
                                                   select p.Type is ToolParameterType.EnumType
                                                              ? Enum.TryParse(pinfo.ParameterType, pvalue.ToString(), out var ev) ? ev : throw new ArgumentOutOfRangeException($"{pvalue} is not a valid literal for enum type {p.Name}")
                                                              : pvalue
                         let value = valueFromParameters.ToNullable()
                                  ?? (p.DefaultValue ?? throw new ErrorInfoException(StandardErrorCodes.InvalidRequest, $"Missing parameter: {p.Name}")).ToNullable()
                         select value;
        return parameters.ToArray();
    }

    /// <summary>
    /// Invokes the wrapped method with the provided arguments and handles async/sync return values.
    /// </summary>
    /// <param name="args">The arguments to pass to the method.</param>
    /// <returns>The result of the method invocation.</returns>
    /// <exception cref="NotSupportedException">Thrown when the method returns null or has an unsupported return type.</exception>
    public async ValueTask<object> Call(params object?[] args) {
        var result = Method.Invoke(Tool, args);
        return (result switch {
                       Task t when t.GetType().IsGenericType        => await Expression.Lambda<Func<Task<object?>>>(Expression.Call(typeof(ToolWrapper).GetMethod(nameof(ConvertTask), BindingFlags.Static | BindingFlags.NonPublic)!.MakeGenericMethod(t.GetType().GetGenericArguments()[0]), Expression.Constant(t))).Compile()(),
                       ValueTask vt when vt.GetType().IsGenericType => await Expression.Lambda<Func<Task<object?>>>(Expression.Call(typeof(ToolWrapper).GetMethod(nameof(ConvertValueTask), BindingFlags.Static | BindingFlags.NonPublic)!.MakeGenericMethod(vt.GetType().GetGenericArguments()[0]), Expression.Constant(vt))).Compile()(),

                       null => throw new NotSupportedException($"Tool {Method.DeclaringType?.Name}.{Method.Name} returned null"),
                       _    => result
                   })!;
    }

    static async Task<object> ConvertTask<T>(Task<T> task)           => (await task)!;
    static async Task<object> ConvertValueTask<T>(ValueTask<T> task) => (await task)!;

    static object? GetValue(Type propType, JsonValue jv)
        => jv.GetValueKind() switch {
            JsonValueKind.Null   => null,
            JsonValueKind.String => jv.GetValue<string>(),
            JsonValueKind.Number => GetNumberValue(propType, jv),
            JsonValueKind.True or JsonValueKind.False => jv.GetValue<bool>(),

            _ => throw new NotSupportedException($"Unsupported value kind: {jv.GetValueKind()}")
        };

    static object GetNumberValue(Type t, JsonValue jv) {
        var propType = t.IsGenericType && t.GetGenericTypeDefinition() == typeof(Nullable<>) ? t.GenericTypeArguments[0] : t;
        return propType == typeof(int)       ? jv.GetValue<int>()
               : propType == typeof(double)  ? jv.GetValue<double>()
               : propType == typeof(decimal) ? jv.GetValue<decimal>()
                                               : throw new NotSupportedException($"Unsupported type: {t.Name}");
    }
}