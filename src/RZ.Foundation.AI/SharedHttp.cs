namespace RZ.Foundation.AI;

public static class SharedHttp
{
    static readonly Lazy<HttpClient> client = new();
    public static HttpClient Client => client.Value;
}