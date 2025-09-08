# Start MCP server for database access

export def main [
    --port(-p): int = 8000  # Server port
    --host(-h): string = "localhost"  # Server host
] {
    print $"🚀 Starting MCP server on ($host):($port)..."
    python trigent-mcp/mcp_server.py --host $host --port $port
}