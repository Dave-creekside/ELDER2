
import asyncio
import json
import re
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent, ServerCapabilities

def extract_iterations_from_text(user_input: str) -> int:
    """Extract number of iterations from user input"""
    if not user_input:
        return 3
    
    user_lower = user_input.lower()
    
    # Look for number patterns
    patterns = [
        r'(\d+)\s*iterations?',
        r'(\d+)\s*times?', 
        r'for\s+(\d+)',
        r'dream\s+(\d+)',
        r'(\d+)\s*turns?',
        r'(\d+)\s*rounds?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_lower)
        if match:
            try:
                num = int(match.group(1))
                if num >= 1:  # Only ensure it's at least 1
                    return num
            except ValueError:
                continue
    
    return 3  # Default

server = Server("utility")

@server.list_tools()
async def handle_list_tools():
    return [
        Tool(
            name="extract_dream_iterations",
            description="Extract number of dream iterations from user input",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {"type": "string", "description": "User's dream request"}
                },
                "required": ["user_input"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    if name == "extract_dream_iterations":
        user_input = arguments.get("user_input", "")
        iterations = extract_iterations_from_text(user_input)
        return [TextContent(type="text", text=str(iterations))]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="utility",
                server_version="1.0.0",
                capabilities=ServerCapabilities(tools={})
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
