import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configuration
API_KEY = os.getenv("REGISTRY_API_KEY", "your-api-key-here")
SERVER_SCRIPT = "catapult_mcp.server"

async def main():
    print("Starting MCP Integration Test...")
    
    # Define server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["-m", SERVER_SCRIPT],
        env={
            **os.environ,
            "REGISTRY_API_KEY": API_KEY,
            "REGISTRY_URL": "http://localhost/api" 
        }
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("Connected to MCP server.")
                
                # Initialize
                await session.initialize()
                print("Initialized.")

                # List Tools
                print("\n--- Available Tools ---")
                tools = await session.list_tools()
                for tool in tools.tools:
                    print(f"- {tool.name}: {tool.description}")

                # Call Health Check
                print("\n--- Health Check ---")
                try:
                    result = await session.call_tool("health_check", {})
                    print(f"Result: {result}")
                except Exception as e:
                    print(f"Health Check Failed: {e}")

                # Call List Models
                print("\n--- List Models ---")
                try:
                    result = await session.call_tool("list_models", {"limit": 3})
                    print(f"Result: {result}")
                except Exception as e:
                    print(f"List Models Failed: {e}")

    except Exception as e:
        print(f"Test Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
