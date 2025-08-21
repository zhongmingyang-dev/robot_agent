
import asyncio
from fastmcp import Client


client = Client("http://localhost:8000/mcp")


async def main():
    async with client:
        # Basic server interaction
        await client.ping()

        # List available operations
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        print("tools: ", tools)
        print("resources: ", resources)
        print("prompts: ", prompts)

        # Execute operations
        # result = await client.call_tool("example_tool", {"param": "value"})
        # print(result)


asyncio.run(main())
