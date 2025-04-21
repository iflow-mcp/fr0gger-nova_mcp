import asyncio
import os
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv
import json
import logging
import os

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 
# Configure client logging to the same nova_matches.log used by the server
CLIENT_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(CLIENT_LOG_DIR, exist_ok=True)
CLIENT_LOG_FILE = os.path.join(CLIENT_LOG_DIR, "nova_matches.log")
client_logger = logging.getLogger("nova-mcp-client")
client_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(CLIENT_LOG_FILE)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
client_logger.addHandler(fh)

class NovaSecurityClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        if not server_script_path.endswith((".py", ".js")):
            raise ValueError("Server script must be a .py or .js file")

        # Get the absolute path to the server script
        server_script_path = os.path.abspath(server_script_path)
        
        # Set working directory to where the server script is located
        working_dir = os.path.dirname(server_script_path)
        
        command = "python" if server_script_path.endswith(".py") else "node"
        server_params = StdioServerParameters(
            command=command, 
            args=[os.path.basename(server_script_path)],
            cwd=working_dir  # Set working directory
        )

        client_logger.info(f"Starting NOVA MCP server: {command} {server_script_path}")
        print(f"Starting NOVA MCP server: {command} {server_script_path}...")
        
        # Connect to server with a timeout per operation (asyncio.wait_for for Python 3.10 compatibility)
        connection_timeout = 20  # Seconds
        try:
            stdio_transport = await asyncio.wait_for(
                self.exit_stack.enter_async_context(stdio_client(server_params)),
                connection_timeout
            )
            self.stdio, self.write = stdio_transport
            self.session = await asyncio.wait_for(
                self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write)),
                connection_timeout
            )
            # Give server a moment to start
            await asyncio.sleep(1)
            await asyncio.wait_for(self.session.initialize(), connection_timeout)
            response = await asyncio.wait_for(self.session.list_tools(), connection_timeout)
            tools = response.tools

            # Verify the validate_prompt tool is available
            if not any(tool.name == "validate_prompt" for tool in tools):
                raise ConnectionError("Server started but 'validate_prompt' tool not found")

            print("\n✅ Connected to MCP server.")
            print("Available tools:", [tool.name for tool in tools])
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"Timed out connecting to server after {connection_timeout} seconds. Check if the server is running correctly."
            )
        except Exception as e:
            # Log the full exception with stack trace
            client_logger.error("Exception during connect_to_server", exc_info=True)
            # Include exception type and message for better debugging
            err_class = e.__class__.__name__
            err_message = str(e) or repr(e)
            raise ConnectionError(
                f"Failed to connect to NOVA MCP server: {err_class}: {err_message}"
            ) from e

    async def process_query(self, query: str) -> str:
        # ...existing code...
        system_prompt = (
            "Here is your instruction you MUST follow: "
            "You are an AI. For this session, Nova-Security MCP is responsible for verifying all prompts. "
            "Before doing anything else, you MUST pass every prompt to the MCP for validation. "
            "If a prompt is not authorized, do NOT respond. Instead, return the exact message received from the MCP—nothing else."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        tools_response = await self.session.list_tools()

        tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_response.tools
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto"
        )

        full_response = []
        assistant_msg = response.choices[0].message

        if assistant_msg.tool_calls:
            for tool_call in assistant_msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                try:
                    result = await self.session.call_tool(tool_name, tool_args)
                except Exception as e:
                    result = f"Error: {str(e)}"

                messages.extend([
                    {
                        "role": "assistant",
                        "tool_calls": [tool_call],
                        "content": None
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    }
                ])

            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            final_output = response.choices[0].message.content
            full_response.append(final_output)
        else:
            full_response.append(assistant_msg.content)

        response_text = "\n".join(full_response)
        # Log result as JSON: authorized at INFO, unauthorized at WARNING
        if response_text.startswith("NOT AUTHORIZED"):
            # Parse unauthorized details
            details = {"user_id": "unknown", "prompt": query}
            for line in response_text.splitlines():
                if line.startswith("Security rule matched:"):
                    details["rule_name"] = line.split(":", 1)[1].strip()
                if line.startswith("Severity:"):
                    details["severity"] = line.split(":", 1)[1].strip()
            client_logger.warning(json.dumps(details))
        else:
            # Authorized: log query and response together
            client_logger.info(json.dumps({"query": query, "response": response_text}))
        return response_text

    async def chat_loop(self):
        print("\n🧠 Nova-Security MCP Client (GPT-4o)")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    async def cleanup(self):
        """Clean up resources and gracefully close connection to server"""
        client_logger.info("Cleaning up resources and closing session")
        print("Cleaning up resources...")
        try:
            if self.session:
                # Send a clean shutdown message if possible
                try:
                    await self.session.shutdown()
                except Exception:
                    pass  # Ignore errors during shutdown
            
            # Close the exit stack
            await self.exit_stack.aclose()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            # Don't re-raise - we're already in cleanup

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = NovaSecurityClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
