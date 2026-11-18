from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastmcp import Client


load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MCP_SERVER = "mcp_server.py"
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = 4096
MAX_AGENT_STEPS = 12


# Tools in this set can run without asking the user.
READ_ONLY_TOOLS = {
    "add_numbers",
    "list_workspace_files",
    "read_workspace_file",
    "search_workspace",
}


# Tools in this set require explicit user approval.
APPROVAL_REQUIRED_TOOLS = {
    "write_workspace_file",
}


@dataclass
class ApprovalDecision:
    approved: bool
    reason: str


# ---------------------------------------------------------------------------
# MCP conversion helpers
# ---------------------------------------------------------------------------

def mcp_tools_to_anthropic_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """
    Convert MCP tool definitions into Anthropic tool definitions.

    MCP and Anthropic both use JSON Schema for tool input, but their Python
    objects expose the fields with slightly different names.
    """

    anthropic_tools: list[dict[str, Any]] = []

    for tool in mcp_tools:
        input_schema = getattr(tool, "inputSchema", None)

        if input_schema is None:
            input_schema = getattr(tool, "input_schema", None)

        if hasattr(input_schema, "model_dump"):
            input_schema = input_schema.model_dump(
                mode="json",
                exclude_none=True,
            )

        if input_schema is None:
            input_schema = {
                "type": "object",
                "properties": {},
            }

        anthropic_tools.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": input_schema,
            }
        )

    return anthropic_tools


def serialize_mcp_result(result: Any) -> str:
    """
    Convert a FastMCP CallToolResult into text that can be returned to Claude.
    """

    if result.data is not None:
        try:
            return json.dumps(
                result.data,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        except TypeError:
            return str(result.data)

    text_parts: list[str] = []

    for block in result.content:
        text = getattr(block, "text", None)

        if text is not None:
            text_parts.append(text)
        else:
            text_parts.append(str(block))

    if text_parts:
        return "\n".join(text_parts)

    return json.dumps(
        {
            "is_error": result.is_error,
            "message": "The tool returned no displayable content.",
        }
    )


async def read_text_resource(client: Client, uri: str) -> str:
    """Read and combine the text blocks returned by an MCP resource."""

    contents = await client.read_resource(uri)
    text_parts: list[str] = []

    for item in contents:
        text = getattr(item, "text", None)

        if text is not None:
            text_parts.append(text)

    return "\n".join(text_parts)


async def render_prompt(
    client: Client,
    name: str,
    arguments: dict[str, Any],
) -> str:
    """Retrieve a rendered MCP prompt and flatten its messages into text."""

    result = await client.get_prompt(name, arguments)
    rendered_messages: list[str] = []

    for message in result.messages:
        role = str(message.role)
        content = message.content
        text = getattr(content, "text", str(content))

        rendered_messages.append(f"{role.upper()}:\n{text}")

    return "\n\n".join(rendered_messages)


# ---------------------------------------------------------------------------
# Approval boundary
# ---------------------------------------------------------------------------

async def obtain_approval(
    tool_name: str,
    arguments: dict[str, Any],
) -> ApprovalDecision:
    """
    Ask the user before executing a mutating tool.

    In a web application, replace input() with an approval interface.
    """

    print("\nApproval required")
    print(f"Tool: {tool_name}")
    print("Arguments:")
    print(json.dumps(arguments, indent=2, ensure_ascii=False))

    answer = await asyncio.to_thread(
        input,
        "Execute this tool? [y/N]: ",
    )

    approved = answer.strip().casefold() in {"y", "yes"}

    if approved:
        return ApprovalDecision(
            approved=True,
            reason="The user explicitly approved the tool call.",
        )

    return ApprovalDecision(
        approved=False,
        reason="The user rejected or did not approve the tool call.",
    )


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

async def execute_tool_call(
    mcp_client: Client,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[bool, str]:
    """
    Enforce authorization policy and execute an MCP tool.
    """

    allowed_tools = READ_ONLY_TOOLS | APPROVAL_REQUIRED_TOOLS

    if tool_name not in allowed_tools:
        return (
            True,
            f"Tool execution denied: {tool_name!r} is not on the allowlist.",
        )

    if tool_name in APPROVAL_REQUIRED_TOOLS:
        decision = await obtain_approval(tool_name, arguments)

        if not decision.approved:
            return (
                True,
                f"Tool execution denied. {decision.reason}",
            )

    try:
        result = await mcp_client.call_tool(tool_name, arguments)

        return (
            bool(result.is_error),
            serialize_mcp_result(result),
        )

    except Exception as error:
        return (
            True,
            f"Tool execution failed: {type(error).__name__}: {error}",
        )


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def run_agent(
    anthropic_client: AsyncAnthropic,
    mcp_client: Client,
    user_request: str,
    extra_prompt: str | None = None,
) -> str:
    """
    Run the model -> tool -> model loop until Claude produces a final answer.
    """

    instructions = await read_text_resource(
        mcp_client,
        "agent://instructions",
    )

    system_prompt = instructions

    if extra_prompt:
        system_prompt += f"\n\nTASK-SPECIFIC INSTRUCTIONS\n{extra_prompt}"

    discovered_tools = await mcp_client.list_tools()
    anthropic_tools = mcp_tools_to_anthropic_tools(discovered_tools)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": user_request,
        }
    ]

    for step in range(1, MAX_AGENT_STEPS + 1):
        print(f"\nAgent step {step}/{MAX_AGENT_STEPS}")

        response = await anthropic_client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=anthropic_tools,
            messages=messages,
        )

        # Preserve Claude's complete response in conversation history.
        messages.append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )

        tool_uses = [
            block
            for block in response.content
            if block.type == "tool_use"
        ]

        # No tool request means Claude has completed the turn.
        if not tool_uses:
            final_text = "\n".join(
                block.text
                for block in response.content
                if block.type == "text"
            )

            return final_text or "(The model returned no text.)"

        tool_results: list[dict[str, Any]] = []

        for tool_use in tool_uses:
            print(f"Requested tool: {tool_use.name}")
            print(
                json.dumps(
                    tool_use.input,
                    indent=2,
                    ensure_ascii=False,
                )
            )

            is_error, output = await execute_tool_call(
                mcp_client=mcp_client,
                tool_name=tool_use.name,
                arguments=tool_use.input,
            )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": output,
                    "is_error": is_error,
                }
            )

        # Tool results must be returned in the next user message.
        messages.append(
            {
                "role": "user",
                "content": tool_results,
            }
        )

    return (
        f"Agent stopped after {MAX_AGENT_STEPS} steps to prevent an "
        "unbounded tool-calling loop."
    )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Add it to your .env file."
        )

    anthropic_client = AsyncAnthropic()

    # Passing a Python filename tells FastMCP to launch the server over stdio.
    async with Client(MCP_SERVER) as mcp_client:
        await mcp_client.ping()

        print(f"Connected to MCP server: {MCP_SERVER}")
        print(f"Using model: {MODEL}")
        print("Type 'exit' to stop.")

        while True:
            user_request = await asyncio.to_thread(
                input,
                "\nYou: ",
            )

            if user_request.strip().casefold() in {"exit", "quit"}:
                break

            if not user_request.strip():
                continue

            # Demonstrates using an MCP-managed prompt.
            task_prompt = await render_prompt(
                mcp_client,
                "coding_task",
                {
                    "task": user_request,
                    "constraints": (
                        "Work only inside the approved workspace. "
                        "Require approval before writing files."
                    ),
                },
            )

            answer = await run_agent(
                anthropic_client=anthropic_client,
                mcp_client=mcp_client,
                user_request=user_request,
                extra_prompt=task_prompt,
            )

            print(f"\nAgent:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())