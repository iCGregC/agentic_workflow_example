from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------

mcp = FastMCP("Barebones Agent Server")

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = (PROJECT_ROOT / "workspace").resolve()

WORKSPACE_ROOT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Security helper
# ---------------------------------------------------------------------------

def safe_workspace_path(relative_path: str) -> Path:
    """
    Resolve a user-provided path and ensure it remains inside workspace/.

    This protects against paths such as:
        ../../secrets.txt
        C:\\Users\\GregoryCherry\\.ssh\\id_rsa
    """

    candidate = (WORKSPACE_ROOT / relative_path).resolve()

    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as error:
        raise ValueError(
            "Access denied: path must remain inside the workspace directory."
        ) from error

    return candidate


# ===========================================================================
# TOOLS
# ===========================================================================
#
# Tools perform actions.
#
# FastMCP automatically derives:
#   - tool name
#   - description
#   - JSON input schema
#   - parameter validation
#
# from the function name, docstring, and Python type annotations.
# ===========================================================================


@mcp.tool
def add_numbers(a: float, b: float) -> dict[str, float]:
    """
    Add two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        A structured result containing the sum.
    """

    return {
        "a": a,
        "b": b,
        "sum": a + b,
    }


@mcp.tool
def list_workspace_files() -> list[dict[str, Any]]:
    """
    List all files available inside the agent workspace.

    Returns:
        File paths and sizes relative to the workspace directory.
    """

    files: list[dict[str, Any]] = []

    for path in WORKSPACE_ROOT.rglob("*"):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(WORKSPACE_ROOT)),
                    "size_bytes": path.stat().st_size,
                }
            )

    return files


@mcp.tool
def read_workspace_file(relative_path: str) -> dict[str, str]:
    """
    Read a UTF-8 text file from the agent workspace.

    Args:
        relative_path: Path relative to workspace/, such as example.txt.

    Returns:
        The relative path and file contents.
    """

    path = safe_workspace_path(relative_path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {relative_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {relative_path}")

    return {
        "path": relative_path,
        "content": path.read_text(encoding="utf-8"),
    }


@mcp.tool
def write_workspace_file(
    relative_path: str,
    content: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Write a UTF-8 text file inside the agent workspace.

    Args:
        relative_path: Destination relative to workspace/.
        content: Complete text to write.
        overwrite: Whether an existing file may be replaced.

    Returns:
        Information about the written file.

    Important:
        This is a mutating tool. Production applications should require user
        approval before allowing the model to execute it.
    """

    path = safe_workspace_path(relative_path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{relative_path} already exists. Set overwrite=true to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return {
        "path": relative_path,
        "size_bytes": path.stat().st_size,
        "overwritten": overwrite,
    }


@mcp.tool
def search_workspace(query: str) -> list[dict[str, Any]]:
    """
    Search text files in the workspace for a case-insensitive string.

    Args:
        query: Text to locate within workspace files.

    Returns:
        Matching file paths, line numbers, and matching lines.
    """

    normalized_query = query.casefold()
    matches: list[dict[str, Any]] = []

    for path in WORKSPACE_ROOT.rglob("*"):
        if not path.is_file():
            continue

        try:
            contents = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for line_number, line in enumerate(contents.splitlines(), start=1):
            if normalized_query in line.casefold():
                matches.append(
                    {
                        "path": str(path.relative_to(WORKSPACE_ROOT)),
                        "line_number": line_number,
                        "line": line,
                    }
                )

    return matches


# ===========================================================================
# RESOURCES
# ===========================================================================
#
# Resources expose read-only data. A resource is retrieved by URI.
#
# Unlike tools, resources are not inherently actions selected by every model.
# The MCP client normally lists or reads them and decides whether to put their
# content into the model's context.
# ===========================================================================


@mcp.resource(
    "agent://instructions",
    name="AgentInstructions",
    description="Core operating instructions for the barebones agent.",
    mime_type="text/plain",
)
def agent_instructions() -> str:
    """Return the agent's core behavioral instructions."""

    return """
You are a practical software-development agent.

Rules:
1. Use tools only when they are needed.
2. Inspect existing information before modifying files.
3. Never claim a tool succeeded unless its returned result confirms success.
4. Do not access paths outside the approved workspace.
5. Explain proposed destructive or mutating actions before executing them.
6. Give concise final answers that identify completed work and remaining risks.
""".strip()


@mcp.resource(
    "agent://workspace-policy",
    name="WorkspacePolicy",
    description="Security policy for workspace access.",
    mime_type="application/json",
)
def workspace_policy() -> str:
    """Return the current workspace security policy."""

    return json.dumps(
        {
            "workspace_root": str(WORKSPACE_ROOT),
            "allow_read": True,
            "allow_create": True,
            "allow_overwrite": "requires explicit approval",
            "allow_delete": False,
            "allow_outside_workspace": False,
        },
        indent=2,
    )


@mcp.resource(
    "file://workspace/{relative_path}",
    name="WorkspaceFile",
    description="Read a text file from the approved workspace.",
    mime_type="text/plain",
)
def workspace_file_resource(relative_path: str) -> str:
    """
    Return a workspace file as a parameterized resource.

    Example URI:
        file://workspace/example.txt
    """

    path = safe_workspace_path(relative_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(relative_path)

    return path.read_text(encoding="utf-8")


# ===========================================================================
# PROMPTS
# ===========================================================================
#
# Prompts are server-managed, parameterized message templates.
#
# The MCP client retrieves them explicitly and can provide the resulting
# messages to the LLM.
# ===========================================================================


@mcp.prompt
def coding_task(
    task: str,
    constraints: str = "Keep the implementation simple and testable.",
) -> str:
    """
    Create a disciplined coding-task prompt.

    Args:
        task: The work that must be completed.
        constraints: Technical or operational constraints.
    """

    return f"""
Complete the following coding task:

TASK
{task}

CONSTRAINTS
{constraints}

PROCESS
1. Inspect relevant files before modifying anything.
2. Explain the planned change briefly.
3. Make the smallest sufficient change.
4. Validate the result.
5. Report modified files, validation results, and unresolved risks.
""".strip()


@mcp.prompt
def review_code(
    relative_path: str,
    focus: str = "correctness, security, and maintainability",
) -> str:
    """
    Create a code-review request for a workspace file.

    Args:
        relative_path: File to review.
        focus: Review criteria.
    """

    return f"""
Review the workspace file "{relative_path}".

Focus on:
{focus}

Use the file-reading tool before forming conclusions. Separate:
- confirmed defects
- potential risks
- optional improvements
""".strip()


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()