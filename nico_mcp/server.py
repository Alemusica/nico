#!/usr/bin/env python3
"""
NICO MCP Server - Context Gate for AI Agents

This MCP server provides tools that MUST be called before any action.
It ensures agents have full context of the project.

Usage:
    python -m mcp.server

Tools provided:
    - read_gate: Read the full project context (MANDATORY before any action)
    - get_file_map: Get current file statistics
    - check_before_edit: Verify context before editing a file
    - update_gate: Update the GATE.md file
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("MCP SDK not installed. Install with: pip install mcp")
    raise

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
GATE_FILE = PROJECT_ROOT / "GATE.md"


def get_file_stats() -> dict:
    """Get line counts for key files."""
    key_files = [
        "app/components/tabs.py",
        "src/services/export_service.py",
        "src/services/slcci_service.py",
        "src/services/cmems_l4_service.py",
        "src/services/gebco_service.py",
        "src/services/transport_service.py",
    ]
    
    stats = {}
    for rel_path in key_files:
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            with open(full_path) as f:
                stats[rel_path] = len(f.readlines())
        else:
            stats[rel_path] = 0
    return stats


def get_export_functions() -> list:
    """Get list of export functions in export_service.py."""
    export_file = PROJECT_ROOT / "src/services/export_service.py"
    functions = []
    if export_file.exists():
        with open(export_file) as f:
            for i, line in enumerate(f, 1):
                if line.startswith("def export_"):
                    func_name = line.split("(")[0].replace("def ", "")
                    functions.append({"line": i, "name": func_name})
    return functions


def get_git_status() -> dict:
    """Get current git status."""
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            text=True
        ).strip()
        
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            text=True
        ).strip()
        
        return {"branch": branch, "changed_files": status.split("\n") if status else []}
    except:
        return {"branch": "unknown", "changed_files": []}


def read_gate_content() -> str:
    """Read and enhance GATE.md with live data."""
    
    # Read base GATE.md
    if GATE_FILE.exists():
        with open(GATE_FILE) as f:
            gate_content = f.read()
    else:
        gate_content = "# GATE.md not found!"
    
    # Add live statistics
    file_stats = get_file_stats()
    export_funcs = get_export_functions()
    git_status = get_git_status()
    
    live_section = f"""

# 🔴 LIVE DATA (Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

## Git Status
- **Branch**: `{git_status['branch']}`
- **Changed files**: {len(git_status['changed_files'])}

## File Statistics (Live)
| File | Lines |
|------|-------|
"""
    for path, lines in file_stats.items():
        live_section += f"| `{path}` | {lines} |\n"
    
    live_section += f"""
## Export Functions ({len(export_funcs)} total)
| Line | Function |
|------|----------|
"""
    for func in export_funcs:
        live_section += f"| {func['line']} | `{func['name']}` |\n"
    
    return gate_content + live_section


# Initialize MCP Server
server = Server("nico-gate")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="read_gate",
            description="""🚨 MANDATORY: Read the project context gate before ANY action.
            
This tool returns:
- Complete project architecture
- Current state and known issues
- File map with line counts
- Live git status and file statistics

ALWAYS call this tool FIRST in every session.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_file_map",
            description="Get current file statistics and export function list.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="check_before_edit",
            description="""Check context before editing a file.
            
Returns relevant information about the file and related components.
ALWAYS call this before using edit tools.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file you want to edit (relative to project root)"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="update_gate",
            description="Update the GATE.md timestamp and optionally add notes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "string",
                        "description": "Optional notes to add to the change log"
                    }
                },
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "read_gate":
        content = read_gate_content()
        return [TextContent(type="text", text=content)]
    
    elif name == "get_file_map":
        stats = get_file_stats()
        funcs = get_export_functions()
        result = {
            "file_statistics": stats,
            "export_functions": funcs,
            "total_export_functions": len(funcs)
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "check_before_edit":
        file_path = arguments.get("file_path", "")
        full_path = PROJECT_ROOT / file_path
        
        info = {
            "file_path": file_path,
            "exists": full_path.exists(),
            "lines": 0,
            "related_files": [],
            "warnings": []
        }
        
        if full_path.exists():
            with open(full_path) as f:
                info["lines"] = len(f.readlines())
        
        # Add context based on file
        if "export_service" in file_path:
            info["related_files"] = ["app/components/tabs.py"]
            info["warnings"] = [
                "Export functions must return bytes",
                "Use 300 DPI for images",
                "Follow existing function pattern"
            ]
            info["existing_functions"] = get_export_functions()
        
        elif "tabs.py" in file_path:
            info["warnings"] = [
                "This file is very large (~7000 lines)",
                "Test changes carefully",
                "Check session state keys"
            ]
        
        return [TextContent(type="text", text=json.dumps(info, indent=2))]
    
    elif name == "update_gate":
        notes = arguments.get("notes", "")
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Update timestamp in GATE.md
        if GATE_FILE.exists():
            with open(GATE_FILE) as f:
                content = f.read()
            
            # Update date
            import re
            content = re.sub(
                r"Last Updated: \d{4}-\d{2}-\d{2}",
                f"Last Updated: {today}",
                content
            )
            
            # Add to change log if notes provided
            if notes:
                log_entry = f"| {today} | Agent | {notes} |"
                content = content.replace(
                    "| 2026-01-26 | Agent | Initial GATE.md creation |",
                    f"| 2026-01-26 | Agent | Initial GATE.md creation |\n{log_entry}"
                )
            
            with open(GATE_FILE, "w") as f:
                f.write(content)
            
            return [TextContent(type="text", text=f"✅ GATE.md updated. Date: {today}")]
        
        return [TextContent(type="text", text="❌ GATE.md not found")]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
