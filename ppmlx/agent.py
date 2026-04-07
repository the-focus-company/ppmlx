"""Built-in agent runtime for executing tool calls emitted by models.

Implements a generate -> parse tool calls -> execute -> feed back loop
with configurable safety controls.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ── Safety defaults ──────────────────────────────────────────────────────

DEFAULT_COMMAND_ALLOWLIST: list[str] = [
    "ls", "cat", "head", "tail", "find", "grep", "rg", "wc",
    "pwd", "echo", "date", "whoami", "which", "env", "tree",
    "diff", "sort", "uniq", "tr", "cut", "file", "du", "df",
    "uname", "python", "python3", "pip", "uv", "git",
]


# ── Tool result types ────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Result of executing a single tool call."""
    tool_call_id: str
    name: str
    output: str
    is_error: bool = False


@dataclass
class AgentStep:
    """One iteration of the agent loop."""
    iteration: int
    assistant_text: str
    tool_calls: list[dict]
    tool_results: list[ToolResult]


# ── Built-in tool definitions (OpenAI function calling schema) ───────────

BUILTIN_TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at the given path. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory, optionally filtering by a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to list (default: working directory).",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '*.py'). Default: '*'.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

BUILTIN_TOOL_NAMES: set[str] = {
    t["function"]["name"] for t in BUILTIN_TOOL_DEFINITIONS
}


# ── Path validation ─────────────────────────────────────────────────────

def _resolve_path(path_str: str, working_dir: str) -> Path:
    """Resolve a path relative to the working directory."""
    p = Path(path_str)
    if not p.is_absolute():
        p = Path(working_dir) / p
    return p.resolve()


def _validate_path(
    path_str: str,
    working_dir: str,
    allowed_directories: list[str],
) -> Path:
    """Validate that a path is within allowed directories.

    Raises ValueError if the path escapes the allowed directories.
    """
    resolved = _resolve_path(path_str, working_dir)
    for allowed in allowed_directories:
        allowed_resolved = Path(allowed).resolve()
        try:
            resolved.relative_to(allowed_resolved)
            return resolved
        except ValueError:
            continue
    raise ValueError(
        f"Path '{resolved}' is outside allowed directories: "
        f"{[str(Path(d).resolve()) for d in allowed_directories]}"
    )


# ── Built-in tool implementations ────────────────────────────────────────

def _tool_read_file(
    args: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool,
) -> str:
    path_str = args.get("path", "")
    if not path_str:
        return "Error: 'path' argument is required"
    resolved = _validate_path(path_str, working_dir, allowed_directories)
    if not resolved.exists():
        return f"Error: file not found: {resolved}"
    if not resolved.is_file():
        return f"Error: not a file: {resolved}"
    content = resolved.read_text(encoding="utf-8", errors="replace")
    # Truncate very large files
    max_chars = 100_000
    if len(content) > max_chars:
        return content[:max_chars] + f"\n\n... [truncated, {len(content)} chars total]"
    return content


def _tool_write_file(
    args: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool,
) -> str:
    if sandbox:
        return "Error: write_file is disabled in sandbox mode"
    path_str = args.get("path", "")
    content = args.get("content", "")
    if not path_str:
        return "Error: 'path' argument is required"
    resolved = _validate_path(path_str, working_dir, allowed_directories)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to {resolved}"


def _tool_list_files(
    args: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool,
) -> str:
    directory = args.get("directory", working_dir)
    pattern = args.get("pattern", "*")
    resolved = _validate_path(directory, working_dir, allowed_directories)
    if not resolved.exists():
        return f"Error: directory not found: {resolved}"
    if not resolved.is_dir():
        return f"Error: not a directory: {resolved}"
    matches = sorted(resolved.glob(pattern))
    # Limit output
    max_entries = 500
    lines = []
    for i, m in enumerate(matches):
        if i >= max_entries:
            lines.append(f"... and {len(matches) - max_entries} more")
            break
        suffix = "/" if m.is_dir() else ""
        try:
            rel = m.relative_to(resolved)
        except ValueError:
            rel = m
        lines.append(f"{rel}{suffix}")
    return "\n".join(lines) if lines else "(empty directory)"


def _extract_base_command(command: str) -> str:
    """Extract the base command name, skipping env var assignments."""
    for part in command.split():
        if "=" not in part:
            return os.path.basename(part)
    return ""


_SANDBOX_BLOCKED_COMMANDS = frozenset({
    "rm", "rmdir", "mv", "cp", "dd", "mkfs", "chmod", "chown",
})


def _tool_run_command(
    args: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool,
    command_allowlist: list[str] | None = None,
) -> str:
    command = args.get("command", "")
    if not command:
        return "Error: 'command' argument is required"

    base_cmd = _extract_base_command(command)

    if command_allowlist is not None and base_cmd not in command_allowlist:
        return (
            f"Error: command '{base_cmd}' is not in the allowlist. "
            f"Allowed: {', '.join(sorted(command_allowlist))}"
        )

    if sandbox and base_cmd in _SANDBOX_BLOCKED_COMMANDS:
        return f"Error: command '{base_cmd}' is blocked in sandbox mode"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=working_dir,
        )
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")
        if result.returncode != 0:
            output_parts.append(f"[exit code: {result.returncode}]")
        output = "\n".join(output_parts) if output_parts else "(no output)"
        # Truncate
        max_chars = 50_000
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n... [truncated, {len(output)} chars total]"
        return output
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as exc:
        return f"Error executing command: {exc}"


# ── Tool dispatcher ──────────────────────────────────────────────────────

_TOOL_HANDLERS: dict[str, Callable] = {
    "read_file": _tool_read_file,
    "write_file": _tool_write_file,
    "list_files": _tool_list_files,
    "run_command": _tool_run_command,
}


def execute_tool(
    name: str,
    arguments: dict,
    working_dir: str,
    allowed_directories: list[str],
    sandbox: bool = False,
    command_allowlist: list[str] | None = None,
) -> str:
    """Execute a built-in tool by name. Returns the output string."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'"

    kwargs: dict[str, Any] = {
        "args": arguments,
        "working_dir": working_dir,
        "allowed_directories": allowed_directories,
        "sandbox": sandbox,
    }
    if name == "run_command":
        kwargs["command_allowlist"] = command_allowlist
    return handler(**kwargs)


# ── Agent Runtime ────────────────────────────────────────────────────────

def _make_tool_call_id(iteration: int, index: int) -> str:
    return f"call_{iteration}_{index}"


def _build_system_prompt(
    user_system_prompt: str | None,
    tools: list[dict],
) -> str:
    """Build the system prompt that instructs the model to use tools."""
    parts: list[str] = []

    if user_system_prompt:
        parts.append(user_system_prompt)

    parts.append(
        "You are an agent with access to tools. "
        "When you need to use a tool, emit a <tool_call> block with JSON inside:\n"
        "<tool_call>\n"
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "</tool_call>\n\n"
        "You may call multiple tools in one response by emitting multiple "
        "<tool_call> blocks. After each tool call round, you will receive the "
        "results and can decide what to do next.\n\n"
        "When you have completed the task, respond with your final answer "
        "without any tool calls."
    )

    tool_descriptions = []
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        tool_descriptions.append(
            f"- **{name}**: {desc}\n"
            f"  Parameters: {json.dumps(params)}"
        )

    if tool_descriptions:
        parts.append("Available tools:\n" + "\n".join(tool_descriptions))

    return "\n\n".join(parts)


@dataclass
class AgentConfig:
    """Configuration for the agent runtime."""
    model: str
    max_iterations: int = 10
    sandbox: bool = False
    allowed_directories: list[str] = field(default_factory=list)
    command_allowlist: list[str] | None = field(default=None)
    working_dir: str = field(default_factory=os.getcwd)
    enabled_tools: list[str] | None = None  # None = all built-in tools
    temperature: float = 0.7
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.allowed_directories:
            self.allowed_directories = [self.working_dir]
        if self.command_allowlist is None and not self.sandbox:
            self.command_allowlist = list(DEFAULT_COMMAND_ALLOWLIST)
        elif self.command_allowlist is None and self.sandbox:
            # In sandbox, only allow read-only commands
            self.command_allowlist = [
                "ls", "cat", "head", "tail", "find", "grep", "rg", "wc",
                "pwd", "echo", "date", "whoami", "which", "tree",
                "diff", "sort", "uniq", "file", "du", "df", "uname",
            ]


class AgentRuntime:
    """Agent runtime that executes tool calls emitted by models.

    Implements a generate -> parse tool calls -> execute -> feed back loop.
    """

    def __init__(
        self,
        config: AgentConfig,
        engine: Any | None = None,
        on_step: Callable[[AgentStep], None] | None = None,
    ):
        self.config = config
        self._engine = engine
        self._on_step = on_step
        self._tool_defs = self._get_tool_definitions()
        self._enabled_names: set[str] = {
            t["function"]["name"] for t in self._tool_defs
        }

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        from ppmlx.engine import get_engine
        return get_engine()

    def _get_tool_definitions(self) -> list[dict]:
        """Get the tool definitions based on enabled_tools config."""
        if self.config.enabled_tools is None:
            defs = list(BUILTIN_TOOL_DEFINITIONS)
        else:
            enabled = set(self.config.enabled_tools)
            defs = [
                t for t in BUILTIN_TOOL_DEFINITIONS
                if t["function"]["name"] in enabled
            ]
        # In sandbox mode, exclude write_file
        if self.config.sandbox:
            defs = [
                t for t in defs
                if t["function"]["name"] != "write_file"
            ]
        return defs

    def _parse_tool_calls(self, text: str) -> tuple[str, list[dict]]:
        """Parse tool calls from model output. Reuses server logic."""
        from ppmlx.server import _parse_tool_calls
        return _parse_tool_calls(text)

    def _execute_tool_call(
        self, name: str, arguments_str: str, call_id: str,
    ) -> ToolResult:
        """Execute a single tool call and return the result."""
        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except (json.JSONDecodeError, TypeError):
            return ToolResult(
                tool_call_id=call_id,
                name=name,
                output=f"Error: invalid JSON arguments: {arguments_str}",
                is_error=True,
            )

        if name not in self._enabled_names:
            return ToolResult(
                tool_call_id=call_id,
                name=name,
                output=f"Error: tool '{name}' is not available",
                is_error=True,
            )

        try:
            output = execute_tool(
                name=name,
                arguments=arguments,
                working_dir=self.config.working_dir,
                allowed_directories=self.config.allowed_directories,
                sandbox=self.config.sandbox,
                command_allowlist=self.config.command_allowlist,
            )
            is_error = output.startswith("Error:")
        except ValueError as exc:
            output = str(exc)
            is_error = True
        except Exception as exc:
            output = f"Error: {exc}"
            is_error = True

        return ToolResult(
            tool_call_id=call_id,
            name=name,
            output=output,
            is_error=is_error,
        )

    def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, list[AgentStep]]:
        """Run the agent loop.

        Returns (final_answer, steps).
        """
        engine = self._get_engine()
        full_system = _build_system_prompt(system_prompt, self._tool_defs)

        messages: list[dict] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]

        steps: list[AgentStep] = []

        for iteration in range(self.config.max_iterations):
            # Generate response
            result = engine.generate(
                repo_id=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                strip_thinking=True,
            )
            text = result.text

            # Parse tool calls
            remaining_text, tool_calls = self._parse_tool_calls(text)

            if not tool_calls:
                # No tool calls -- this is the final answer
                step = AgentStep(
                    iteration=iteration,
                    assistant_text=remaining_text or text,
                    tool_calls=[],
                    tool_results=[],
                )
                steps.append(step)
                if self._on_step:
                    self._on_step(step)
                return remaining_text or text, steps

            # Execute tool calls
            tool_results: list[ToolResult] = []
            for i, tc in enumerate(tool_calls):
                call_id = _make_tool_call_id(iteration, i)
                result = self._execute_tool_call(
                    name=tc["name"],
                    arguments_str=tc["arguments"],
                    call_id=call_id,
                )
                tool_results.append(result)

            step = AgentStep(
                iteration=iteration,
                assistant_text=remaining_text,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            steps.append(step)
            if self._on_step:
                self._on_step(step)

            # Build assistant message with tool call blocks (for context)
            messages.append({"role": "assistant", "content": text})

            # Feed tool results back as tool messages
            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "name": tr.name,
                    "tool_call_id": tr.tool_call_id,
                    "content": tr.output,
                })

        # Hit max iterations -- return whatever we have
        final_text = steps[-1].assistant_text if steps else "Max iterations reached without a final answer."
        return final_text, steps
