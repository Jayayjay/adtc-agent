"""
Filesystem tool: read/write/list within a sandboxed root directory.

Sandboxing matters here more than in most of this codebase -- this is the
one tool that touches the user's real filesystem, so path traversal outside
the allowed root must be blocked, not just discouraged.
"""

from __future__ import annotations

from pathlib import Path

from src.core.exceptions import ToolExecutionError
from src.tools.base import BaseTool


class FilesystemTool(BaseTool):
    name = "filesystem"
    description = "Read, write, or list files within the agent's sandboxed working directory."
    parameters = {
        "action": {"type": "string", "enum": ["read", "write", "list"]},
        "path": {"type": "string", "description": "Relative path within the sandbox root."},
        "content": {"type": "string", "description": "Content to write (only for 'write')."},
    }

    def __init__(self, sandbox_root: str | Path):
        self.sandbox_root = Path(sandbox_root).resolve()
        self.sandbox_root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self.sandbox_root / relative_path).resolve()
        if not str(candidate).startswith(str(self.sandbox_root)):
            raise ToolExecutionError(
                self.name, f"Path '{relative_path}' escapes the sandbox root."
            )
        return candidate

    def run(self, action: str, path: str, content: str | None = None) -> dict:
        target = self._resolve(path)

        if action == "read":
            if not target.exists():
                raise ToolExecutionError(self.name, f"File not found: {path}")
            return {"content": target.read_text()}

        if action == "write":
            if content is None:
                raise ToolExecutionError(self.name, "'content' is required for write.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return {"status": "written", "path": str(target.relative_to(self.sandbox_root))}

        if action == "list":
            if not target.exists():
                raise ToolExecutionError(self.name, f"Directory not found: {path}")
            entries = [p.name for p in target.iterdir()]
            return {"entries": sorted(entries)}

        raise ToolExecutionError(self.name, f"Unknown action: {action}")
