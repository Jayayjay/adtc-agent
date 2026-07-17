"""
Lightweight argument validation against a tool's declared parameter spec
(src/tools/base.py's `parameters` dict). Not a full JSON Schema implementation
-- just enough to catch obviously wrong LLM-generated tool calls before they
hit real code (e.g. filesystem writes).
"""

from __future__ import annotations

from src.core.exceptions import ToolExecutionError

_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def validate_arguments(tool_name: str, parameters_spec: dict, provided: dict) -> None:
    """Raises ToolExecutionError on the first validation failure found."""
    for param_name, spec in parameters_spec.items():
        expected_type = _TYPE_MAP.get(spec.get("type"))
        is_required = "default" not in spec

        if param_name not in provided:
            if is_required:
                raise ToolExecutionError(
                    tool_name, f"Missing required argument '{param_name}'"
                )
            continue

        value = provided[param_name]
        if expected_type and not isinstance(value, expected_type):
            raise ToolExecutionError(
                tool_name,
                f"Argument '{param_name}' expected type {spec.get('type')}, "
                f"got {type(value).__name__}",
            )

        allowed = spec.get("enum")
        if allowed and value not in allowed:
            raise ToolExecutionError(
                tool_name, f"Argument '{param_name}'={value!r} not in allowed values {allowed}"
            )
