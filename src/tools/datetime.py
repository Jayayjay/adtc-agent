"""
Date/time utilities tool. Pure stdlib -- no network calls, so it works fully
offline and doesn't touch your TPS/RAM budget beyond trivial CPU cost.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from src.core.exceptions import ToolExecutionError
from src.tools.base import BaseTool


class DateTimeTool(BaseTool):
    name = "datetime"
    description = "Get the current date/time, or compute a date offset."
    parameters = {
        "action": {"type": "string", "enum": ["now", "offset"]},
        "days": {"type": "integer", "description": "Days to offset from now (only for 'offset')."},
    }

    def run(self, action: str = "now", days: int | None = None) -> dict:
        if action == "now":
            return {"iso": datetime.now().isoformat()}
        if action == "offset":
            if days is None:
                raise ToolExecutionError(self.name, "'days' is required for offset.")
            result = datetime.now() + timedelta(days=days)
            return {"iso": result.isoformat()}
        raise ToolExecutionError(self.name, f"Unknown action: {action}")
