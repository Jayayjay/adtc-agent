"""Custom exceptions -- keep error handling explicit rather than catching bare Exception."""


class AgentError(Exception):
    """Base class for all agent-related errors."""


class ModelLoadError(AgentError):
    """Raised when a GGUF or HRM checkpoint fails to load."""


class ModelNotReadyError(AgentError):
    """Raised when inference is attempted before a model is loaded."""


class ToolExecutionError(AgentError):
    """Raised when a tool call fails during execution."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ToolNotFoundError(AgentError):
    """Raised when the LLM requests a tool that isn't registered."""


class RoutingError(AgentError):
    """Raised when the router cannot make a routing decision."""


class HRMEncodingError(AgentError):
    """Raised when a task cannot be encoded into HRM's input representation."""


class ResourceBudgetExceeded(AgentError):
    """Raised when RAM or thermal budgets are exceeded during a self-check."""
