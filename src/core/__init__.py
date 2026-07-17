from src.core.exceptions import (
    AgentError,
    ModelLoadError,
    ModelNotReadyError,
    ToolExecutionError,
    ToolNotFoundError,
    RoutingError,
    HRMEncodingError,
    ResourceBudgetExceeded,
)

# NOTE: Agent and Lifecycle are deliberately NOT imported here. Importing
# any submodule of a package (e.g. src.core.exceptions) triggers this
# __init__.py first -- eagerly importing Agent here pulls in the full
# llm -> core.exceptions chain and causes a circular import. Import them
# directly where needed instead:
#   from src.core.agent import Agent
#   from src.core.lifecycle import Lifecycle

__all__ = [
    "AgentError",
    "ModelLoadError",
    "ModelNotReadyError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "RoutingError",
    "HRMEncodingError",
    "ResourceBudgetExceeded",
]
