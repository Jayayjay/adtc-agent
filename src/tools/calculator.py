"""
Calculator tool: safe arithmetic evaluation.

Deliberately does NOT use eval()/exec() on arbitrary strings -- uses Python's
ast module to parse and evaluate only a whitelisted set of numeric operations.
This matters because tool inputs ultimately originate from LLM output, which
you should treat as untrusted.
"""

from __future__ import annotations

import ast
import operator

from src.core.exceptions import ToolExecutionError
from src.tools.base import BaseTool

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_func = _ALLOWED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_func = _ALLOWED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate a basic arithmetic expression (+, -, *, /, %, **)."
    parameters = {
        "expression": {"type": "string", "description": "e.g. '3 * (4 + 2) / 2'"}
    }

    def run(self, expression: str) -> dict:
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree.body)
        except Exception as e:
            raise ToolExecutionError(self.name, f"Invalid expression '{expression}': {e}")
        return {"result": result}
