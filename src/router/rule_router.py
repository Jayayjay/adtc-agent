"""
Rule-based router: decides FAST (Qwen only) vs REASONING (HRM) path.

This is the primary router. src/router/intent_classifier.py is an optional
learned upgrade -- only worth building once you have enough real task
examples (from eval/tasks/) to train it on; don't reach for it prematurely.
"""

from dataclasses import dataclass
from enum import Enum


class RoutePath(str, Enum):
    FAST = "fast"
    REASONING = "reasoning"


@dataclass
class RouteDecision:
    path: RoutePath
    reason: str


_REASONING_SIGNALS = (
    # Multi-step/constraint language (domain-agnostic, kept from original scaffold)
    "schedule", "sequence", "constraint", "optimi",
    "allocate", "route", "conflict", "dependency", "order of operations",
    "step by step", "multi-step", "prioriti",
    # IMCI/medical triage domain signals -- symptom assessment is inherently
    # a structured, multi-step classification task, exactly the REASONING
    # path's intended use case.
    "danger sign", "classify", "classification", "triage", "referral",
    "refer", "assess", "symptom", "cough", "diarrhea", "diarrhoea", "fever",
    "breathing", "dehydration", "convulsion",
)

_LONG_REQUEST_TOKEN_THRESHOLD = 40
_MULTI_TOOL_CALL_THRESHOLD = 2


def route(message: str, requested_tools: list[str] | None = None) -> RouteDecision:
    text = message.lower()
    requested_tools = requested_tools or []

    matched_signals = [s for s in _REASONING_SIGNALS if s in text]
    if matched_signals:
        return RouteDecision(
            path=RoutePath.REASONING,
            reason=f"matched reasoning signal(s): {matched_signals}",
        )

    if len(requested_tools) >= _MULTI_TOOL_CALL_THRESHOLD:
        return RouteDecision(
            path=RoutePath.REASONING,
            reason=f"multi-tool request ({len(requested_tools)} tools referenced)",
        )

    if len(text.split()) >= _LONG_REQUEST_TOKEN_THRESHOLD:
        return RouteDecision(
            path=RoutePath.REASONING,
            reason="long/complex request, routing for structured decomposition",
        )

    return RouteDecision(path=RoutePath.FAST, reason="no reasoning signals matched")
