"""
SUPERSEDED by src/hrm/expert_policy.py + src/hrm/state_machine.py.

Originally scoped as a separate learned "high-level planning" module. Once
the actual task was concretely designed (adaptive IMCI question-asking, see
expert_policy.py's module docstring for the full rationale), a single
unified policy function (expert_policy.next_question) turned out to capture
both the "slow" strategic decision (which symptom category to prioritize,
via _ordered_categories) and the "fast" step-level decision (which specific
field to ask within a category, via the per-category _resolve_* functions)
without needing two separate weight sets or forward passes.

Kept in the codebase as a historical marker of the design process (see
report/REPORT_TEMPLATE_NOTES.md) rather than deleted outright. If a future
iteration DOES want physically separate slow/fast learned modules (e.g. for
a training-efficiency reason), this file is the natural place to reintroduce
that split -- but it isn't necessary for the current design to work.
"""

from __future__ import annotations


class SlowModule:
    """See module docstring -- superseded by expert_policy.next_question()."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SlowModule is superseded by src.hrm.expert_policy.next_question() "
            "and src.hrm.state_machine.HRMSession. See this module's docstring."
        )