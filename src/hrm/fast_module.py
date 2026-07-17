"""
SUPERSEDED by src/hrm/expert_policy.py + src/hrm/state_machine.py.
See src/hrm/slow_module.py's module docstring for the full explanation --
the same rationale applies here.
"""

from __future__ import annotations


class FastModule:
    """See module docstring -- superseded by expert_policy.next_question()."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FastModule is superseded by src.hrm.expert_policy.next_question() "
            "and src.hrm.state_machine.HRMSession. See this module's docstring."
        )