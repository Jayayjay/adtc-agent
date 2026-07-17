"""
Placeholder integration test for HRM + LLM working together. Currently
skipped unconditionally -- HRM's encode_task/decode_output/module loading
are all unimplemented stubs (see src/hrm/*.py) pending your task-encoding
design decision and a fine-tuned checkpoint. Un-skip once that exists.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="HRM task encoding not yet designed and no fine-tuned checkpoint exists. "
           "See src/hrm/encoders.py docstring for what needs to be decided first."
)


def test_hrm_produces_plan_llm_can_format():
    pass
