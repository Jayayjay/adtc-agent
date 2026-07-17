"""
Two things to lock down here, neither of which needs the model loaded:

1. parse_classification must find the classification in the rigid header the
   generator produces, and return None (a failure, not a crash) when it can't.
2. The NL eval set must not drift from ground truth: every case's structured
   `input` must produce its claimed `expected_classification` through the SAME
   rule engine that labels the training data. Otherwise the "expected" answers
   are just an author's opinion, and a passing model score means nothing.
"""

import json
from pathlib import Path

import pytest

from eval.scoring.model_sacc_scorer import DEFAULT_TASKS, load_tasks, parse_classification
from src.tools.imci_triage_tool import IMCITriageTool


class TestParseClassification:
    def test_reads_the_rigid_header(self):
        text = "CLASSIFICATION: MODERATE — Pneumonia (IMCI yellow: treat and follow up)\n\nWHY: ..."
        assert parse_classification(text) == "moderate"

    def test_case_insensitive(self):
        assert parse_classification("classification: severe — very severe disease") == "severe"

    def test_bare_word_fallback(self):
        assert parse_classification("This looks like severe dehydration to me.") == "severe"

    def test_unparseable_returns_none(self):
        """None counts as a failure downstream -- if the scorer can't find the
        classification, a grader can't either."""
        assert parse_classification("I'm not sure, could you tell me more?") is None
        assert parse_classification("") is None

    def test_header_wins_over_stray_word(self):
        text = "CLASSIFICATION: MILD — Cough or cold. This is not severe."
        assert parse_classification(text) == "mild"


class TestNLEvalSet:
    def _tasks(self):
        return json.loads(Path(DEFAULT_TASKS).read_text())["examples"]

    def test_structured_input_matches_expected_classification(self):
        """THE cross-check: the NL prompt and its expected label must agree with
        what the rule engine says about the structured input. This makes it
        impossible for the model scorer and the protocol scorer to disagree
        about what 'correct' means."""
        tool = IMCITriageTool()
        for t in self._tasks():
            if t.get("input") is None or t.get("expected_classification") is None:
                continue
            result = tool.run(**t["input"])
            assert result["classification"] == t["expected_classification"], (
                f"{t['id']}: input yields {result['classification']}, "
                f"but expected_classification says {t['expected_classification']}"
            )

    def test_condition_labels_also_agree(self):
        tool = IMCITriageTool()
        for t in self._tasks():
            if t.get("input") is None or not t.get("expected_condition_label"):
                continue
            result = tool.run(**t["input"])
            assert result["condition_label"] == t["expected_condition_label"], t["id"]

    def test_loader_keeps_only_scorable_cases(self):
        """Small talk / out-of-band cases have no expected_classification and
        must not be scored for accuracy (they're checked for over-triage
        separately)."""
        scorable = load_tasks(Path(DEFAULT_TASKS))
        assert all(t.get("prompt") and t.get("expected_classification") for t in scorable)
        assert len(scorable) < len(self._tasks())  # the null-label cases were dropped

    def test_covers_every_severity(self):
        labels = {t["expected_classification"] for t in self._tasks()
                  if t.get("expected_classification")}
        assert labels == {"severe", "moderate", "mild"}

    def test_has_out_of_scope_probes(self):
        """The set must include small talk and an out-of-band age -- the model
        must not classify those."""
        null_cases = [t for t in self._tasks() if t.get("expected_classification") is None]
        assert len(null_cases) >= 2
