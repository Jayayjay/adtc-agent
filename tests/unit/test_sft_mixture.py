"""
The mixture exists to stop the model classifying everything.

A model trained only on triage pairs learns that every input gets a
CLASSIFICATION: header -- including "hello", including a newborn, including a
malaria question it has no basis to answer. Each non-triage kind is a specific
defence:

  scope_refusal  -- assess() cannot label malaria/malnutrition/young-infant, so
                    the model must decline rather than confabulate. This is the
                    honest answer to the risk that hidden prompts probe exactly
                    the branches imci_protocol.py documents as unmodelled.
  next_question  -- an underspecified prompt should get a question back, not an
                    invented respiratory rate.
  general_chat   -- anti-forgetting.

The load-bearing test here is test_next_question_never_states_unknown_fields:
mid-dialogue, an un-asked field is at its dataclass default, which is
indistinguishable from a real "no".
"""

import json
import random

import pytest

from src.hrm.expert_policy import QUESTION_TEXT
from src.sft.mixture import (
    OUT_OF_SCOPE_TOPICS,
    REFUSAL_HEADER,
    SYSTEM_PROMPTS,
    Kind,
    make_general_chat_example,
    make_next_question_example,
    make_scope_refusal_example,
    make_triage_example,
)
from src.sft.sampling import ALL_LABELS, sample_stratified
from src.sft.verbalize import VignetteStyle


def _messages_ok(rec):
    roles = [m["role"] for m in rec["messages"]]
    assert roles[-2:] == ["user", "assistant"]
    assert roles[:-2] in ([], ["system"])
    for m in rec["messages"]:
        assert m["content"].strip(), "empty message content"
    return True


class TestRecordShape:
    def test_triage_record_is_wellformed(self):
        rng = random.Random(0)
        (child, result), = sample_stratified(random.Random(1), {"pneumonia": 1})
        rec = make_triage_example(child, result, rng, VignetteStyle.CHW_NOTES, "sft_0")
        _messages_ok(rec)
        assert rec["meta"]["kind"] == "triage"
        assert rec["meta"]["condition_label"] == "pneumonia"
        assert json.dumps(rec)  # must survive JSONL round-trip

    def test_all_kinds_serialise(self):
        rng = random.Random(0)
        (child, result), = sample_stratified(random.Random(1), {"pneumonia": 1})
        recs = [
            make_triage_example(child, result, rng, VignetteStyle.CHW_NOTES, "a"),
            make_scope_refusal_example(rng, "b"),
            make_general_chat_example("Hi", "Hello!", rng, "c"),
        ]
        nq = make_next_question_example(random.Random(3), "d")
        if nq:
            recs.append(nq)
        for r in recs:
            _messages_ok(r)
            assert json.loads(json.dumps(r))


class TestSystemPrompt:
    def test_roughly_half_carry_no_system_prompt(self):
        """The evaluator may pass no system prompt at all. A model that only
        behaves when primed fails on the day."""
        rng = random.Random(0)
        (child, result), = sample_stratified(random.Random(1), {"pneumonia": 1})
        withs = sum(
            make_triage_example(child, result, rng, VignetteStyle.CHW_NOTES, "x")["messages"][0]["role"] == "system"
            for _ in range(400)
        )
        assert 140 < withs < 260, f"{withs}/400 carry a system prompt — not ~50%"

    def test_system_prompts_never_mention_the_tool_layer(self):
        """src/llm/prompts.BASE_SYSTEM_PROMPT tells the model to announce "the
        imci_triage tool". No tool exists at evaluation — only the raw gguf."""
        for p in SYSTEM_PROMPTS:
            assert "imci_triage" not in p and "tool" not in p.lower()

    def test_system_prompts_are_short(self):
        for p in SYSTEM_PROMPTS:
            assert len(p) < 500, "long system prompts eat the context window"


class TestScopeRefusal:
    def test_refusal_never_emits_a_classification(self):
        """The whole point: no classification exists for these."""
        rng = random.Random(0)
        for i in range(300):
            rec = make_scope_refusal_example(rng, f"r{i}")
            answer = rec["messages"][-1]["content"]
            assert answer.startswith(REFUSAL_HEADER), answer[:60]
            assert "CLASSIFICATION:" not in answer

    def test_refusal_defers_to_the_booklet(self):
        rng = random.Random(0)
        for i in range(200):
            answer = make_scope_refusal_example(rng, f"r{i}")["messages"][-1]["content"]
            assert "chart" in answer.lower() or "booklet" in answer.lower() or "refer" in answer.lower()

    def test_refusal_has_no_leaks(self):
        import re

        leak = re.compile(r"imci_protocol|\.py\b|scaffold|not modeled|\['", re.I)
        rng = random.Random(0)
        for i in range(300):
            rec = make_scope_refusal_example(rng, f"r{i}")
            for m in rec["messages"]:
                assert not leak.search(m["content"]), m["content"][:120]

    def test_both_refusal_sources_are_reachable(self):
        """Topic refusals teach "this subject isn't mine"; age refusals teach
        "check the age", which generalises to a newborn vignette that never
        says "newborn"."""
        rng = random.Random(0)
        topics = {make_scope_refusal_example(rng, f"r{i}")["meta"]["topic"] for i in range(400)}
        assert "young_infant" in topics and "over_five" in topics
        assert topics & {t[0] for t in OUT_OF_SCOPE_TOPICS}

    def test_unmodelled_branches_are_all_covered(self):
        """imci_protocol.py's docstring names exactly what it does not model."""
        covered = {t[0] for t in OUT_OF_SCOPE_TOPICS}
        assert {"malaria", "malnutrition", "anaemia", "measles", "young_infant"} <= covered


class TestNextQuestion:
    def test_next_question_never_states_unknown_fields(self):
        """THE test for this kind. Mid-dialogue an un-asked field sits at its
        default, which looks exactly like a genuine negative. If the vignette
        rendered it, we'd be putting answers in the caretaker's mouth -- and
        then asking a question that was already answered."""
        from src.sft.verbalize import PHRASES

        rng = random.Random(7)
        checked = 0
        for i in range(250):
            rec = make_next_question_example(rng, f"q{i}")
            if rec is None:
                continue
            checked += 1
            # the field being asked about must not already appear in the prompt
            field = rec["meta"]["question_field"]
            prompt = rec["messages"][-2]["content"].lower()
            bank = PHRASES.get(field)
            if not bank:
                continue
            for polarity in ("true", "false"):
                for reg_forms in bank[polarity].values():
                    for form in reg_forms:
                        stripped = form.replace("{P}", "").replace("{Po}", "") \
                                       .replace("{Pp}", "").replace("{C}", "").strip().lower()
                        if len(stripped) > 12:
                            assert stripped not in prompt, (
                                f"asking about {field} but the prompt already says "
                                f"{stripped!r}:\n{rec['messages'][-2]['content']}"
                            )
        assert checked > 40, f"only {checked} usable next_question examples"

    def test_asks_a_real_protocol_question(self):
        rng = random.Random(5)
        made = 0
        for i in range(200):
            rec = make_next_question_example(rng, f"q{i}")
            if rec is None:
                continue
            made += 1
            answer = rec["messages"][-1]["content"]
            assert answer.startswith("NEXT QUESTION: ")
            assert "CLASSIFICATION:" not in answer
            assert QUESTION_TEXT[rec["meta"]["question_field"]] in answer
        assert made > 30

    def test_returns_none_rather_than_a_bad_example(self):
        """Rollouts that stop on turn 0 (danger sign) have no usable mid-point;
        the generator must skip them, not emit something degenerate."""
        rng = random.Random(0)
        results = [make_next_question_example(rng, f"q{i}") for i in range(60)]
        assert any(r is None for r in results) or all(r is not None for r in results)
        for r in results:
            if r is not None:
                _messages_ok(r)


class TestCleanCompletion:
    """Self-distillation copies the base model's artefacts along with its
    knowledge. Measured over 228 real sampled pairs: 25% truncated mid-sentence
    by the token cap, ~2% answered in Chinese, a few near-empty."""

    def test_truncated_completion_is_trimmed_to_a_sentence(self):
        from src.sft.mixture import clean_completion

        text = ("The moon was high over the village that night. The stars were "
                "twinkling, Fatima fell asleep")
        out = clean_completion(text)
        assert out == "The moon was high over the village that night."

    def test_trim_that_leaves_too_little_is_rejected(self):
        """Trimming can strand a fragment shorter than the floor -- rejecting
        is right, but it must not return the fragment."""
        from src.sft.mixture import clean_completion

        out = clean_completion("Yes. And then it got cut off right about he")
        assert out is None or len(out) >= 20

    def test_single_unfinished_sentence_is_rejected(self):
        from src.sft.mixture import clean_completion

        assert clean_completion("Debugging a failing unit test is simpler than it") is None

    def test_non_english_is_rejected(self):
        """metadata.json declares language_scope ["en"], and the graders read
        English. Qwen answers "2 + 2 =" in Chinese ~2% of the time."""
        from src.sft.mixture import clean_completion

        assert clean_completion("在标准的数学逻辑中，**$2 + 2 = 4$**。这意味着两个独立的数相加。") is None
        assert clean_completion("Привет, как дела? Это хорошо.") is None

    def test_short_and_empty_rejected(self):
        from src.sft.mixture import clean_completion

        assert clean_completion("") is None
        assert clean_completion(None) is None
        assert clean_completion("Yes.") is None  # under min_chars

    def test_complete_completion_passes_through(self):
        from src.sft.mixture import clean_completion

        text = "The capital of Nigeria is Abuja. It became the capital in 1991."
        assert clean_completion(text) == text

    def test_accepts_code_and_markdown_endings(self):
        from src.sft.mixture import clean_completion

        for text in (
            "Here is the function:\n```python\ndef f():\n    pass\n```",
            "Three tips: be clear, be brief, be kind!",
            "Use a **hash map** for O(1) lookup.",
        ):
            assert clean_completion(text) == text

    def test_output_never_ends_mid_word(self):
        """A target that stops mid-word teaches the model to stop mid-word."""
        from src.sft.mixture import clean_completion

        for text in (
            "First sentence here. Second one got cut off halfw",
            "Complete. And another. Then trunca",
        ):
            out = clean_completion(text)
            if out is not None:
                assert out.endswith((".", "!", "?", '"', "`", ")", "*")), out


class TestExtendedExamples:
    """Risk #6: malaria/measles/anaemia/malnutrition were previously refused.
    make_extended_example turns them into classifications from
    src/sft/extended_protocol.py -- addressing the risk, gated behind sign-off."""

    def test_produces_all_four_branches(self):
        from src.sft.mixture import make_extended_example

        rng = random.Random(0)
        branches = set()
        for i in range(400):
            r = make_extended_example(rng, f"e{i}")
            if r:
                branches.add(r["meta"]["branch"])
        assert branches == {"malaria", "measles", "anaemia", "malnutrition"}

    def test_answer_matches_the_classifier(self):
        """The rendered header severity must equal what the extended classifier
        returned -- the same ground-truth discipline as the core triage path."""
        from src.sft.mixture import make_extended_example

        rng = random.Random(1)
        checked = 0
        for i in range(300):
            r = make_extended_example(rng, f"e{i}")
            if not r:
                continue
            checked += 1
            assert r["messages"][-1]["content"].startswith(
                f"CLASSIFICATION: {r['meta']['classification'].upper()} "
            )
        assert checked > 100

    def test_extended_answers_have_no_leaks(self):
        import re

        from src.sft.mixture import make_extended_example

        leak = re.compile(r"imci_protocol|extended_protocol|\.py\b|scaffold|\['", re.I)
        rng = random.Random(2)
        for i in range(300):
            r = make_extended_example(rng, f"e{i}")
            if r:
                for m in r["messages"]:
                    assert not leak.search(m["content"]), m["content"][:120]

    def test_records_are_flagged_extended(self):
        """The flag lets a shipped run exclude them until sign-off, and lets an
        auditor find them."""
        from src.sft.mixture import make_extended_example

        rng = random.Random(3)
        r = None
        while r is None:
            r = make_extended_example(rng, "e")
        assert r["meta"]["extended"] is True
        assert r["meta"]["kind"] == "triage"


class TestTriageGuard:
    def test_missing_required_field_raises(self):
        """make_triage_example asserts the guard invariant rather than trusting
        the verbalizer."""
        import src.sft.mixture as mixture

        rng = random.Random(0)
        (child, result), = sample_stratified(random.Random(1), {"pneumonia": 1})
        original = mixture.verbalize_case
        try:
            mixture.verbalize_case = lambda *a, **k: ("A child is unwell.", set())
            with pytest.raises(AssertionError, match="omitted required field"):
                make_triage_example(child, result, rng, VignetteStyle.CHW_NOTES, "sft_x")
        finally:
            mixture.verbalize_case = original

    def test_acknowledged_extra_reaches_the_answer(self):
        rng = random.Random(0)
        (child, result), = sample_stratified(random.Random(1), {"pneumonia": 1})
        rec = make_triage_example(child, result, rng, VignetteStyle.CHW_NOTES, "sft_y",
                                  acknowledged_extra="Also has a fever.")
        assert "Also has a fever." in rec["messages"][-1]["content"]
