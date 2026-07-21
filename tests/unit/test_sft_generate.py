"""
Splitting is where a fine-tune quietly lies to you.

There are two ways to leak here, and only one of them is obvious:

  case leakage     -- the same case in train and val. split_by_case (reused
                      from the HRM generator) already handles it.
  phrasing leakage -- val drawn from the same style families as train. This is
                      the one that matters: the two hidden organizer prompts
                      are written by someone who has never seen our templates,
                      so a val set sharing our phrasing measures memorisation
                      and reports ~100% while the model is about to fail.

val_ood exists to be the honest number. These tests make sure it stays honest.
"""

import random

import pytest

from scripts.generate_sft_data import (
    DEFAULT_HOLDOUT_STYLES,
    MIXTURE,
    _acknowledged_extra,
    build_target,
    generate,
    split_sft,
)
from src.sft.sampling import ALL_LABELS, sample_stratified
from src.sft.verbalize import VignetteStyle, decisive_branch_of


class TestBuildTarget:
    def test_covers_every_label(self):
        assert set(build_target(7500)) == set(ALL_LABELS)

    def test_rare_labels_get_real_representation(self):
        """mastoiditis is 0.4% of raw draws. If the target left it there, the
        model would never learn the branch."""
        t = build_target(7500)
        assert t["mastoiditis"] == t["pneumonia"]
        assert t["mastoiditis"] > 400

    def test_no_classification_matched_is_capped_but_present(self):
        """It must not dominate -- but it must exist, or the model learns to
        force a triage onto small talk."""
        t = build_target(7500)
        total = sum(t.values())
        frac = t["no_classification_matched"] / total
        assert 0 < frac < 0.06, f"no_classification_matched at {frac:.1%}"

    def test_small_targets_do_not_collapse_to_zero(self):
        """Rehearsal runs use tiny counts; a quota of 0 would silently drop a
        whole classification."""
        t = build_target(50)
        assert all(v >= 1 for v in t.values()), t


class TestSplitSft:
    def _records(self, n=400):
        rng = random.Random(0)
        recs = []
        styles = list(VignetteStyle)
        for i, style in enumerate(styles * (n // len(styles) + 1)):
            recs.append({
                "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
                "case_id": f"case_{i}",
                "meta": {"kind": "triage", "style": style.value},
            })
        return recs[:n]

    def test_val_ood_contains_only_holdout_styles(self):
        recs = self._records()
        holdout = set(DEFAULT_HOLDOUT_STYLES)
        splits = split_sft(recs, holdout, 0.1, 0.1, 0)
        assert splits["val_ood"], "val_ood is empty — no signal on unseen phrasing"
        for r in splits["val_ood"]:
            assert r["meta"]["style"] in holdout

    def test_holdout_styles_never_appear_in_train(self):
        """THE test: if a held-out style leaks into train, val_ood stops
        measuring generalisation and the number becomes a lie."""
        recs = self._records()
        holdout = set(DEFAULT_HOLDOUT_STYLES)
        splits = split_sft(recs, holdout, 0.1, 0.1, 0)
        for name in ("train", "val", "test"):
            for r in splits[name]:
                assert r["meta"]["style"] not in holdout, f"{r['meta']['style']} leaked into {name}"

    def test_no_case_id_shared_between_train_and_val(self):
        recs = self._records()
        splits = split_sft(recs, set(DEFAULT_HOLDOUT_STYLES), 0.1, 0.1, 0)
        train_ids = {r["case_id"] for r in splits["train"]}
        for name in ("val", "test"):
            assert not (train_ids & {r["case_id"] for r in splits[name]})

    def test_every_record_lands_somewhere(self):
        recs = self._records()
        splits = split_sft(recs, set(DEFAULT_HOLDOUT_STYLES), 0.1, 0.1, 0)
        assert sum(len(v) for v in splits.values()) == len(recs)

    def test_empty_holdout_yields_empty_val_ood(self):
        recs = self._records()
        splits = split_sft(recs, set(), 0.1, 0.1, 0)
        assert splits["val_ood"] == []

    def test_non_triage_records_are_not_lost(self):
        """scope_refusal/general_chat carry style=None and must still be split,
        not silently dropped by the style filter."""
        recs = self._records(100)
        recs += [{"messages": [], "case_id": f"r_{i}", "meta": {"kind": "scope_refusal", "style": None}}
                 for i in range(20)]
        splits = split_sft(recs, set(DEFAULT_HOLDOUT_STYLES), 0.1, 0.1, 0)
        kept = sum(1 for v in splits.values() for r in v if r["meta"]["kind"] == "scope_refusal")
        assert kept == 20


class TestAcknowledgedExtra:
    def test_fires_sometimes_not_always(self):
        """~15%: if every example were pruned to one symptom, a real
        multi-symptom prompt would train the model to drop what it was told."""
        (child, _), = sample_stratified(random.Random(0), {"pneumonia": 1})
        rng = random.Random(0)
        hits = sum(_acknowledged_extra(child, rng) is not None for _ in range(1000))
        assert 80 < hits < 240, f"{hits}/1000 — not ~15%"

    def test_text_matches_the_branch(self):
        rng = random.Random(0)
        for label, expect in [("pneumonia", "cough"), ("some_dehydration", "diarrhoea")]:
            (child, _), = sample_stratified(random.Random(1), {label: 1})
            texts = {_acknowledged_extra(child, rng) for _ in range(200)}
            texts.discard(None)
            assert texts, f"never fired for {label}"
            assert any(expect in t for t in texts), f"{label}: {texts}"

    def test_returns_none_for_branches_without_a_note(self):
        """danger/ear/none have no later branch to defer -- no note is correct,
        and inventing one would state something untrue."""
        rng = random.Random(0)
        (child, _), = sample_stratified(random.Random(1), {"very_severe_disease": 1})
        assert decisive_branch_of(child) == "danger"
        assert all(_acknowledged_extra(child, rng) is None for _ in range(50))


class TestGenerateEndToEnd:
    def test_small_run_is_wellformed(self):
        recs = generate(num_triage=60, seed=1, chat_path=None)
        assert recs
        kinds = {r["meta"]["kind"] for r in recs}
        assert "triage" in kinds and "scope_refusal" in kinds
        for r in recs:
            roles = [m["role"] for m in r["messages"]]
            assert roles[-2:] == ["user", "assistant"]
            assert r["case_id"] and r["meta"]["kind"]

    def test_mixture_weights_sum_to_one(self):
        assert abs(sum(MIXTURE.values()) - 1.0) < 1e-9

    def test_missing_chat_pool_warns_but_does_not_crash(self, capsys):
        """The pool takes ~1-2h of CPU to build. Generation must stay usable
        without it, but must say loudly what's missing."""
        from pathlib import Path

        generate(num_triage=30, seed=2, chat_path=Path("/nonexistent/chat.jsonl"))
        out = capsys.readouterr().out
        assert "WARNING" in out and "general-chat" in out


class TestCorpusAgainstGroundTruth:
    """Audits the emitted records by re-running assess() on the fields each one
    records, rather than by pattern-matching its prose.

    This distinction is the point. Grepping vignettes for "lethargic" flags
    "lethargic -" -- clinical shorthand for NEGATIVE -- and reports a
    catastrophic under-triage bug that does not exist. One character separates
    the assertion from its negation. Text can't be the source of truth for what
    a vignette claims; the recorded fields can.
    """

    def _corpus(self, n=200):
        return [r for r in generate(num_triage=n, seed=3, chat_path=None)
                if r["meta"]["kind"] == "triage"]

    def test_recorded_label_matches_a_fresh_assess(self):
        from src.tools.imci_protocol import ChildAssessment, assess

        for r in self._corpus():
            m = r["meta"]
            res = assess(ChildAssessment(age_months=m["age_months"], **m["fields"]))
            assert res.condition_label == m["condition_label"]
            assert res.classification.value == m["classification"]

    def test_no_case_asserting_lethargy_is_non_severe(self):
        """THE invariant, checked against the fields rather than the prose."""
        from src.tools.imci_protocol import ChildAssessment, Classification, assess

        checked = 0
        for r in self._corpus(400):
            m = r["meta"]
            child = ChildAssessment(age_months=m["age_months"], **m["fields"])
            leth = (child.child_lethargic_or_unconscious
                    or "lethargic_or_unconscious" in child.danger_signs_present)
            if not leth:
                continue
            checked += 1
            assert assess(child).classification is Classification.SEVERE, m["fields"]
        assert checked > 0, "no lethargic cases generated — the invariant went untested"

    def test_header_agrees_with_ground_truth(self):
        from src.tools.imci_protocol import ChildAssessment, assess

        for r in self._corpus():
            m = r["meta"]
            expected = assess(ChildAssessment(age_months=m["age_months"], **m["fields"]))
            assert r["messages"][-1]["content"].startswith(
                f"CLASSIFICATION: {expected.classification.value.upper()} "
            )

    def test_dosing_line_matches_the_tables_no_drift(self):
        """Every DOSING line must be exactly what the reviewed dosing tables
        produce for the recorded weight+age -- no hallucinated or drifted dose."""
        from src.sft.treatment import render_dosing

        checked = 0
        for r in self._corpus(400):
            m = r["meta"]
            content = r["messages"][-1]["content"]
            dosing_lines = [ln[len("DOSING: "):] for ln in content.splitlines()
                            if ln.startswith("DOSING: ")]
            expected = render_dosing(m["condition_label"], m["weight_kg"], m["age_months"])
            if expected is None:
                assert not dosing_lines, f"{m['condition_label']} should have no DOSING line"
            else:
                assert dosing_lines == [expected], (m["condition_label"], dosing_lines, expected)
                checked += 1
        assert checked > 0, "no dosed cases generated — the dosing audit went untested"

    def test_extended_dosing_line_matches_the_tables_no_drift(self):
        """Same no-drift guarantee for --include-extended records (rendered by a
        SEPARATE path, render_extended_answer)."""
        from src.sft.treatment import render_dosing

        # young-infant records are extended but carry no weight_kg (no dosing by
        # design -- their treatment is IM antibiotics / referral, not in the tables).
        recs = [r for r in generate(num_triage=200, seed=5, chat_path=None, include_extended=True)
                if r["meta"].get("extended") and "weight_kg" in r["meta"]]
        checked = 0
        for r in recs:
            m = r["meta"]
            content = r["messages"][-1]["content"]
            dosing_lines = [ln[len("DOSING: "):] for ln in content.splitlines()
                            if ln.startswith("DOSING: ")]
            expected = render_dosing(m["condition_label"], m["weight_kg"], m["age_months"])
            if expected is None:
                assert not dosing_lines, f"{m['condition_label']} should have no DOSING line"
            else:
                assert dosing_lines == [expected], (m["condition_label"], dosing_lines, expected)
                checked += 1
        assert checked > 0, "no dosed extended cases generated — the extended dosing audit went untested"

    def test_fields_round_trip_through_json(self):
        """meta.fields is only useful to an auditor if it survives JSONL."""
        import json

        from src.tools.imci_protocol import ChildAssessment, assess

        for r in self._corpus(60):
            m = json.loads(json.dumps(r))["meta"]
            assess(ChildAssessment(age_months=m["age_months"], **m["fields"]))

    def test_fields_records_only_what_differs_from_default(self):
        """Recording every field would triple the corpus to say nothing --
        absent means False/None to assess()."""
        for r in self._corpus(60):
            assert "age_months" not in r["meta"]["fields"]
            assert all(v not in (False, None, []) for v in r["meta"]["fields"].values()), \
                r["meta"]["fields"]
