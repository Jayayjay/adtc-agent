"""
Two properties matter here, and they pull in opposite directions.

CORRECTNESS: every true sign must appear in the vignette. A vignette that omits
a sign its answer depends on trains the model to invent that sign from nothing.
test_guard_invariant_holds is the one that must never be relaxed.

DIVERSITY: the two hidden organizer prompts are written by someone who has never
seen these templates, specifically to catch overfitting. If the generator emits
recognisable boilerplate, the fine-tune memorises the boilerplate and the hidden
prompts expose it. The diversity tests are a floor, not a target -- passing them
means the generator isn't obviously degenerate, not that it's varied enough.

The grammar tests exist because slot-filling silently produces text like "giving
she water" and "he's breathing doesn't seem right" -- fluent-looking bugs that a
schema check would never catch, and that teach the model bad English.
"""

import random
import re

import pytest

from src.sft.sampling import ALL_LABELS, required_fields_for, sample_stratified
from src.sft.verbalize import (
    PHRASES,
    STYLE_REGISTER,
    Register,
    VignetteStyle,
    render_age,
    render_rate,
    verbalize_case,
)


def _corpus(n_per_label=6, seed=0):
    """A spread of cases across every label and every style."""
    rng = random.Random(seed)
    out = []
    for child, result in sample_stratified(random.Random(seed + 1), {l: n_per_label for l in ALL_LABELS}):
        for style in VignetteStyle:
            text, rendered = verbalize_case(child, rng, style)
            out.append((child, result, style, text, rendered))
    return out


class TestGuardInvariant:
    def test_guard_invariant_holds(self):
        """THE test: every required (true) field is rendered."""
        for child, result, style, text, rendered in _corpus(n_per_label=8):
            missing = required_fields_for(child) - rendered
            assert not missing, (
                f"{style.value} vignette for {result.condition_label} omitted "
                f"required field(s) {missing}:\n{text}"
            )

    def test_positive_signs_appear_for_every_style(self):
        for child, result, style, text, rendered in _corpus(n_per_label=4):
            for field in required_fields_for(child):
                assert field in rendered, f"{field} missing in {style.value}"

    def test_unslotted_placeholders_never_survive(self):
        for child, result, style, text, rendered in _corpus(n_per_label=4):
            for slot in ("{P}", "{Po}", "{Pp}", "{C}"):
                assert slot not in text, f"unfilled slot {slot} in {style.value}:\n{text}"


class TestGrammar:
    def test_no_subject_pronoun_in_object_position(self):
        """"giving she water" -- the bug that motivated the {Po} slot."""
        bad = re.compile(
            r"\b(giving|give|put|take|took|offer|bothering|wake|tell|bring|carry|hold)\s+(she|he)\b",
            re.I,
        )
        for child, result, style, text, rendered in _corpus(n_per_label=6):
            m = bad.search(text)
            assert m is None, f"subject pronoun as object: {m.group(0)!r} in {style.value}:\n{text}"

    def test_no_contraction_where_a_possessive_was_meant(self):
        """"he's breathing doesn't seem right" -- {P}'s vs {Pp}."""
        bad = re.compile(r"\b(he's|she's)\s+(breathing|stomach|body|temperature|chest|neck|ear|eyes)\b", re.I)
        for child, result, style, text, rendered in _corpus(n_per_label=6):
            m = bad.search(text)
            assert m is None, f"contraction used as possessive: {m.group(0)!r} in {style.value}:\n{text}"

    def test_no_subject_verb_disagreement_in_closers(self):
        bad = re.compile(r"\bDo (she|he) need\b", re.I)
        for child, result, style, text, rendered in _corpus(n_per_label=6):
            assert not bad.search(text), f"agreement error in {style.value}:\n{text}"

    def test_narrative_sentences_are_capitalized(self):
        lower_start = re.compile(r"(?<=[.!?] )[a-z]")
        for child, result, style, text, rendered in _corpus(n_per_label=6):
            if style in (VignetteStyle.CARETAKER_NARRATIVE, VignetteStyle.VERBOSE_PARAGRAPH):
                m = lower_start.search(text)
                assert m is None, f"uncapitalized sentence in {style.value}:\n{text}"

    def test_pronouns_are_internally_consistent(self):
        """One child, one gender: a vignette mixing "she" and "his" reads as two
        different children and muddies what the model learns to track."""
        for child, result, style, text, rendered in _corpus(n_per_label=6):
            if style is VignetteStyle.SMS_REFERRAL:
                continue  # lowercased + typo'd; not worth asserting on
            fem = re.search(r"\b(she|her)\b", text, re.I)
            masc = re.search(r"\b(he|him|his)\b", text, re.I)
            assert not (fem and masc), f"mixed pronouns in {style.value}:\n{text}"

    def test_predicative_age_is_grammatical(self):
        """"Halima is 57-month-old" -- the bare adjectival form can't be a
        predicate."""
        bad = re.compile(r"\bis \d+-month-old\b")
        rng = random.Random(0)
        for _ in range(400):
            age = render_age(rng.randint(2, 59), rng, Register.NARRATIVE)
            assert not bad.search(f"The child is {age}."), f"ungrammatical: is {age}"

    def test_age_pluralization(self):
        """"2 years and 1 months old" -- and "1 months old", which is only
        reachable through the scope_refusal slice's out-of-band ages."""
        bad = re.compile(r"\b1 (months|years)\b|\bone months\b")
        rng = random.Random(0)
        for age in range(0, 145):  # 0-144: the full range refusals can draw
            for reg in Register:
                for _ in range(20):
                    text = render_age(age, rng, reg)
                    assert not bad.search(text), f"plural error at {age}mo/{reg.value}: {text!r}"

    def test_article_agreement(self):
        """"a 18-month-old" -- the article follows the spoken form, so it can't
        be decided from the leading digit."""
        from src.sft.verbalize import _article_for

        for n in range(2, 60):
            expected = "an" if n in (8, 11, 18) else "a"
            assert _article_for(n) == expected, f"{_article_for(n)} {n}-month-old"

    def test_article_agreement_over_the_refusal_age_range(self):
        """scope_refusal draws ages 0-1 and 61-144. "one" starts with a vowel
        LETTER but a consonant SOUND, so "a one-month-old" / "a 100-month-old"."""
        from src.sft.verbalize import _article_for

        assert _article_for(1) == "a"      # "a one-month-old"
        assert _article_for(0) == "a"      # "a zero-month-old"
        assert _article_for(100) == "a"    # "a one hundred..."
        assert _article_for(111) == "a"
        assert _article_for(80) == "an"    # "an eighty-month-old"
        assert _article_for(88) == "an"

    def test_number_spelling_survives_the_refusal_age_range(self):
        """_spell_number only handled 0-99 and raised IndexError at 100 -- on
        exactly the out-of-band ages the refusal slice generates."""
        from src.sft.verbalize import _spell_number

        for n in range(0, 145):
            assert _spell_number(n) and not _spell_number(n).startswith("-")
        assert _spell_number(138) == "one hundred and thirty-eight"
        assert _spell_number(100) == "one hundred"

    def test_render_age_never_raises_across_every_reachable_age(self):
        rng = random.Random(0)
        for age in range(0, 145):
            for reg in Register:
                assert render_age(age, rng, reg)

    def test_vocative_opener_does_not_lowercase_a_name(self):
        """"Doctor, halima is..." -- the run-on fix must not eat proper nouns."""
        from src.sft.verbalize import NAMES, _uncap

        for name in NAMES:
            assert _uncap(f"{name} is 3 months old.").startswith(name)
        assert _uncap("The baby is 3.").startswith("the")
        assert _uncap("RR 68.") == "RR 68."  # acronym untouched

    def test_no_capital_after_a_vocative_opener(self):
        bad = re.compile(r"^[A-Z][a-z ]+, [A-Z][a-z]")
        for child, result, style, text, rendered in _corpus(n_per_label=6):
            if style in (VignetteStyle.CARETAKER_NARRATIVE, VignetteStyle.VERBOSE_PARAGRAPH):
                first_word = text.split(maxsplit=1)[0].rstrip(",")
                m = bad.match(text)
                # a name legitimately follows the comma; anything else must not
                if m and not any(text.split(", ", 1)[1].startswith(n) for n in
                                 __import__("src.sft.verbalize", fromlist=["NAMES"]).NAMES):
                    raise AssertionError(f"capital after vocative in {style.value}:\n{text}")


class TestCoherence:
    def test_deep_negatives_stay_in_the_assessed_branch(self):
        """"No stridor" in a child with no cough is not something anyone says --
        and it trains the model to expect an exhaustive negative checklist that
        no real prompt contains. Gateway negatives ("no fever") are fine."""
        from src.sft.verbalize import BRANCH_OF_FIELD, TOP_LEVEL_FIELDS, decisive_branch_of

        for child, result, style, text, rendered in _corpus(n_per_label=6):
            branch = decisive_branch_of(child)
            for field in rendered:
                if field == "danger_signs_present" or getattr(child, field, None):
                    continue  # positives are required; only negatives are filtered
                if field in TOP_LEVEL_FIELDS:
                    continue
                assert BRANCH_OF_FIELD.get(field) == branch, (
                    f"{style.value} states deep negative {field!r} but the assessed "
                    f"branch is {branch!r}:\n{text}"
                )

    def test_mutex_negative_suppressed_when_partner_is_positive(self):
        """A pinch that goes back very slowly must never also be described as
        immediate -- they are two rungs of one observation, not two facts."""
        from src.sft.verbalize import MUTEX_GROUPS

        for child, result, style, text, rendered in _corpus(n_per_label=8):
            for group in MUTEX_GROUPS:
                positives = [f for f in group if getattr(child, f)]
                if not positives:
                    continue
                for other in group:
                    if other not in positives:
                        assert other not in rendered, (
                            f"{style.value} states {other!r} as absent while {positives} "
                            f"is present — same observation, contradictory:\n{text}"
                        )

    def test_one_observation_is_mentioned_once(self):
        """Both rungs false must not render as two separate "it's normal"
        fragments."""
        from src.sft.verbalize import MUTEX_GROUPS

        for child, result, style, text, rendered in _corpus(n_per_label=8):
            for group in MUTEX_GROUPS:
                assert len([f for f in group if f in rendered]) <= 1, (
                    f"{style.value} mentions the same observation twice via {group}:\n{text}"
                )


class TestDiversity:
    def test_same_case_renders_differently_across_draws(self):
        (child, _), = sample_stratified(random.Random(0), {"pneumonia": 1})
        rng = random.Random(0)
        seen = {verbalize_case(child, rng, VignetteStyle.CARETAKER_NARRATIVE)[0] for _ in range(60)}
        assert len(seen) > 45, f"only {len(seen)}/60 distinct renderings — too templated"

    def test_every_style_produces_distinct_text(self):
        (child, _), = sample_stratified(random.Random(0), {"pneumonia": 1})
        rng = random.Random(0)
        texts = {s: verbalize_case(child, rng, s)[0] for s in VignetteStyle}
        assert len(set(texts.values())) == len(VignetteStyle)

    def test_no_single_opening_dominates(self):
        """If most vignettes start the same way, the model keys on the opener
        instead of the signs."""
        rng = random.Random(0)
        (child, _), = sample_stratified(random.Random(0), {"pneumonia": 1})
        openers = [verbalize_case(child, rng, VignetteStyle.CARETAKER_NARRATIVE)[0][:14]
                   for _ in range(200)]
        most_common = max(set(openers), key=openers.count)
        assert openers.count(most_common) < 100, f"{most_common!r} opens >50% of vignettes"

    def test_age_and_rate_rendering_vary(self):
        rng = random.Random(0)
        assert len({render_age(14, rng, Register.NARRATIVE) for _ in range(60)}) >= 4
        assert len({render_rate(52, rng, Register.CLINICAL) for _ in range(60)}) >= 4

    def test_gender_varies_across_the_corpus(self):
        rng = random.Random(0)
        (child, _), = sample_stratified(random.Random(0), {"pneumonia": 1})
        texts = [verbalize_case(child, rng, VignetteStyle.CARETAKER_NARRATIVE)[0] for _ in range(60)]
        assert any(re.search(r"\bshe\b", t, re.I) for t in texts)
        assert any(re.search(r"\bhe\b", t, re.I) for t in texts)


class TestStyles:
    def test_every_style_has_a_register(self):
        assert set(STYLE_REGISTER) == set(VignetteStyle)

    def test_every_phrase_bank_covers_every_register(self):
        for field, polarities in PHRASES.items():
            for polarity, banks in polarities.items():
                assert set(banks) == set(Register), (
                    f"{field}/{polarity} is missing a register: {set(Register) - set(banks)}"
                )
                for reg, forms in banks.items():
                    assert len(forms) >= 2, f"{field}/{polarity}/{reg.value} has <2 surface forms"

    def test_structured_list_is_a_list(self):
        (child, _), = sample_stratified(random.Random(0), {"pneumonia": 1})
        text, _ = verbalize_case(child, random.Random(0), VignetteStyle.STRUCTURED_LIST)
        assert text.startswith("Age:") and "\n- " in text

    def test_dialogue_does_not_repeat_one_question(self):
        """A transcript asking "What else have you noticed?" fourteen times is
        not a consultation."""
        (child, _), = sample_stratified(random.Random(0), {"some_dehydration": 1})
        text, _ = verbalize_case(child, random.Random(0), VignetteStyle.DIALOGUE_TRANSCRIPT)
        probes = [l for l in text.splitlines() if l.startswith("HW: ")]
        if len(probes) >= 5:
            assert len(set(probes)) > 2, f"transcript reuses the same probe:\n{text}"

    def test_sms_is_lowercase(self):
        (child, _), = sample_stratified(random.Random(0), {"pneumonia": 1})
        text, _ = verbalize_case(child, random.Random(0), VignetteStyle.SMS_REFERRAL)
        assert text == text.lower()
