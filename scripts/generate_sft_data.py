"""
Generates the supervised fine-tuning corpus: (vignette -> clinical answer) pairs
labelled by src/tools/imci_protocol.assess().

Why this exists: the ADTC evaluator runs a single raw .gguf through llama.cpp
and nothing else. The deterministic rule engine, the tool layer, and the HRM
orchestrator are never executed there, so the IMCI knowledge they encode has to
be moved into the weights. assess() stays the ground truth -- it labels this
data, and it scores the result in eval/.

Pipeline (order is load-bearing, see src/sft/sampling.py):
    _random_case -> repair_coherence -> assess -> prune_to_decisive_branch
                 -> verbalize -> render_answer

Usage:
    python scripts/generate_sft_data.py --num-triage 7500
    python scripts/generate_sft_data.py --num-triage 200 --out-dir /tmp/smoke   # rehearsal
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_hrm_training_data import split_by_case
from src.sft.answer import render_answer
from src.sft.mixture import (
    Kind,
    clean_completion,
    make_extended_example,
    make_general_chat_example,
    make_next_question_example,
    make_scope_refusal_example,
    make_triage_example,
)
from src.sft.sampling import ALL_LABELS, prune_to_decisive_branch, sample_stratified
from src.sft.verbalize import VignetteStyle
from src.tools.imci_protocol import assess

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sft"

# Held out of training entirely, so val_ood measures generalisation to phrasing
# the model has never seen. This is the ONLY number here that predicts
# performance on the two hidden organizer prompts -- train accuracy and
# in-distribution val accuracy will both read ~100% and both mean nothing.
DEFAULT_HOLDOUT_STYLES = ("verbose_paragraph", "dialogue_transcript")

MIXTURE = {Kind.TRIAGE: 0.75, Kind.SCOPE_REFUSAL: 0.05,
           Kind.NEXT_QUESTION: 0.05, Kind.GENERAL_CHAT: 0.15}


def build_target(num_triage: int) -> dict[str, int]:
    """Near-balanced across the 14 producible labels.

    _random_case's Bernoullis give cough_or_cold/no_dehydration/pneumonia ~16%
    each and mastoiditis 0.4%; a model trained on that never learns the rare
    branches, which are the ones that kill children. no_classification_matched
    is capped low but never zero -- the model must not force a triage onto
    small talk.
    """
    n_other = max(1, int(num_triage * 0.03))
    rest = [l for l in ALL_LABELS if l != "no_classification_matched"]
    per = max(1, (num_triage - n_other) // len(rest))
    target = {l: per for l in rest}
    target["no_classification_matched"] = n_other
    return target


def _acknowledged_extra(child, rng: random.Random) -> str | None:
    """~15% of cases keep a non-decisive symptom in play.

    assess() short-circuits, so a cough+fever case is decided by cough alone and
    prune_to_decisive_branch drops the fever. If EVERY example is pruned, the
    model only ever sees single-symptom prompts -- and a real multi-symptom
    prompt would train it to silently drop what it was told. These examples say
    the quiet part out loud instead.
    """
    if rng.random() >= 0.15:
        return None
    branch_extra = {
        "cough": "The child also has a fever. The chart booklet works through cough and "
                 "breathing first, so classify that, then reassess the fever separately.",
        "diarrhea": "The child also has a fever. Classify the diarrhoea first as the booklet "
                    "orders, then come back to the fever.",
        "fever": "There is also an ear complaint. Fever is assessed before ear problems, so "
                 "deal with this classification first and then check the ear.",
    }
    from src.sft.verbalize import decisive_branch_of

    return branch_extra.get(decisive_branch_of(child))


def generate(num_triage: int, seed: int, chat_path: Path | None,
             include_extended: bool = False, extended_frac: float = 0.15) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []

    # --- triage -------------------------------------------------------------
    cases = sample_stratified(random.Random(seed + 1), build_target(num_triage))
    styles = list(VignetteStyle)
    for i, (child, result) in enumerate(cases):
        style = rng.choice(styles)
        records.append(
            make_triage_example(child, result, rng, style, f"sft_triage_{i:06d}",
                                acknowledged_extra=_acknowledged_extra(child, rng))
        )

    # --- extended branches (malaria/measles/anaemia/malnutrition) -----------
    # OFF by default: these classifiers are UNREVIEWED clinical logic. When on,
    # they replace part of the refusal behaviour with actual classifications for
    # the branches imci_protocol.py declares unmodelled -- addressing risk #6,
    # but only after sign-off (see src/sft/extended_protocol.py).
    if include_extended:
        want_ext = int(len(records) * extended_frac)
        made = attempts = 0
        while made < want_ext and attempts < want_ext * 20:
            attempts += 1
            ex = make_extended_example(rng, f"sft_ext_{made:06d}")
            if ex is not None:
                records.append(ex)
                made += 1
        print(f"extended branches: {made} examples "
              f"(malaria/measles/anaemia/malnutrition) -- UNREVIEWED clinical logic")

    n_triage = len(records)
    n_total = int(n_triage / MIXTURE[Kind.TRIAGE])

    # --- scope refusals -----------------------------------------------------
    for i in range(int(n_total * MIXTURE[Kind.SCOPE_REFUSAL])):
        records.append(make_scope_refusal_example(rng, f"sft_refusal_{i:06d}"))

    # --- next question ------------------------------------------------------
    want = int(n_total * MIXTURE[Kind.NEXT_QUESTION])
    made = attempts = 0
    while made < want and attempts < want * 20:
        attempts += 1
        ex = make_next_question_example(rng, f"sft_nextq_{made:06d}")
        if ex is not None:
            records.append(ex)
            made += 1
    if made < want:
        print(f"WARNING: only {made}/{want} next_question examples -- most rollouts stop "
              f"on turn 0 (danger sign) and have no usable mid-point.")

    # --- general chat -------------------------------------------------------
    want_chat = int(n_total * MIXTURE[Kind.GENERAL_CHAT])
    if chat_path and chat_path.exists():
        raw = [json.loads(l) for l in chat_path.read_text().splitlines() if l.strip()]
        # Filter at load, not only at distill time: the pool is expensive to
        # rebuild (~1-2h of idle CPU) and may predate the cleaner. Self-
        # distillation copies the base model's artefacts too -- mid-sentence
        # truncation from the token cap, the occasional Chinese answer.
        pool = []
        for item in raw:
            cleaned = clean_completion(item["completion"])
            if cleaned:
                pool.append({"prompt": item["prompt"], "completion": cleaned})
        dropped = len(raw) - len(pool)
        if dropped:
            print(f"general-chat pool: kept {len(pool)}/{len(raw)} "
                  f"({dropped} dropped: truncated, non-English, or too short)")
        if not pool:
            print(f"WARNING: {chat_path} yielded no usable pairs after cleaning.")
        for i in range(want_chat):
            if not pool:
                break
            item = pool[i % len(pool)]
            records.append(
                make_general_chat_example(item["prompt"], item["completion"], rng,
                                          f"sft_chat_{i:06d}")
            )
        if pool and want_chat > len(pool) * 3:
            print(f"NOTE: {want_chat} chat slots from a pool of {len(pool)} means each pair "
                  f"repeats ~{want_chat/len(pool):.0f}x. Consider a larger --num-prompts.")
    else:
        print(
            f"WARNING: no general-chat pool at {chat_path}. Skipping the "
            f"{MIXTURE[Kind.GENERAL_CHAT]:.0%} anti-forgetting slice — the fine-tune will "
            f"lose general ability. Run scripts/distill_general_chat.py first."
        )

    rng.shuffle(records)
    return records


def split_sft(records: list[dict], holdout_styles: set[str], val_frac: float,
              test_frac: float, seed: int) -> dict[str, list[dict]]:
    """
    Splits on TWO axes, because there are two ways to leak.

    split_by_case (reused from the HRM generator) stops the same case appearing
    in train and val. But the real risk here is PHRASING leakage: the hidden
    organizer prompts are written by someone who never saw our templates, so a
    val set drawn from the same style families measures memorisation, not
    generalisation. Every example in a held-out style goes to val_ood instead.
    """
    ood = [r for r in records if r["meta"].get("style") in holdout_styles]
    rest = [r for r in records if r["meta"].get("style") not in holdout_styles]
    splits = split_by_case(rest, val_frac, test_frac, seed)
    splits["val_ood"] = ood
    return splits


def _report(records: list[dict], splits: dict[str, list[dict]]) -> None:
    kinds = Counter(r["meta"]["kind"] for r in records)
    print(f"\nGenerated {len(records)} examples.")
    print("\nMixture:")
    for k, n in kinds.most_common():
        print(f"  {k:16} {n:6}  {100*n/len(records):5.1f}%")

    labels = Counter(r["meta"].get("condition_label") for r in records
                     if r["meta"]["kind"] == Kind.TRIAGE.value)
    print("\nTriage label balance:")
    for k, n in labels.most_common():
        print(f"  {k:44} {n:5}  {100*n/sum(labels.values()):5.1f}%")

    styles = Counter(r["meta"].get("style") for r in records if r["meta"].get("style"))
    print("\nVignette styles:")
    for k, n in styles.most_common():
        print(f"  {k:22} {n:6}  {100*n/sum(styles.values()):5.1f}%")

    print("\nSplits:")
    for name, rs in splits.items():
        print(f"  {name:8} {len(rs):6}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-triage", type=int, default=7500,
                    help="8-12k is the useful range; a 0.8B at r=16 saturates then memorises")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--holdout-styles", type=str, default=",".join(DEFAULT_HOLDOUT_STYLES))
    ap.add_argument("--chat-pool", type=str,
                    default=str(OUT_DIR / "general_chat.jsonl"))
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--include-extended", action="store_true",
                    help="add malaria/measles/anaemia/malnutrition classifications from "
                         "src/sft/extended_protocol.py. UNREVIEWED clinical logic -- off by "
                         "default; requires clinician sign-off before it trains a shipped model.")
    args = ap.parse_args()

    holdout = {s.strip() for s in args.holdout_styles.split(",") if s.strip()}
    unknown = holdout - {s.value for s in VignetteStyle}
    if unknown:
        raise SystemExit(f"--holdout-styles names unknown styles: {unknown}")

    records = generate(args.num_triage, args.seed, Path(args.chat_pool),
                       include_extended=args.include_extended)
    splits = split_sft(records, holdout, args.val_frac, args.test_frac, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rs in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")

    _report(records, splits)
    print(f"\nHeld-out styles (val_ood): {sorted(holdout)}")
    print("val_ood accuracy is the only number here that predicts the hidden prompts.")
    print(f"\nWrote {len(splits)} files to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
