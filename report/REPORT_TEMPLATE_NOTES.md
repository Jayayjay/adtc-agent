# Report Notes (placeholder until the official ADTC 2026 Report Template is dropped in)

## Required sections (per challenge brief)
- [ ] Problem definition and context
- [ ] Identified constraints (power, data, compute, connectivity)
- [ ] Documentation of design alternatives and final decisions
- [ ] Tools used and why they were chosen
- [ ] Performance tests and benchmarks
- [ ] Screenshots or short video of the build in action
- [ ] 2-minute solution/journey video

## Niche and problem definition

**Domain**: Healthcare & Medical. **Specific problem**: offline decision
support for community health workers applying the WHO/IMCI (Integrated
Management of Childhood Illness) protocol for children 2 months to 5 years.

**Why this niche, concretely**: Nigeria adopted IMCI in 1997; a Kano State
study found under 25% of first-level facilities had CHWs trained in IMCI, and
a digital clinical-decision-support pilot there measurably improved protocol
adherence. This gives the submission a real, sourced justification rather than
a rubric-driven guess -- offline, low-connectivity primary care is exactly
where this kind of tool has documented value.

**Scope boundary (safety-critical, stated explicitly)**: this is protocol-
following decision support, not diagnosis. It helps a trained health worker
correctly apply IMCI's published algorithm; it is not a source of independent
medical judgment and does not replace the official chart booklet, training, or
professional care.

## Current architecture decision: Option A (Qwen3.5-0.8B only)
Single model (Qwen3.5-0.8B, Q4_K_M GGUF, ~533MB), CPU-only inference via
llama.cpp. HRM (~27M params) scoped to ORCHESTRATION only (deciding which
symptom modules to assess and in what order for multi-symptom cases) --
NOT the medical classification itself. Classification is handled by a
deterministic rule engine (`src/tools/imci_protocol.py`) encoding the
published IMCI algorithm structure directly in code.

**Why classification is deterministic, not learned**: for a life-relevant
classification task, correctness should come from code matching a published
standard, not from a 27M/0.8B-parameter model's learned weights. This was a
late but important design pivot -- see the HRM docstrings in `src/hrm/` for
the full rationale.

## Design alternatives considered

### Model architecture (Qwen vs. hybrid)
| Option | Description | Status |
|---|---|---|
| A | Qwen3.5-0.8B only + HRM (orchestration) + deterministic rule engine | **chosen** -- lowest TPS/RAM risk, real benchmark numbers confirm large margin on Sperf/Seff |
| B | MiniCPM5-1B only + HRM | rejected -- higher TPS risk, no accuracy justification once classification moved to a deterministic engine |
| C | Qwen + MiniCPM hybrid on-demand routing | rejected -- unresolved Sperf measurement methodology risk, swap-latency risk, and moot once Sacc no longer depends on LLM reasoning quality for the core task |

### HRM's role (learned classification vs. orchestration-only)
| Option | Description | Status |
|---|---|---|
| HRM classifies directly | Fine-tune HRM to output IMCI classifications from symptom input | rejected -- unacceptable safety risk; no way to guarantee correctness against a published clinical standard from a small model's learned weights alone |
| HRM orchestrates, rule engine classifies | HRM decides assessment order/sequencing; `imci_protocol.py` (deterministic) produces the actual classification | **chosen** -- preserves the "hierarchical reasoning" architecture story for the challenge while keeping the safety-critical output verifiable and correct-by-construction |

## Real benchmark results (collected, not estimated)

**Hardware used**: Dell Latitude 7480, Intel i5-7300U (7th gen, 4 threads),
16GB RAM, Kali GNU/Linux Rolling.

**Caveat -- read before quoting these as final**: the ADTC spec calls for
10th-12th gen Intel or Ryzen 5 3000-5000; the i5-7300U is 7th gen and weaker.
These numbers should be framed as a conservative lower bound, not confirmed
reference-hardware results. The OS (Kali, not Ubuntu 22.04) is also a
mismatch worth disclosing. Re-test on in-spec hardware before final submission
if at all possible.

| Metric | Value | Formula check |
|---|---|---|
| Mean TPS | 19.34 (stdev 1.08, range 17.85-20.46) | 129% of TPS_REFERENCE (15.0) |
| Sperf | 100.0 | `100 * min(1, 19.34/15.0)` = 100 (capped) |
| Peak RAM | 954.86 MB | -- |
| Seff | 86.36 | `100 * (7000-954.86)/7000` = 86.36 ✓ matches reported value |
| Peak temp | 78°C | under 85°C limit |
| Throttling | Not detected (24.74 -> 24.26 TPS over 300s, ~2% drift) | Thermal penalty: 0 |

**Known artifact**: `bench_thermal.py`'s TPS readings (~24.5 avg) are higher
than `bench_tps.py`'s (~19.34 avg) because the thermal script repeats an
identical prompt, and llama.cpp likely caches/reuses the KV-cache prefix on
repeated identical prompts, inflating measured TPS. The thermal script's
early-vs-late comparison is still internally valid (both halves benefit
equally), but **19.34 from bench_tps.py is the correct Sperf input**, not
24.5 -- `export_metrics.py` already only reads from `tps_benchmark.json`, so
no code fix needed, just don't misquote the higher number in the report text.

**Fixed-component score contribution** (independent of Sacc):
`0.30 * Sperf + 0.20 * Seff = 0.30*100 + 0.20*86.36 = 47.27 points` out of a
possible 50 -- Sacc is the dominant remaining lever for total score.

## Open questions for organizers / before finalizing
- [ ] Is Sperf measured via fixed benchmark suite or sampled workload distribution?
- [ ] Confirm TPS_REFERENCE = 15.0 is final, not provisional.
- [ ] Confirm peak RAM measurement methodology (sampling interval, workload duration).
- [ ] Confirm the African-relevance/regional-impact rubric weighting so the
      problem-definition section can address it directly and proportionately.

## A real bug worth documenting in the report

During eval setup, a type-confusion bug caused genuine danger signs to be
silently missed: `danger_signs_present` was passed as a comma-separated
**string** in eval JSON (matching the tool's input format), but an earlier
version of `eval/scoring/sacc_scorer.py` constructed `ChildAssessment`
directly with that string instead of routing through the tool wrapper's
comma-split logic. Python dataclasses don't enforce types at runtime, so the
string was silently accepted and iterated character-by-character in the
danger-sign check -- meaning a case with `"convulsions"` present was
classified as if no danger signs existed, downgrading it from `severe` to
`mild`. Caught by the Sacc scorer itself (2 of 5 vignettes failed
unexpectedly), not by inspection.

**Fix applied**: (1) `ChildAssessment.__post_init__` now raises `TypeError`
immediately if `danger_signs_present` is a string instead of a list -- fails
loudly instead of silently producing a wrong classification; (2)
`sacc_scorer.py` now calls `IMCITriageTool.run()` directly instead of
duplicating input-parsing logic, so eval and the live agent share exactly one
parsing path.

This is worth including in the report almost verbatim -- it's a concrete,
honest example of taking safety seriously in a life-relevant domain: the
system caught its own bug via testing before it could produce a wrong
classification in a real case, and the fix targets the failure mode
(silent-wrong output) directly rather than just patching the one instance.


## HRM orchestration design (completed, expert policy working today)

**Design correction worth documenting**: the original scope ("HRM decides
which symptom module to check first") didn't survive contact with the real
IMCI algorithm -- the published protocol checks all reported symptom
categories in a fixed order, so "reordering modules" isn't a problem the
real protocol has. Revised scope: **adaptive question-asking** -- deciding
the single next question to ask a caretaker, given answers so far, stopping
the moment enough is known to classify (mirroring how IMCI is actually
taught: ask danger signs first, follow the chief complaint's branch, don't
keep asking once the outcome is already determined). This is a genuine
sequential decision problem, checkable against a ground truth, and a
legitimate use of HRM's dual-timescale shape (category-level strategy +
field-level execution) without duplicating the classification itself.

**Key implementation decision**: the "expert policy" (`src/hrm/expert_policy.py`)
that both generates HRM's training data AND serves as a working interim
orchestrator today is derived directly from `imci_protocol.assess()`'s own
branch structure -- not hand-written separately. This was validated with a
consistency test suite (`tests/unit/test_expert_policy.py`, including 20
randomized fuzz cases): for every case, the PARTIAL fields the policy
chooses to ask about are checked to reproduce the exact same classification
`assess()` would give if every field had been asked. This caught two real
bugs during development:
1. An early version kept asking about unrelated symptom categories after a
   result was already determined (e.g. continuing to ask about diarrhea
   after confirming stridor, which alone means urgent referral) -- fixed by
   recognizing that `assess()` is a sequential if-chain that returns from
   the FIRST category with a positive top-level symptom, not just severe ones.
2. (See "chart booklet sourcing" section above for the dehydration
   two-of-four-signs bug, caught the same way.)

**Training and validation pipeline (completed, real numbers)**:
`scripts/generate_hrm_training_data.py` rolls the expert policy out over
synthetic cases with a case-level train/val/test split (70/15/15 -- split by
`case_id`, not individual example, since turns from the same case share
heavy state overlap and would leak across splits otherwise). A baseline MLP
(`src/hrm/model.py`, ~10K params -- explicitly NOT the real Sapient HRM
architecture, see that file's docstring) trains via `scripts/train_hrm.py`
and is validated via `scripts/validate_hrm.py` on TWO levels, not one:
1. Action-level accuracy on the held-out TEST split
2. **End-to-end classification accuracy** -- actually running the trained
   model as the live orchestrator on fresh synthetic cases and checking
   whether the FINAL classification matches a full assessment. This is the
   metric that actually matters; action-level accuracy alone can hide
   cascading errors.

**A second real bug, caught specifically by the two-level validation gap**:
the first trained model hit 99.8% action-level accuracy but only 86.7%
end-to-end accuracy -- a real, informative gap, not noise. Root cause: the
expert policy's chief-complaint-based question REORDERING (asking about
what the caretaker mentioned first, for conversational flow) could cause
the dialogue to stop on a lower-precedence category before checking a
higher-precedence one that was also positive -- even though `assess()`
always evaluates categories in a FIXED order (cough, diarrhea, fever, ear)
regardless of what was asked first. Worse: **an existing unit test had
asserted this buggy behavior as correct**, without checking it against what
`assess()` on full information actually returns -- caught only because the
end-to-end validation gap prompted re-examining that test's assumption.

Fix went through two iterations:
1. First attempt: ask all 4 top-level questions before deciding which to
   deep-dive into. Correct, but wasteful (asks questions that can't affect
   the outcome once an earlier-precedence category is already known
   positive) -- caught by a stale test expectation.
2. Final: scan top-level questions in FIXED order, stop the scan the
   instant one is positive, deep-dive only into that category. Both correct
   and minimal. Chief-complaint-based reordering was removed entirely from
   the correctness-critical path (kept only as an informational feature in
   the state encoding) since it's fundamentally incompatible with safe
   short-circuiting.

After the fix: 150-case randomized fuzz test (varying both symptom
combinations AND chief-complaint text -- the original 20-case fuzz test
used a fixed empty chief-complaint and could never have caught this bug)
passes 150/150. Retrained model: 100% action-level accuracy, **100%
end-to-end classification accuracy on 1000 fresh unseen synthetic cases**.

This is worth including in the report close to verbatim -- it's a concrete
demonstration of validating at the right level of abstraction (outcomes,
not just intermediate predictions) and of a test suite that initially
encoded a wrong assumption as "expected," caught only once a stronger
validation signal (end-to-end accuracy) surfaced the discrepancy.

**Current status**: `src/hrm/state_machine.py`'s `HRMSession` supports both
the expert policy (`use_trained_model=False`, the default -- fully working
today) and the trained checkpoint (`use_trained_model=True`, real inference
now implemented, not a stub) as interchangeable orchestrators. Both pass the
same test suite.

**Known integration gap, honestly flagged**: `src/core/agent.py`'s
single-message contract doesn't yet extract `known_answers` from free text
via the LLM's structured output -- currently passes an empty dict, so every
REASONING-path message starts a fresh assessment from the first question.
Real multi-turn session persistence (keyed by session_id, surviving across
messages) and LLM-based field extraction from free text are both real,
separate pieces of work, flagged here rather than rushed.

## Chart booklet sourcing (completed)


Sourced and fetched the official WHO IMCI Chart Booklet, March 2014
(ISBN 978-92-4-150682-3) -- the standard global reference most country
adaptations, including Nigeria's, build from.
https://cdn.who.int/media/docs/default-source/mca-documents/child/imci-integrated-management-of-childhood-illness/imci-in-service-training/imci-chart-booklet.pdf

**What was confirmed correct**: the age-banded fast-breathing thresholds
(50+ breaths/min for 2-11 months, 40+ for 12-59 months) matched this
scaffold's earlier placeholder values exactly.

**What was corrected -- a real bug caught by sourcing**: the severe/some
dehydration logic previously checked two SPECIFIC signs (e.g. sunken eyes
AND slow skin pinch) as if both were required. The actual WHO protocol
requires ANY TWO of four signs at each severity level:
- Severe: lethargic/unconscious, sunken eyes, unable to drink/drinking
  poorly, skin pinch very slowly (>2s) -- any 2 of 4
- Some: restless/irritable, sunken eyes, drinking eagerly/thirsty, skin
  pinch slowly -- any 2 of 4

This is a meaningfully different (and more clinically sensitive) rule --
the old logic would have MISSED genuine severe-dehydration cases presenting
with a different pair of the four qualifying signs than the two hardcoded
ones. Fixed in `_classify_dehydration()`, with regression tests added
(`test_single_dehydration_sign_is_not_enough`,
`test_some_dehydration_two_of_four_signs`) to lock in the correct behavior.

**What remains explicitly out of scope** (documented in
`src/tools/imci_protocol.py`'s module docstring, not silently omitted):
malaria-risk-dependent fever branching and RDT-based testing, measles
complications, acute malnutrition, anaemia, HIV status, and the real
protocol's practice of returning multiple simultaneous classifications per
symptom category (this scaffold returns one primary classification with
secondary findings noted separately, e.g. dysentery/persistent diarrhoea
alongside a dehydration classification).

## Known unfinished pieces (be upfront about these in the report)
- Fever branch is a deliberately simplified stand-in for the real
  malaria-risk-area/RDT-dependent IMCI fever algorithm -- clearly flagged
  in both code and tool output, not silently passed off as complete.
- HRM orchestration layer has no trained checkpoint yet -- currently the
  `imci_triage` tool can be called directly without HRM sequencing multi-
  symptom cases; single-symptom cases work end-to-end today.
- Sacc scoring (`eval/scoring/sacc_scorer.py`) measures rule-engine
  classification accuracy against hand-built vignettes -- a proxy for the
  objective portion of Sacc, not the judge panel's qualitative assessment.
- Vector memory uses keyword retrieval, not real embeddings (deliberate
  RAM/complexity tradeoff -- see `src/memory/vector_store.py` docstring).
- Benchmarks collected on below-spec, wrong-OS hardware (see caveat above) --
  re-test on in-spec hardware before finalizing report numbers.
- Malnutrition, anaemia, HIV status, and immunization/vitamin A status
  branches from the full IMCI chart are not modeled at all (scope decision,
  documented, not a bug).

## Benchmark log
```
Date       | Hardware              | OS        | Mean TPS | Peak RAM (MB) | Peak Temp (C) | Throttled | Notes
-----------|------------------------|-----------|----------|----------------|----------------|-----------|------
2026-07-10 | Dell Latitude 7480,    | Kali      | 19.34    | 954.86         | 78             | No        | Below-spec CPU (7th gen); treat as
           | i5-7300U (7th gen)     | Rolling   |          |                |                |           | conservative lower bound, re-test on
           |                        |           |          |                |                |           | in-spec hardware before finalizing
```