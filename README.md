# ADTC 2026 — Healthcare & Medical Submission (IMCI Triage Decision Support)

Local-first, offline decision-support tool for community health workers applying
the WHO/IMCI (Integrated Management of Childhood Illness) protocol for children
2 months to 5 years old. Runs entirely offline, CPU-only, targeting the ADTC
Standard Laptop reference profile (i5 10th-12th gen / Ryzen 5 3000-5000, 8GB
DDR4, Ubuntu 22.04).

**This is decision support, not diagnosis.** The system helps a health worker
correctly apply a known, published protocol — it is not a source of medical
judgment. See `src/tools/imci_protocol.py` for the full safety scope and
required-verification notes before this is anything beyond a competition
scaffold.

## Why this niche

Nigeria adopted IMCI in 1997, but adherence remains a real problem: a Kano
State study found under 25% of first-level facilities had community health
workers trained in IMCI, and a digital clinical-decision-support pilot there
measurably improved protocol adherence. That's the real-world case for this
submission — not a stretch to fit a rubric.

## Architecture (Option A: Qwen-only, safety-driven HRM scope)

```
User message (health worker's case notes)
     |
     v
 Router (src/router/rule_router.py)
     |
     +-- FAST path ------------> Qwen3.5-0.8B (conversational, non-clinical replies)
     |
     +-- REASONING path -------> HRM: assessment ORCHESTRATION only
                                       (which symptom modules to check, in what order)
                                       |
                                       v
                              src/tools/imci_protocol.py
                              DETERMINISTIC rule engine
                              (the actual classification)
                                       |
                                       v
                              Qwen formats the result into
                              a clear response for the health worker
```

**Key safety design decision**: HRM does NOT perform the medical classification
itself. That's deliberately handled by a deterministic rule engine encoding the
published IMCI algorithm structure (danger signs first, then symptom-specific
assessment, then traffic-light severity classification). HRM's role is
orchestration across multi-symptom cases — still a genuine use of its
dual-timescale (slow=planning, fast=execution) architecture, just not the
safety-critical component. See `src/hrm/encoders.py` for the full rationale.

MiniCPM5-1B was evaluated as a second-model hybrid option and deliberately
**not** included — see `report/REPORT_TEMPLATE_NOTES.md`. Plumbing for a
secondary model still exists in `src/llm/model_manager.py`, disabled by default.

## Status

- Router, tools (including the working IMCI rule engine), memory, and LLM
  client: **implemented and tested**.
- HRM orchestration layer: **stub**. Falls back gracefully to direct rule-engine
  classification (via the `imci_triage` tool) until the orchestration model is
  trained — see `src/core/agent.py`.
- Benchmarks: **real numbers collected**, see `report/data/*.json` and the
  Benchmarking section below.

Model weights are not included — see Quick Start below.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

bash scripts/download_models.sh      # pulls Qwen3.5-0.8B-Q4_K_M.gguf (~533MB)

python -m pytest tests/unit/ -v      # fast tests, no model required
python -c "
from src.tools.imci_protocol import ChildAssessment, assess
c = ChildAssessment(age_months=18, danger_signs_present=['convulsions'])
print(assess(c))
"                                     # sanity-check the rule engine directly

python -m src.main                   # interactive REPL (requires model downloaded)
```

## Benchmarking (real numbers, collected on a Dell Latitude 7480, i5-7300U, Kali Linux)

**Note on hardware**: i5-7300U is 7th-gen, below the ADTC-specified 10th-12th
gen range. Treat these as a conservative lower bound, not confirmed
reference-hardware numbers — see `report/REPORT_TEMPLATE_NOTES.md` for the
full caveat and re-test plan.

| Metric | Value |
|---|---|
| Mean TPS | 19.34 (range 17.85–20.46) |
| Sperf | 100.0 (capped; 129% of TPS_REFERENCE=15.0) |
| Peak RAM | 954.9 MB |
| Seff | 86.36 |
| Peak temp | 78°C |
| Thermal penalty | 0 (no throttling detected) |

```bash
python scripts/bench_tps.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf
python scripts/bench_ram.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf
python scripts/bench_ram_crosscheck.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf
python scripts/bench_thermal.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf --duration 300

python eval/scoring/sacc_scorer.py               # Sacc proxy on IMCI vignettes
python scripts/export_metrics.py --sacc <score>  # consolidated score summary
```

## Directory layout

```
adtc-agent/
├── src/
│   ├── main.py, config.py
│   ├── core/        # Agent orchestration, lifecycle, exceptions
│   ├── llm/          # Qwen client, model manager, formatter, IMCI-safety system prompt
│   ├── router/        # Rule-based router (medical/triage signals) + learned classifier (dormant)
│   ├── hrm/            # Orchestration-only reasoning core -- STUB, needs training
│   ├── tools/          # imci_protocol.py (rule engine) + imci_triage_tool.py (wrapper);
│   │                     calculator, datetime (generic support); arxiv/web_search (unused, network-dependent)
│   ├── memory/         # SQLite + keyword retrieval (real); embeddings.py (dormant upgrade)
│   └── utils/          # Resource monitoring, logging, ADTC scoring formulas
├── scripts/             # Model download, benchmarks (tps/ram/thermal + crosscheck), classifier training
├── tests/               # unit/ (fast, no model needed) + integration/ (needs model)
├── eval/                # eval/tasks/imci_triage.json (real vignettes) + working sacc_scorer.py
└── report/              # Report notes, real benchmark data, figures
```

## Next steps
1. Get 10-15 min on hardware actually in the 10th-12th gen / Ryzen 5 3000-5000
   range to close the CPU-generation gap in the current benchmarks.
2. ~~Source the actual official Nigeria/WHO IMCI chart booklet~~ **Done** --
   `src/tools/imci_protocol.py` now cites and implements verified thresholds
   from the WHO 2014 Chart Booklet (ISBN 978-92-4-150682-3). One real bug was
   caught and fixed in the process (dehydration classification logic) -- see
   `report/REPORT_TEMPLATE_NOTES.md`.
3. Design and train HRM's orchestration model (`src/hrm/encoders.py` has the
   scoped-down design target: assessment ordering, not classification).
4. Expand `eval/tasks/imci_triage.json` with more vignettes, ideally reviewed
   by someone with clinical/IMCI training -- especially for the branches
   explicitly NOT modeled yet (malaria-risk fever logic, malnutrition,
   anaemia, HIV status).
5. Fill in the report with real benchmark numbers (already collected) and the
   design-alternatives writeup (already drafted in REPORT_TEMPLATE_NOTES.md).
