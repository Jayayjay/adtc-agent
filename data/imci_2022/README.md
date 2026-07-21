# 2022 IMCI chart booklet — parse & gap analysis

Structured source for extending the deterministic rule engine
(`src/tools/imci_protocol.py::assess` and `src/sft/extended_protocol.py`).
**Not clinician-reviewed.** See the `adtc-sft-corpus-built` memory for the
audit discipline any new branch must clear before it labels training data.

## Files
- `booklet_layout_text.txt` — full 79-page layout-mode text extraction (source of truth for transcription).
- `classifications.json` — hand-transcribed classification tables, tagged with `modeled_status`.
- `extract_summary.json` — per-page char/image counts (regenerate with `scripts/parse_imci_booklet.py`).

## Two booklets on disk (cross-checked)
- `data/2022 IMCI chart booklet_final.pdf` — **2022 South African national adaptation** (clean text). Transcribed here.
- `data/imci-chart-booklet.pdf` — **WHO generic March 2014** (scanned/OCR, chart pages use a +29 char-shifted font; decode in `scripts/`). This is the edition `imci_protocol.py` cites.

**Correction (2026-07-20):** an earlier draft assumed the two booklets diverged a lot.
Cross-checking the actual 2014 text shows they **agree far more than assumed** — SpO₂<90%
oximetry, amoxicillin first-line, the wheeze branch, persistent diarrhoea, and
dysentery+ciprofloxacin are all already in the 2014 generic. The real gap is
**engine-subset vs full booklet**, not 2014 vs 2022. So the Sacc-drift risk is small.

## Gap summary (50 classification rows)
| status | count | meaning |
|---|---|---|
| `modeled` | 9 | already in `assess()`, label matches |
| `extended` | 12 | already in `extended_protocol.py` (gated behind `--include-extended`, UNREVIEWED) |
| `engine_gap` | 18 | in **both** booklets but in neither engine — a coverage gap, not a version diff |
| `divergent` | 1 | engine models the branch, 2022 genuinely differs |
| `new_2022` | 10 | absent from the 2014 generic — genuine 2022-SA additions |

### Divergent (the only true 2014→2022 difference that touches a modeled branch)
- **fever** — 2014 label `VERY SEVERE FEBRILE DISEASE`; 2022 renames to **SUSPECTED MENINGITIS** and adds **bulging fontanelle** as a trigger. Underlying trigger (danger sign / stiff neck) unchanged.

### Engine gaps (in both booklets — safe to add without version worry)
- **cough:** `wheeze` sub-algorithm — engine models no wheeze.
- **diarrhoea:** persistent diarrhoea + dysentery as separate simultaneous classifications.
- **young infant (0–2 mo):** bacterial infection/jaundice/diarrhoea + coarse HIV — currently a **scope refusal**.

### Genuinely new in 2022-SA (not in the 2014 generic)
- **throat:** `streptococcal_sore_throat` (≥3 yr).
- **malnutrition:** `growth_problem` (RTHB weight curve), `overweight_obese`.
- **HIV:** the finer tiers (`suspected_symptomatic_hiv`, `possible_hiv_infection`, `hiv_infection_unlikely`) — 2014 has only coarse `CONFIRMED` / `HIV EXPOSED`.
- **young infant:** congenital-problems branch (macrocephaly, RPR/syphilis, cleft, …).

## How this feeds training (chosen path: extend the rule engine)
1. Clinician signs off each branch against the current national adaptation.
2. Encode signed-off branches into `assess()` (modeled/divergent) or `extended_protocol.py` (new), keeping the audit-via-re-`assess()` discipline.
3. Regenerate the synthetic SFT corpus the proven way (`scripts/generate_sft_data.py`).
   Booklet prose never reaches the weights directly.

**Open decision (now small):** only the fever label rename (`very_severe_febrile_disease`
→ `suspected_meningitis` + bulging fontanelle) is a true 2014→2022 difference on a modeled
branch. Decide whether to keep the 2014 label (grader-safe) or adopt 2022. Everything else
is additive (engine gaps + 2022-SA branches) and carries no Sacc-drift risk.
