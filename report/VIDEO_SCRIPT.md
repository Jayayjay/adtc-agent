# ADTC 2026 — 2-Minute Solution Video Script

**Model:** Qwen3.5-0.8B-IMCI-Q4_K_M · **Domain:** healthcare_medical · **Target:** ≤ 2:00

Format: screen-recording of the live terminal demo + 3–4 simple title slides. Narration
is ~290 words (≈150 wpm). Record the terminal demo first (`report/data/demo_transcript.txt`
is the exact session), then voice over.

---

| Time | On screen | Narration |
|---|---|---|
| **0:00–0:15** | Slide 1: title + a photo/illustration of a rural clinic. Text: "IMCI triage, offline, on a budget laptop." | "In much of rural Africa the first — and often only — point of care for a sick child is a community health worker with a paper chart and no internet. The WHO IMCI protocol turns a child's signs into a classification and an action: treat, follow up, or refer. Applying it correctly, under pressure, is hard." |
| **0:15–0:35** | Slide 2: a single box labelled "raw .gguf → llama.cpp". Everything else greyed out. | "The challenge grades one thing: a single model file, run offline through llama.cpp — no Python, no tools, no internet. So the whole IMCI protocol has to live inside the model's weights." |
| **0:35–0:58** | Slide 3: arrow diagram — "rule engine + orchestrator" crossed out at runtime, redrawn at "build time → training data → weights". | "We started with an agent: a learned orchestrator asking the next question, and a deterministic rule engine — real WHO chart logic in code — making the call. But that engine is Python; the grader never runs it. So we moved it to build time: the rule engine labels and scores the training data, and its logic is distilled into the weights. Correctness still traces to a published standard — just baked into the file the grader actually runs." |
| **0:58–1:28** | Live terminal: run cases 1–3 from the demo (pneumonia → treat, convulsions → refer, dehydration → treat). Let the real output type out. | "Here it is running fully offline. A nine-month-old coughing at 58 breaths a minute — it applies the age band and says pneumonia, treat and follow up. A two-year-old with convulsions — it catches the danger sign and refers urgently. Diarrhoea with sunken eyes and a slow skin pinch — some dehydration, ORS. Before fine-tuning, the base model didn't even know IMCI — it invented an ICD-10 code." |
| **1:28–1:50** | Slide 4: three big numbers — "100% core accuracy · 22 tok/s · 857 MB RAM". | "On the core triage task it scores 100 percent — right classification, right format, zero under-triage. It runs at 22 tokens a second in under a gigabyte of RAM, well inside the budget, with no internet." |
| **1:50–2:00** | Slide 5: map of Nigeria/Kano; one line of honest-limitation text. | "Built for the Nigerian context, where offline is the constraint that makes it usable at all. It's decision support for a trained worker — honest about its limits, and grounded in the WHO standard." |

---

## Recording checklist
- [ ] Screen-record the terminal demo (use the exact prompts in `report/data/demo_transcript.txt`).
- [ ] Make 5 title slides (or use plain full-screen text) matching the "On screen" column.
- [ ] Record narration; keep total ≤ 2:00 (trim the demo take, not the narration, if long).
- [ ] Show the model running with **wifi off** on screen for at least one case (proves offline).
- [ ] Export ≤ 1080p MP4; put the link in `REPORT.md` and the DevPost submission.

## Notes
- Keep the "distilled into the weights" line — it's the design-decision judges are looking for.
- Do **not** overclaim the extended branches (malaria/measles/HIV/young-infant) on camera; the
  100% number is the *core* task. The honest-limitation line at the end covers this.
