"""
SFT data generation for the ADTC 2026 submission.

Context: the competition's evaluator runs a single raw .gguf through
llama.cpp and nothing else -- no Python, no tool calls. So the IMCI knowledge
that src/tools/imci_protocol.py encodes in code has to be moved into the
model's weights. This package turns assess()'s deterministic, WHO-sourced
classifications into natural-language (vignette -> answer) training pairs.

The rule engine stays the ground truth: it LABELS the data here, and it
scores the fine-tuned model in eval/. The model learns to imitate it.
"""
