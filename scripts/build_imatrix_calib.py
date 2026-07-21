"""
Builds data/sft/imatrix_calib.txt -- the calibration text llama-imatrix uses to
quantize the fine-tuned model (scripts/merge_and_quantize.py --calib).

Why it must match the corpus: imatrix records per-tensor activation statistics
over representative text, and the quantizer uses them to spend bits where the
model actually computes. Calibrating on a distribution the model was NOT trained
to produce (e.g. the old corpus, before treatment/dosing/extended/young-infant)
wastes precision on the wrong tensors. So this is regenerated whenever the SFT
corpus is regenerated.

Content: every user and assistant message from the training split, one chunk per
message, blank-line separated -- the exact distribution the fine-tune emits and
is prompted with. System prompts are skipped (boilerplate).

Usage:
    python scripts/build_imatrix_calib.py            # reads data/sft/train.jsonl
    python scripts/build_imatrix_calib.py --in data/sft/train.jsonl --out data/sft/imatrix_calib.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(REPO / "data" / "sft" / "train.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data" / "sft" / "imatrix_calib.txt"))
    args = ap.parse_args()

    chunks: list[str] = []
    for line in Path(args.inp).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for m in rec["messages"]:
            if m["role"] == "system":
                continue
            text = m["content"].replace("\n", " ").strip()
            if text:
                chunks.append(text)

    Path(args.out).write_text("\n\n".join(chunks) + "\n")
    print(f"wrote {args.out}: {len(chunks)} chunks from {args.inp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
