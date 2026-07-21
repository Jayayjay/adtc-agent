"""
Extracts text from the 2022 IMCI chart booklet PDF for inspection.

This is a RECON script, not a training-data generator. The IMCI chart booklet
is a heavily tabular, multi-column, colour-coded document. pypdf's LAYOUT mode
preserves the three-column geometry (assess | classify | treat) via whitespace,
which is what makes the classification tables readable; plain-text mode
linearises the columns into garble.

The structured transcription of the classification tables lives in
data/imci_2022/classifications.json -- that was produced by reading the layout
text below, NOT by an automated cell parser, because this data drives a
safety-relevant rule engine and silent parse errors are unacceptable (see the
adtc-sft-corpus-built memory's audit-discipline note).

Usage:
    python scripts/parse_imci_booklet.py --pdf "data/2022 IMCI chart booklet_final.pdf"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pypdf import PdfReader

# pypdf emits "Rotated text discovered" via logging (the booklet's rotated
# section tabs), once per affected page -- it does not affect the body text.
logging.getLogger("pypdf").setLevel(logging.ERROR)

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "imci_2022"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(args.pdf)

    summary = {"pages": len(reader.pages), "per_page": []}
    chunks = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text(extraction_mode="layout")
        except Exception:
            text = page.extract_text() or ""
        try:
            images = len(page.images)
        except Exception:
            images = 0
        chars = len(text.strip())
        summary["per_page"].append({
            "page": i + 1, "chars": chars, "images": images,
            "likely_scanned": chars < 40 and images > 0,
        })
        chunks.append(f"\n\n===== PAGE {i + 1} =====\n{text}")

    (out / "booklet_layout_text.txt").write_text("".join(chunks))
    (out / "extract_summary.json").write_text(json.dumps(summary, indent=2))

    scanned = [p["page"] for p in summary["per_page"] if p["likely_scanned"]]
    total = sum(p["chars"] for p in summary["per_page"])
    print(f"pages: {summary['pages']}")
    print(f"total extracted chars: {total}")
    print(f"likely-scanned pages: {len(scanned)} {scanned}")
    print(f"wrote {out / 'booklet_layout_text.txt'} and {out / 'extract_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
