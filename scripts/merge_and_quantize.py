"""
Turns a trained LoRA adapter into the final graded artifact: a Q4_K_M .gguf
matching the base model's exact quantization profile.

This is B3. It orchestrates the four steps that each have a way to silently ship
a worse model:

  1. merge   -- LoRA adapter + base fp16 -> merged fp16 HF weights.
                TRAP: tied embeddings. The base has no lm_head tensor; a careless
                merge materialises a duplicate 254M-param head and the gguf ships
                ~500MB fatter. We assert the merged model still ties them.
  2. convert -- convert_hf_to_gguf.py --no-mtp --outtype f16.
                TRAP: without --no-mtp the MTP block becomes a 25th layer and the
                tensor count / RAM no longer match the base.
  3. imatrix -- llama-imatrix over a calibration set mixing general text + IMCI
                vignettes. The base was imatrix-quantized (imatrix_unsloth.gguf);
                skipping this ships a model worse than the base before the
                fine-tune even counts.
  4. quantize-- llama-quantize Q4_K_M with the base's per-tensor overrides
                (ssm_alpha/beta=q8_0, attn_qkv/ssm_out=q5_K). A plain Q4_K_M is
                SMALLER because it's WORSE -- it crushes the SSM path the base
                deliberately protected.

Steps 2-4 shell out to the llama.cpp tooling staged during B0 (prebuilt
binaries + the converter's conversion/ package + a version-matched gguf-py).
Point --llama-cpp at that directory; this script does not re-download it.

Usage:
    python scripts/merge_and_quantize.py \
        --adapter models/lora_adapter \
        --llama-cpp /path/to/lcpp \
        --out models/Qwen3.5-0.8B-IMCI-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BASE = REPO / "models" / "Qwen3.5-0.8B-hf"
DEFAULT_REF_GGUF = REPO / "models" / "Qwen3.5-0.8B-Q4_K_M.gguf"

# The base's per-tensor overrides, from reading its header (see plan). Everything
# else follows llama.cpp's standard Q4_K_M.
TENSOR_TYPE_OVERRIDES = [
    ("attn_qkv", "q5_K"),
    ("ssm_out", "q5_K"),
    ("ssm_alpha", "q8_0"),
    ("ssm_beta", "q8_0"),
]


def _run(cmd: list[str], **kw) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def merge(adapter: Path, base: Path, out: Path) -> None:
    """LoRA + base -> merged fp16 HF weights, asserting embeddings stay tied."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/4] merging adapter {adapter} into base {base}")
    model = AutoModelForCausalLM.from_pretrained(str(base), torch_dtype=torch.float16)
    tied_before = model.config.tie_word_embeddings
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()

    # The trap: a merged model that no longer ties embeddings will write a
    # separate lm_head into the gguf, ~500MB of duplicate weights.
    assert model.config.tie_word_embeddings == tied_before, (
        "merge changed tie_word_embeddings -- the gguf will materialise a duplicate "
        "254M-param lm_head. Do not ship this."
    )
    has_lm_head = any("lm_head" in n and p.data_ptr() !=
                      model.get_input_embeddings().weight.data_ptr()
                      for n, p in model.named_parameters())
    assert not has_lm_head, "merged model has an untied lm_head tensor -- see above"

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(str(base)).save_pretrained(str(out))
    print(f"      merged fp16 -> {out}")


def convert(merged: Path, llama_cpp: Path, out_f16: Path) -> None:
    print(f"[2/4] convert -> f16 gguf (--no-mtp)")
    _run([sys.executable, llama_cpp / "convert_hf_to_gguf.py", merged,
          "--outfile", out_f16, "--outtype", "f16", "--no-mtp"],
         env=_env_with_gguf(llama_cpp))


def build_imatrix(f16: Path, llama_cpp: Path, calib: Path, out_imatrix: Path) -> None:
    print(f"[3/4] imatrix over {calib}")
    binary = _find_binary(llama_cpp, "llama-imatrix")
    _run([binary, "-m", f16, "-f", calib, "-o", out_imatrix, "--chunks", "80"],
         env=_ld_env(binary))


def quantize(f16: Path, llama_cpp: Path, imatrix: Path | None, out: Path) -> None:
    print(f"[4/4] quantize -> Q4_K_M with base per-tensor profile")
    binary = _find_binary(llama_cpp, "llama-quantize")
    cmd = [binary]
    for name, qtype in TENSOR_TYPE_OVERRIDES:
        cmd += ["--tensor-type", f"{name}={qtype}"]
    if imatrix and imatrix.exists():
        cmd += ["--imatrix", imatrix]
    else:
        print("      WARNING: no imatrix -- output will be worse than the base. "
              "The base was imatrix-quantized; supply --calib to match it.")
    cmd += [f16, out, "Q4_K_M"]
    _run(cmd, env=_ld_env(binary))


def verify(out: Path, ref_gguf: Path, llama_cpp: Path) -> None:
    """Confirm the final gguf matches the base's shape: 320 tensors, token_embd
    Q6_K, ~532MB, eos 248046."""
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    from gguf import GGUFReader

    r = GGUFReader(str(out))
    n = len(r.tensors)
    size_mb = out.stat().st_size / 1e6
    eos = r.get_field("tokenizer.ggml.eos_token_id")
    eos_val = eos.contents() if eos else None
    embd = {t.name: str(t.tensor_type).split(".")[-1] for t in r.tensors}.get("token_embd.weight")

    print(f"\n=== verification ===")
    print(f"  tensors:     {n}       (expect 320)")
    print(f"  size:        {size_mb:.1f} MB  (expect ~532)")
    print(f"  eos_token:   {eos_val}   (expect 248046 = <|im_end|>)")
    print(f"  token_embd:  {embd}     (expect q6_K)")
    ok = n == 320 and eos_val == 248046 and 480 < size_mb < 620
    print("  " + ("MATCHES base profile ✓" if ok else "!! DOES NOT MATCH — investigate before shipping"))


def _env_with_gguf(llama_cpp: Path) -> dict:
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(llama_cpp / "gguf-py") + ":" + env.get("PYTHONPATH", "")
    return env


def _ld_env(binary: Path) -> dict:
    import os
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(Path(binary).parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def _find_binary(llama_cpp: Path, name: str) -> Path:
    for cand in llama_cpp.rglob(name):
        if cand.is_file():
            return cand
    raise SystemExit(f"{name} not found under {llama_cpp} -- stage the llama.cpp release binaries there")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir from train_lora.py")
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--llama-cpp", required=True, help="dir with convert_hf_to_gguf.py, conversion/, gguf-py, and the release binaries")
    ap.add_argument("--calib", help="calibration text for imatrix (mix general + IMCI vignettes). Omit to skip imatrix (NOT recommended).")
    ap.add_argument("--out", default=str(REPO / "models" / "Qwen3.5-0.8B-IMCI-Q4_K_M.gguf"))
    ap.add_argument("--ref-gguf", default=str(DEFAULT_REF_GGUF))
    ap.add_argument("--work-dir", default=str(REPO / "models" / "merge_work"))
    args = ap.parse_args()

    llama_cpp = Path(args.llama_cpp)
    work = Path(args.work_dir)
    merged_dir = work / "merged_fp16"
    f16_gguf = work / "merged-f16.gguf"
    imatrix_path = work / "imatrix.dat"
    out = Path(args.out)

    merge(Path(args.adapter), Path(args.base), merged_dir)
    convert(merged_dir, llama_cpp, f16_gguf)
    if args.calib:
        build_imatrix(f16_gguf, llama_cpp, Path(args.calib), imatrix_path)
    quantize(f16_gguf, llama_cpp, imatrix_path if args.calib else None, out)
    verify(out, Path(args.ref_gguf), llama_cpp)

    print(f"\nfinal artifact: {out}")
    print("Next: scripts/bench_tps.py / bench_ram.py on THIS file (idle machine), then "
          "eval/scoring/model_sacc_scorer.py --model <this> --base "
          f"{args.ref_gguf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
