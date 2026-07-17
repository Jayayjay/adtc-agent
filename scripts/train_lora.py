"""
LoRA fine-tune of Qwen3.5-0.8B on the IMCI SFT corpus.

This is the SAME script for the laptop rehearsal and the donated GPU -- only
--max-steps and the device differ. The rehearsal's whole job is to prove the
pipeline (data -> tokeniser -> LoRA attach -> a few steps -> save -> merge)
works here, on CPU, so nothing structural fails on the 5-hour GPU budget.

Verified facts baked in (see the plan's "Verified facts" section):
  - Base HF weights: models/Qwen3.5-0.8B-hf (converted round-trip cleared B0).
  - Hybrid Mamba/attention: 24 layers = 18 linear-attn + 6 full-attn, plus an
    MTP block at layer 24 that --no-mtp drops. LoRA must NOT touch layer 24, or
    the adapter trains weights that get discarded at conversion.
  - Tied embeddings (no lm_head tensor). Never put embed_tokens in
    modules_to_save, or merge materialises a duplicate 254M-param head.
  - Chat template emits <|im_start|>assistant\n<think>\n\n</think>\n\n before
    the answer; completion-only masking keys on exactly that string.

Usage (rehearsal):
    python scripts/train_lora.py --max-steps 20 --max-seq-len 512 \
        --output-dir /tmp/lora_rehearsal
Usage (GPU):
    python scripts/train_lora.py --epochs 2 --bf16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO / "models" / "Qwen3.5-0.8B-hf"
DEFAULT_DATA = REPO / "data" / "sft"

# Resolved from the real HF module names (see plan). peft matches on suffix.
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # 6 full-attention layers
    "in_proj_qkv", "in_proj_z", "out_proj",   # 18 SSM layers
    "gate_proj", "up_proj", "down_proj",       # all layers
]
# Layer 24 is the MTP block, dropped at --no-mtp convert. Training an adapter on
# it is wasted compute that never ships.
TRANSFORM_LAYERS = list(range(24))

# Exact inference-time assistant prefix; completion-only loss masks everything
# up to and including it.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_split(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_text(record: dict, tokenizer) -> str:
    """Renders one record's messages through the chat template, thinking off."""
    return tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--output-dir", default=str(REPO / "models" / "lora_adapter"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="rehearsal: set small (e.g. 20). GPU: leave -1 to use --epochs.")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512,
                    help="corpus max is 332 tokens; 512 is ample and halves attention cost vs 1024")
    ap.add_argument("--bf16", action="store_true", help="A100/newer GPUs")
    ap.add_argument("--fp16", action="store_true", help="T4 and most Colab GPUs (native fp16)")
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--freeze-ssm", action="store_true",
                    help="fallback if Mamba autograd misbehaves: LoRA attention+FFN only")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # trl accepts a datasets.Dataset or a plain list[dict]. Prefer the list to
    # avoid the datasets dependency (pyarrow/dill/multiprocess), which is a large
    # download the corpus does not otherwise need.
    try:
        from datasets import Dataset
    except ImportError:
        Dataset = None

    # Accept either a local dir (laptop: models/Qwen3.5-0.8B-hf) or a HF hub id
    # (Colab: Qwen/Qwen3.5-0.8B). Only reject a local-looking path that's absent.
    model_ref = args.model
    if ("/" in model_ref and Path(model_ref).exists()) or Path(model_ref).exists():
        model_ref = str(Path(model_ref))
    elif Path(model_ref).parent != Path(model_ref) and Path(model_ref).drive == "" \
            and model_ref.startswith(("/", "./", "../", "models/")):
        raise SystemExit(f"local model dir not found: {model_ref}")

    if args.fp16 and args.bf16:
        raise SystemExit("pick one of --fp16 / --bf16")
    dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32
    print(f"loading tokenizer + model from {model_ref} (dtype={dtype}) ...")

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    targets = list(LORA_TARGETS)
    if args.freeze_ssm:
        targets = [t for t in targets if t not in ("in_proj_qkv", "in_proj_z", "out_proj")]
        print(f"  --freeze-ssm: LoRA targets reduced to {targets}")

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=targets,
        layers_to_transform=TRANSFORM_LAYERS,   # excludes the MTP block at layer 24
        modules_to_save=[],                       # tied embeddings: never save embed_tokens
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    train = load_split(Path(args.data_dir) / "train.jsonl")
    rows = [{"text": build_text(r, tokenizer)} for r in train]
    ds = Dataset.from_list(rows) if Dataset is not None else rows
    print(f"  train examples: {len(rows)}")

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_strategy="no",
        max_length=args.max_seq_len,
        packing=False,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        dataset_text_field="text",
        completion_only_loss=True,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tokenizer)
    trainer.train()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nadapter saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
