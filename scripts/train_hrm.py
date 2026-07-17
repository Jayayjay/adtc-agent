"""
Trains the orchestration policy (src/hrm/model.py's OrchestrationPolicyNet --
see that file's docstring for why this is a baseline MLP, not yet the real
HRM architecture) on data from scripts/generate_hrm_training_data.py.

Usage:
    python scripts/generate_hrm_training_data.py --num-cases 3000
    python scripts/train_hrm.py --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.hrm.model import OrchestrationPolicyNet, count_parameters

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "src" / "hrm" / "trained_models"


class JsonlActionDataset(Dataset):
    def __init__(self, path: Path):
        self.examples = []
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        x = torch.tensor(ex["state_vector"], dtype=torch.float32)
        y = torch.tensor(ex["action_index"], dtype=torch.long)
        return x, y


def compute_class_weights(dataset: JsonlActionDataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights -- the action distribution is
    naturally imbalanced (danger_signs_present and STOP appear in every
    trajectory; rare fields like ear_pain appear in ~1%), so an unweighted
    loss would bias the model toward always predicting the common classes."""
    counts = torch.zeros(num_classes)
    for ex in dataset.examples:
        counts[ex["action_index"]] += 1
    counts = torch.clamp(counts, min=1)  # avoid divide-by-zero for unseen classes
    weights = 1.0 / counts
    return weights / weights.sum() * num_classes  # normalize to mean ~1


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "hrm_training_data_train.jsonl"
    val_path = data_dir / "hrm_training_data_val.jsonl"

    if not train_path.exists() or not val_path.exists():
        print(f"Missing {train_path} or {val_path}. Run scripts/generate_hrm_training_data.py first.")
        return

    train_ds = JsonlActionDataset(train_path)
    val_ds = JsonlActionDataset(val_path)
    print(f"Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OrchestrationPolicyNet().to(device)
    print(f"Model parameter count: {count_parameters(model)} (baseline MLP -- see src/hrm/model.py docstring)")

    from src.hrm.dialogue_state import ACTION_SPACE
    class_weights = compute_class_weights(train_ds, len(ACTION_SPACE)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "orchestration_policy.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_ds)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d}/{args.epochs}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
            }, checkpoint_path)

    print(f"\nBest val accuracy: {best_val_acc:.4f}. Checkpoint saved to {checkpoint_path}")
    print("Next: python scripts/validate_hrm.py to check held-out TEST accuracy and "
          "end-to-end classification correctness (not just action-prediction accuracy).")


if __name__ == "__main__":
    main()