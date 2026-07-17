"""
Trains the optional learned router upgrade (TF-IDF + logistic regression) on
labeled examples from eval/tasks/. Only run this once you have real labeled
data -- see src/router/intent_classifier.py's docstring for when this is
actually worth doing.

Expects eval/tasks/*.json files with entries like:
    {"message": "...", "label": "fast" | "reasoning"}

Usage:
    python scripts/train_classifier.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL_TASKS_DIR = Path(__file__).resolve().parent.parent / "eval" / "tasks"
MODEL_OUT = Path(__file__).resolve().parent.parent / "src" / "router" / "model.pkl"


def load_labeled_examples() -> tuple[list[str], list[int]]:
    messages, labels = [], []
    for path in EVAL_TASKS_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        for entry in data.get("examples", []):
            if "message" in entry and "label" in entry:
                messages.append(entry["message"])
                labels.append(1 if entry["label"] == "reasoning" else 0)
    return messages, labels


def main():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("scikit-learn not installed. Run: pip install scikit-learn --break-system-packages")
        return

    messages, labels = load_labeled_examples()
    if len(messages) < 20:
        print(
            f"Only {len(messages)} labeled examples found in {EVAL_TASKS_DIR}. "
            "Need substantially more (dozens+) before training is worthwhile -- "
            "the rule-based router (src/router/rule_router.py) is a better bet "
            "with this little data. Add 'label': 'fast'|'reasoning' to your "
            "eval/tasks/*.json examples and re-run."
        )
        return

    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(messages)

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X, labels)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump((vectorizer, classifier), f)

    print(f"Trained on {len(messages)} examples. Saved to {MODEL_OUT}")
    print("Train accuracy:", classifier.score(X, labels))
    print(
        "\nNote: this is train accuracy, not held-out accuracy -- with this few "
        "examples, treat it as a rough sanity check, not a real eval metric."
    )


if __name__ == "__main__":
    main()
