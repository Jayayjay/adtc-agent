"""
Optional learned router upgrade: TF-IDF + logistic regression, trained on
your eval/tasks/ examples once you've labeled enough of them FAST vs
REASONING.

STATUS: not wired into the live agent. src/core/agent.py calls
src.router.rule_router.route() directly. Only switch over once:
  1. You have enough labeled examples (dozens, not a handful) to train on
  2. You've confirmed the rule-based router's mistakes are actually costing
     you Sacc points in your own eval runs -- don't add ML complexity to fix
     a problem you haven't confirmed exists.

Train via scripts/train_classifier.py, which will populate model.pkl in this
directory (gitignored).
"""

from __future__ import annotations

import pickle
from pathlib import Path

from src.router.rule_router import RoutePath, RouteDecision

MODEL_PATH = Path(__file__).parent / "model.pkl"


class IntentClassifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._vectorizer = None
        self._classifier = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No trained classifier at {self.model_path}. "
                "Run scripts/train_classifier.py first, or use rule_router.route() instead."
            )
        with open(self.model_path, "rb") as f:
            self._vectorizer, self._classifier = pickle.load(f)

    def predict(self, message: str) -> RouteDecision:
        if self._classifier is None:
            self.load()
        X = self._vectorizer.transform([message])
        pred = self._classifier.predict(X)[0]
        proba = self._classifier.predict_proba(X)[0].max()
        path = RoutePath.REASONING if pred == 1 else RoutePath.FAST
        return RouteDecision(path=path, reason=f"classifier confidence={proba:.2f}")
