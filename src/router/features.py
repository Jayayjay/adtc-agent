"""
Feature extraction for the (optional, currently unused) learned router
upgrade. TF-IDF is deliberately chosen over embeddings here -- it's cheap
enough to run on every request without eating into your TPS/RAM budget,
unlike loading a separate embedding model just for routing decisions.
"""

from __future__ import annotations


def extract_features(message: str, requested_tools: list[str] | None = None) -> dict:
    """
    Lightweight, non-ML features that can also be fed into the rule router
    for debugging/logging, or used as additional columns alongside TF-IDF
    vectors when/if you train intent_classifier.py.
    """
    requested_tools = requested_tools or []
    words = message.split()
    return {
        "word_count": len(words),
        "char_count": len(message),
        "num_tools_mentioned": len(requested_tools),
        "has_question_mark": "?" in message,
        "has_numbers": any(ch.isdigit() for ch in message),
    }
