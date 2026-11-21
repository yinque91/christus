"""Load the local auxiliary Transformer and rerank public knowledge."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TransformerSemantic:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.available = False
        self.metrics = {}
        self.model = None
        self.vocab = {}
        try:
            metrics = self.model_path.with_name("local_transformer_metrics.json")
            if metrics.exists():
                self.metrics = json.loads(metrics.read_text(encoding="utf-8"))
            self._load()
        except Exception as exc:
            self.metrics = {"error": f"{type(exc).__name__}: {exc}"}

    def _load(self):
        import sys
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "christus_desktop" / "training"))
        from local_transformer import TinyTransformer
        if not self.model_path.exists():
            return
        self.model, self.vocab = TinyTransformer.load(self.model_path)
        self.available = True

    def _tokens(self, text: str):
        ids = [self.vocab.get(ch, 1) for ch in str(text)[:32]]
        return np.asarray((ids + [0] * 32)[:32], dtype=np.int64)

    def similarity(self, query: str, text: str) -> float:
        if not self.available:
            return 0.0
        a = self.model.encode(self._tokens(query))
        b = self.model.encode(self._tokens(text))
        den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / den)

    def rerank(self, query: str, items: list[dict]) -> list[dict]:
        if not self.available or not items:
            return items
        scored = []
        for i, item in enumerate(items):
            text = str(item.get("title", "")) + " " + str(item.get("summary", ""))
            scored.append((self.similarity(query, text), i, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[2] for x in scored]

    def status(self):
        return {
            "available": self.available,
            "model": str(self.model_path),
            "final_loss": self.metrics.get("final_loss"),
            "initial_loss": self.metrics.get("initial_loss"),
            "documents": self.metrics.get("corpus_documents"),
            "internet_documents": self.metrics.get("internet_documents"),
            "params": self.metrics.get("params"),
            "role": "auxiliary semantic reranker; not the cognitive core",
        }
