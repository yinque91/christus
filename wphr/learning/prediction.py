"""Online prediction learners (MASTER SPEC 5.4: prediction-error learning).

Two mechanisms live here:

* :class:`LinearPredictor` — a bounded normalised-LMS one-step predictor used by
  M1. It learns ``x[t+1]`` from ``x[t]`` and reports its own error, so
  "prediction error decreased" is a measured quantity rather than a narrative.
* :class:`BoundedHebbian` — a local associative update with weight bounds and
  row normalisation, kept for later stages and tested for stability here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wphr.util.numerics import clip_array


class LinearPredictor:
    """Bounded one-step-ahead linear predictor with normalised LMS updates."""

    def __init__(self, dim: int, *, lr: float = 0.03, weight_bound: float = 4.0,
                 output_bound: float = 10.0) -> None:
        if dim < 1:
            raise ValueError("predictor dim must be >= 1")
        self.dim = int(dim)
        self.lr = float(lr)
        self.weight_bound = float(weight_bound)
        self.output_bound = float(output_bound)
        self.W = np.zeros((dim, dim), dtype=np.float64)
        self.b = np.zeros(dim, dtype=np.float64)
        self.updates = 0
        self.last_error = 0.0
        self.error_ema = 0.0
        self.ema_alpha = 0.1

    def predict(self, x: np.ndarray) -> np.ndarray:
        vec = np.asarray(x, dtype=np.float64).reshape(-1)
        if vec.size != self.dim:
            raise ValueError(f"expected {self.dim} inputs, got {vec.size}")
        return np.clip(self.W @ vec + self.b, -self.output_bound, self.output_bound)

    def observe(self, x: np.ndarray, target: np.ndarray) -> float:
        """Predict-then-learn on one transition; returns the prediction error."""
        target_vec = np.asarray(target, dtype=np.float64).reshape(-1)
        prediction = self.predict(x)
        error = target_vec - prediction
        miss = float(np.mean(np.abs(error)))
        self.last_error = miss
        self.error_ema = (1.0 - self.ema_alpha) * self.error_ema + self.ema_alpha * miss
        self._update(np.asarray(x, dtype=np.float64).reshape(-1), error)
        return miss

    def _update(self, x: np.ndarray, error: np.ndarray) -> None:
        norm = float(x @ x) + 1.0
        self.W += self.lr * np.outer(error, x) / norm
        self.b += self.lr * error / norm
        clip_array(self.W, self.weight_bound)
        clip_array(self.b, self.weight_bound)
        self.updates += 1

    def snapshot(self) -> dict[str, np.ndarray]:
        return {"W": self.W.copy(), "b": self.b.copy()}

    def restore(self, payload: dict[str, np.ndarray]) -> None:
        if payload["W"].shape != self.W.shape:
            raise ValueError("predictor checkpoint shape mismatch")
        self.W = np.array(payload["W"], dtype=np.float64, copy=True)
        self.b = np.array(payload["b"], dtype=np.float64, copy=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "lr": self.lr,
            "updates": self.updates,
            "last_error": round(self.last_error, 8),
            "error_ema": round(self.error_ema, 8),
        }


class BoundedHebbian:
    """Local associative learning with bounds and row normalisation.

    Not used by the V0.2 policy path; implemented and tested so that later
    stages can enable it without inventing an untested update rule.
    """

    def __init__(self, n_in: int, n_out: int, *, lr: float = 0.01,
                 weight_bound: float = 1.0, decay: float = 1e-4) -> None:
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.lr = float(lr)
        self.weight_bound = float(weight_bound)
        self.decay = float(decay)
        self.W = np.zeros((self.n_out, self.n_in), dtype=np.float64)
        self.updates = 0

    def update(self, pre: np.ndarray, post: np.ndarray) -> None:
        pre_vec = np.asarray(pre, dtype=np.float64).reshape(-1)
        post_vec = np.asarray(post, dtype=np.float64).reshape(-1)
        if pre_vec.size != self.n_in or post_vec.size != self.n_out:
            raise ValueError("BoundedHebbian input size mismatch")
        self.W += self.lr * np.outer(post_vec, pre_vec) - self.decay * self.W
        clip_array(self.W, self.weight_bound)
        norms = np.linalg.norm(self.W, axis=1, keepdims=True)
        np.divide(self.W, np.maximum(norms, 1e-9), out=self.W, where=norms > 1.0)
        self.updates += 1

    def spectral_radius(self) -> float:
        from wphr.util.numerics import spectral_radius

        return spectral_radius(self.W) if min(self.W.shape) else 0.0
