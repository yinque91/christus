"""Tiny local Transformer encoder (NumPy) for auxiliary semantic retrieval.

This is a real single-layer, multi-head self-attention encoder with an MLM head,
trained from scratch. It is NOT the Christus cognitive core and cannot write
memory/affect/action. It only reranks public-source knowledge by similarity.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class TinyTransformer:
    def __init__(self, vocab_size: int, seq_len: int = 32, d: int = 32,
                 heads: int = 2, ff: int = 64, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.V, self.T, self.d, self.H = vocab_size, seq_len, d, heads
        self.dh = d // heads
        self.E = rng.normal(0, 0.08, (vocab_size, d))
        self.P = rng.normal(0, 0.02, (seq_len, d))
        self.Wq = rng.normal(0, 0.08, (d, d)); self.bq = np.zeros(d)
        self.Wk = rng.normal(0, 0.08, (d, d)); self.bk = np.zeros(d)
        self.Wv = rng.normal(0, 0.08, (d, d)); self.bv = np.zeros(d)
        self.Wo = rng.normal(0, 0.08, (d, d)); self.bo = np.zeros(d)
        self.W1 = rng.normal(0, 0.08, (d, ff)); self.b1 = np.zeros(ff)
        self.W2 = rng.normal(0, 0.08, (ff, d)); self.b2 = np.zeros(d)
        self.lm_b = np.zeros(vocab_size)

    @property
    def params(self):
        return [self.E, self.P, self.Wq, self.bq, self.Wk, self.bk,
                self.Wv, self.bv, self.Wo, self.bo, self.W1, self.b1,
                self.W2, self.b2, self.lm_b]

    def _softmax(self, x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    def forward(self, tokens, mask=None):
        B, T = tokens.shape
        e = self.E[tokens] + self.P[:T][None, :, :]
        q = e @ self.Wq + self.bq
        k = e @ self.Wk + self.bk
        v = e @ self.Wv + self.bv
        q = q.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)
        scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(self.dh)
        att = self._softmax(scores, axis=-1)
        ctx = att @ v
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, self.d)
        o = ctx @ self.Wo + self.bo
        r1 = e + o
        f = np.maximum(0, r1 @ self.W1 + self.b1)
        r2 = r1 + f @ self.W2 + self.b2
        logits = r2 @ self.E.T + self.lm_b
        cache = (tokens, e, q, k, v, att, ctx, o, r1, f, r2, logits,
                 mask if mask is not None else np.zeros_like(tokens, bool))
        return logits, cache

    def loss_grad(self, tokens, mask, labels=None):
        logits, cache = self.forward(tokens, mask)
        tokens_, e, q, k, v, att, ctx, o, r1, f, r2, logits, mask = cache
        labels = tokens if labels is None else labels
        B, T = tokens.shape
        V = self.V
        sel = np.where(mask)
        z = logits[sel]
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        y = labels[sel]
        loss = float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean())
        dlogits = np.zeros_like(logits)
        dp = p.copy(); dp[np.arange(len(y)), y] -= 1.0
        dp /= max(1, len(y))
        dlogits[sel] = dp
        # LM head / tied embedding
        dE = np.zeros_like(self.E)
        dlm = dlogits.sum(axis=(0, 1))
        dE += dlogits.reshape(-1, V).T @ r2.reshape(-1, self.d)
        dr2 = dlogits @ self.E
        # FFN
        dW2 = f.reshape(-1, f.shape[-1]).T @ dr2.reshape(-1, self.d)
        db2 = dr2.sum(axis=(0, 1))
        df = dr2 @ self.W2.T
        dpre = df * (f > 0)
        dW1 = r1.reshape(-1, self.d).T @ dpre.reshape(-1, f.shape[-1])
        db1 = dpre.sum(axis=(0, 1))
        dr1 = dr2 + dpre @ self.W1.T
        # attention
        dctx = dr1 @ self.Wo.T
        dWo = ctx.reshape(-1, self.d).T @ dr1.reshape(-1, self.d)
        dbo = dr1.sum(axis=(0, 1))
        dctx = dctx.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)
        datt = dctx @ v.transpose(0, 1, 3, 2)
        dv = att.transpose(0, 1, 3, 2) @ dctx
        ds = datt * att
        ds = ds - att * ds.sum(axis=-1, keepdims=True)
        ds /= math.sqrt(self.dh)
        dq = ds @ k
        dk = ds.transpose(0, 1, 3, 2) @ q
        # reshape back
        dq = dq.transpose(0, 2, 1, 3).reshape(B, T, self.d)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, T, self.d)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, T, self.d)
        de = dr1 + dq @ self.Wq.T + dk @ self.Wk.T + dv @ self.Wv.T
        dWq = e.reshape(-1, self.d).T @ dq.reshape(-1, self.d); dbq = dq.sum(axis=(0, 1))
        dWk = e.reshape(-1, self.d).T @ dk.reshape(-1, self.d); dbk = dk.sum(axis=(0, 1))
        dWv = e.reshape(-1, self.d).T @ dv.reshape(-1, self.d); dbv = dv.sum(axis=(0, 1))
        np.add.at(dE, tokens.reshape(-1), de.reshape(-1, self.d))
        dP = np.zeros_like(self.P); dP[:T] += de.sum(axis=0)
        grads = [dE, dP, dWq, dbq, dWk, dbk, dWv, dbv, dWo, dbo,
                 dW1, db1, dW2, db2, dlm]
        return loss, grads

    def save(self, path: Path, vocab: dict):
        arrays = {}
        for name in ("E", "P", "Wq", "bq", "Wk", "bk", "Wv", "bv",
                     "Wo", "bo", "W1", "b1", "W2", "b2", "lm_b"):
            arrays[name] = getattr(self, name)
        np.savez(path, **arrays)
        path.with_suffix(".vocab.json").write_text(
            json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        data = np.load(path, allow_pickle=False)
        vocab = json.loads(path.with_suffix(".vocab.json").read_text(encoding="utf-8"))
        obj = cls(len(vocab), seq_len=int(data["P"].shape[0]),
                  d=int(data["E"].shape[1]), seed=1)
        for name in ("E", "P", "Wq", "bq", "Wk", "bk", "Wv", "bv",
                     "Wo", "bo", "W1", "b1", "W2", "b2", "lm_b"):
            setattr(obj, name, data[name])
        return obj, vocab

    def encode(self, tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64)[None, :]
        _, cache = self.forward(tokens)
        return cache[10][0].mean(axis=0)  # r2 mean
