"""Train the local auxiliary Transformer encoder on the local/internet corpus.

No external model, no W/P/Q update. The model is used only for semantic reranking
of public knowledge; it does not generate Christus's identity or modify affect.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from local_transformer import TinyTransformer

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "christus_desktop" / "training" / "corpus.jsonl"
MODEL = ROOT / "christus_platform" / "models" / "local_transformer.npz"
META = ROOT / "christus_platform" / "models" / "local_transformer_metrics.json"
SEQ = 32


def build_vocab(texts):
    from collections import Counter
    c = Counter()
    for t in texts:
        c.update(t)
    chars = [ch for ch, n in c.most_common() if n >= 2][:6000]
    vocab = {"<pad>": 0, "<unk>": 1, "<mask>": 2}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def main():
    rows = [json.loads(x) for x in CORPUS.read_text(encoding="utf-8").splitlines()]
    texts = [r["text"] for r in rows]
    vocab = build_vocab(texts)
    inv = {v: k for k, v in vocab.items()}
    stream = "".join(texts)
    ids = np.asarray([vocab.get(ch, 1) for ch in stream], dtype=np.int64)
    model = TinyTransformer(len(vocab), seq_len=SEQ, d=32, heads=2, ff=64, seed=42)
    # Adam
    m = [np.zeros_like(p) for p in model.params]
    v = [np.zeros_like(p) for p in model.params]
    lr, b1, b2, eps = 0.003, 0.9, 0.999, 1e-8
    rng = random.Random(42)
    metrics = {"steps": [], "vocab": len(vocab), "seq": SEQ, "params":
               int(sum(p.size for p in model.params))}
    losses = []
    t0 = time.time()
    for step in range(1, 1001):
        batch = []
        masks = []
        for _ in range(8):
            start = rng.randrange(0, max(1, len(ids) - SEQ - 1))
            x = ids[start:start + SEQ].copy()
            mask = np.zeros(SEQ, dtype=bool)
            idx = np.array([rng.randrange(SEQ) for _ in range(max(2, SEQ // 7))])
            mask[idx] = True
            batch.append(x.copy()); masks.append(mask)
        originals = np.stack(batch)
        tokens = originals.copy()
        for b in range(len(tokens)):
            tokens[b, masks[b]] = vocab["<mask>"]
        mask = np.stack(masks)
        loss, grads = model.loss_grad(tokens, mask, originals)
        for i, (p, g) in enumerate(zip(model.params, grads)):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * (g * g)
            mh = m[i] / (1 - b1 ** step)
            vh = v[i] / (1 - b2 ** step)
            p -= lr * mh / (np.sqrt(vh) + eps)
        losses.append(loss)
        if step % 25 == 0:
            print(f"step {step} loss {loss:.4f} elapsed {time.time()-t0:.1f}s", flush=True)
            metrics["steps"].append({"step": step, "loss": round(loss, 5)})
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL, vocab)
    metrics.update({"final_loss": float(losses[-1]),
                    "initial_loss": float(losses[0]),
                    "train_seconds": round(time.time() - t0, 2),
                    "corpus_documents": len(rows),
                    "internet_documents": sum(1 for r in rows if r["kind"] == "internet_knowledge"),
                    "licensed_dialogue_documents": sum(1 for r in rows if r["kind"] == "licensed_dialogue"),
                    "model": str(MODEL),
                    "note": "auxiliary semantic encoder; not the cognitive core"})
    META.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("initial_loss", "final_loss",
                                              "train_seconds", "vocab", "params")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
