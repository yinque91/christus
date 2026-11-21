"""Live view of the REAL WPHR subject (wphr.runtime.subject), stepping it in a window.

Does not modify the subject or its config: it only calls the public step() API.
"""

from __future__ import annotations

import sys
import tkinter as tk
from collections import Counter, deque
from pathlib import Path

ROOT = Path(r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR")
sys.path.insert(0, str(ROOT))

from wphr.config import Config  # noqa: E402
from wphr.memory.store import Store  # noqa: E402
from wphr.runtime.subject import Subject  # noqa: E402


class Live:
    def __init__(self, root: tk.Tk, steps_per_tick: int = 3):
        self.root = root
        root.title("WPHR 主体 · 实时行动")
        root.geometry("760x480")
        root.attributes("-topmost", True)

        self.text = tk.Text(root, font=("Consolas", 12), bg="#0b0f0b", fg="#7CFC00")
        self.text.pack(fill="both", expand=True)

        cfg = Config.load()
        store = Store(cfg.resolve_path("paths.database"))
        self.subject = Subject(cfg, store=store, seed=None, persist=True)
        self.subject.initialise()

        self.n = steps_per_tick
        self.last = deque(maxlen=14)
        self.actions = Counter()
        self.mem_before = self._memory_count(store)
        self.step_once()

    def _memory_count(self, store):
        try:
            return sum(store.stats().values())
        except Exception:
            return 0

    def step_once(self):
        try:
            for _ in range(self.n):
                r = self.subject.step()
                self.actions[r.action] += 1
                self.last.append(
                    f"step {r.step:5d} | {r.action:<11} goal={r.goal:<10} cue={r.cue:<6} "
                    f"reward={r.reward:+.3f} err={r.prediction_error:.3f} "
                    f"cur={r.affect.get('curiosity', 0):.2f} fru={r.affect.get('frustration', 0):.2f}")
        except Exception as e:
            self.last.append(f"ERROR: {e!r}")

        top = ", ".join(f"{a}×{c}" for a, c in self.actions.most_common(4))
        body = "\n".join(self.last)
        self.text.delete("1.0", "end")
        self.text.insert("end", f"实时行动（累计 {sum(self.actions.values())} 步）\n")
        self.text.insert("end", f"动作分布: {top}\n")
        self.text.insert("end", "-" * 76 + "\n")
        self.text.insert("end", body)
        self.root.after(400, self.step_once)


if __name__ == "__main__":
    root = tk.Tk()
    Live(root)
    root.mainloop()
