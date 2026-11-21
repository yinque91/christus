"""Read-only observer window: shows what WPHR sees and its current numbers.

Does not touch the agent: only reads frames/latest.png and trainer.jsonl.
"""

from __future__ import annotations

import json
import time
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

WS = Path(r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR_workspace")
FRAME = WS / "frames" / "latest.png"
LOG = WS / "logs" / "trainer.jsonl"
STATE = WS / "state" / "trainer_state.json"


def last_events(n: int = 40) -> list[dict]:
    if not LOG.exists():
        return []
    lines = LOG.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-n:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


class Observer:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("WPHR 观察窗（只读）")
        root.geometry("980x820")
        root.attributes("-topmost", True)

        self.img_label = tk.Label(root, bg="black")
        self.img_label.pack(fill="both", expand=True)

        self.info = tk.Label(root, text="等待数据…", justify="left", anchor="w",
                             font=("Consolas", 11), bg="#111", fg="#0f0")
        self.info.pack(fill="x")
        self.photo = None
        self.tick()

    def tick(self):
        try:
            if FRAME.exists():
                im = Image.open(FRAME).convert("RGB")
                im.thumbnail((960, 700))
                self.photo = ImageTk.PhotoImage(im)
                self.img_label.config(image=self.photo)
        except Exception as e:
            self.img_label.config(text=f"frame error: {e}", fg="red")

        ev = last_events(40)
        with_err = [e for e in ev if e.get("err") is not None]
        last = ev[-1] if ev else {}
        st = {}
        if STATE.exists():
            try:
                st = json.loads(STATE.read_text(encoding="utf-8"))
            except Exception:
                pass
        lines = [
            f"step      : {last.get('step')}",
            f"position  : {last.get('pos')}",
            f"predicted : {last.get('pred')}",
            f"error     : {last.get('err')} px",
            f"motion    : {last.get('motion_px')} px",
            f"mean_err  : {st.get('mean_err')}",
            f"updated   : {time.strftime('%H:%M:%S')}",
        ]
        if with_err:
            errs = [e["err"] for e in with_err]
            lines.append(f"recent err: min={min(errs):.0f} max={max(errs):.0f} n={len(errs)}")
        self.info.config(text="\n".join(lines))
        self.root.after(1000, self.tick)


if __name__ == "__main__":
    root = tk.Tk()
    Observer(root)
    root.mainloop()
