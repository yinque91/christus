"""Native Tkinter window for inspecting and running MIRROR-01.

No browser, no HTTP server, no external model. The experiment itself runs in a
subprocess so the Subject is not duplicated while the viewer is open.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "mirror01" / "results.json"
FRAME = ROOT / "mirror01" / "frames" / "M3_seed12000_marker.png"
LOG = ROOT / "mirror01" / "lab.log"


class MirrorLab:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Christus — Mirror Lab")
        root.geometry("1120x760")
        root.minsize(900, 620)
        root.configure(bg="#0a1013")
        self.data = None
        self._build()
        self.load()

    def _build(self):
        head = tk.Frame(self.root, bg="#0d171b", height=70)
        head.pack(fill="x")
        head.pack_propagate(False)
        tk.Label(head, text="MIRROR LAB", bg="#0d171b", fg="#8ef3e6",
                 font=("Segoe UI", 18, "bold")).pack(side="left", padx=24, pady=(14, 0))
        tk.Label(head, text="MIRROR-01 · 真实主体镜像自我识别实验", bg="#0d171b",
                 fg="#789092", font=("Segoe UI", 9)).pack(side="left", padx=12, pady=(22, 0))
        self.status = tk.StringVar(value="ready")
        tk.Label(head, textvariable=self.status, bg="#0d171b", fg="#6f8f8f",
                 font=("Consolas", 9)).pack(side="right", padx=22)

        body = tk.Frame(self.root, bg="#0a1013")
        body.pack(fill="both", expand=True, padx=18, pady=14)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg="#0f181c")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.text = tk.Text(left, bg="#0f181c", fg="#d8e8e6", bd=0,
                            wrap="word", padx=18, pady=16,
                            font=("Consolas", 10))
        self.text.pack(fill="both", expand=True)
        self.text.tag_config("h", foreground="#7be0d1", font=("Segoe UI", 12, "bold"))
        self.text.tag_config("k", foreground="#e8f6f2")
        self.text.tag_config("m", foreground="#7f999b")

        right = tk.Frame(body, bg="#0f181c")
        right.grid(row=0, column=1, sticky="nsew")
        self.image_label = tk.Label(right, bg="#0f181c", text="M3 marker frame",
                                    fg="#789092")
        self.image_label.pack(fill="both", expand=True, padx=12, pady=(12, 4))
        self.photo = None

        bar = tk.Frame(self.root, bg="#0d171b", height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Button(bar, text="Run M1–M6 (20 seeds)", command=self.run,
                  bg="#0e3d38", fg="#8ef3e6", bd=0, padx=14, pady=7).pack(
            side="left", padx=18, pady=9)
        tk.Button(bar, text="Refresh", command=self.load,
                  bg="#151f23", fg="#b9cecc", bd=0, padx=14, pady=7).pack(
            side="left", pady=9)
        tk.Button(bar, text="Open frames", command=self.open_frames,
                  bg="#151f23", fg="#b9cecc", bd=0, padx=14, pady=7).pack(
            side="left", padx=8, pady=9)
        tk.Label(bar, text="native Tkinter · no browser · one subject per run",
                 bg="#0d171b", fg="#5c7376", font=("Segoe UI", 8)).pack(
            side="right", padx=18)

    def _add(self, s, tag="m"):
        self.text.insert("end", s, tag)

    def load(self):
        if not RESULTS.exists():
            self.text.delete("1.0", "end")
            self._add("No results yet. Run the experiment.", "h")
            return
        self.data = json.loads(RESULTS.read_text(encoding="utf-8"))
        self.text.delete("1.0", "end")
        self._add("MIRROR-01 overview\n", "h")
        self._add("20 independent seeds; conditions share the same task structure; "
                  "evaluator labels are not given to the subject.\n\n", "m")
        A = self.data.get("aggregate", {})
        for c, label in (("A0", "no mirror"), ("A1", "mirror only"),
                         ("A2", "mirror + body"), ("A3", "mirror + body + self"),
                         ("A4", "shuffled temporal"), ("A5", "no body feedback")):
            a = A.get(c, {})
            self._add(f"{c}  {label}\n", "k")
            self._add(f"  n={a.get('n')}  contact={a.get('marker_contact_rate')}  "
                      f"inspection={a.get('inspection_rate')}  sync={a.get('sync_mean')}\n", "m")
        self._add("\nM2 lag peaks (lower=realtime)\n", "h")
        for c in ("realtime", "delay_0.5", "delay_2", "other", "none"):
            rows = [x["conditions"][c] for x in self.data.get("M2", [])]
            lags = [r["lag_peak"] for r in rows if r.get("lag_peak") is not None]
            mean = sum(lags) / len(lags) if lags else None
            self._add(f"  {c:10s} lag_peak={mean}\n", "m")
        m6 = self.data.get("M6", {})
        self._add("\nM6 memory/checkpoint\n", "h")
        self._add(f"  memory_reference={m6.get('memory_reference')}\n"
                  f"  restored_step={m6.get('restored_step')} summaries={m6.get('restored_summaries')}\n", "m")
        if FRAME.exists():
            try:
                im = Image.open(FRAME)
                im.thumbnail((460, 520))
                self.photo = ImageTk.PhotoImage(im)
                self.image_label.config(image=self.photo, text="")
            except Exception:
                pass
        self.status.set("loaded")

    def run(self):
        if getattr(self, "_running", False):
            return
        self._running = True
        self.status.set("running 20 seeds…")
        LOG.parent.mkdir(exist_ok=True)

        def worker():
            try:
                env = dict(os.environ)
                env["PYTHONPATH"] = str(ROOT / "CHRISTUS-DATA-01") + os.pathsep + \
                    r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR\.venv\Lib\site-packages"
                with LOG.open("a", encoding="utf-8") as f:
                    p = subprocess.run([sys.executable, "-m", "christus_mirror.run"],
                                       cwd=str(ROOT), env=env, stdout=f,
                                       stderr=subprocess.STDOUT, timeout=1800)
                self.root.after(0, lambda: self._done(p.returncode))
            except Exception as exc:
                self.root.after(0, lambda: self._done(f"error: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, code):
        self._running = False
        self.status.set(f"finished: {code}")
        self.load()

    def open_frames(self):
        p = ROOT / "mirror01" / "frames"
        if p.exists():
            os.startfile(str(p))


def main():
    root = tk.Tk()
    app = MirrorLab(root)
    if "--smoke" in sys.argv:
        root.after(800, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
