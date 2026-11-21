"""Christus native Windows desktop application (Tkinter, no browser)."""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "CHRISTUS-DATA-01"))
from christus_platform.runtime import PlatformRuntime  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
STATE = APP_DIR / "state"
LOGS = APP_DIR / "logs"
STATE.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

BG = "#080d10"
PANEL = "#0f161b"
PANEL2 = "#131d22"
LINE = "#223038"
TEXT = "#e7efee"
MUTED = "#819295"
TEAL = "#2dd4bf"
USER_BG = "#172329"


class DesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Christus — 本地主体")
        self.root.geometry("1280x820")
        self.root.minsize(980, 650)
        self.root.configure(bg=BG)
        self.runtime = PlatformRuntime()
        self.session_id = None
        self.busy = False
        self.queue: queue.Queue = queue.Queue()
        self.stop_requested = False
        self._sessions = []
        self._build()
        self._load_sessions()
        self._refresh_state()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(300, self._ensure_welcome)
        self._poll_queue()

    def _ensure_welcome(self):
        try:
            if not self.chat.get("1.0", "end").strip():
                self.chat.config(state="normal")
                self.chat.insert("end", "Christus\n", "assistant")
                self.chat.insert("end", "消息会经过现有 Subject、SM-CTRN、记忆、情绪和概念模块。"
                                       "外部知识会标注来源，不会写成亲身经历。\n\n", "meta")
                self.chat.config(state="disabled")
                self.chat.see("1.0")
        except Exception:
            pass

    def _build(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", background=PANEL2, foreground=TEXT,
                        bordercolor=LINE, focusthickness=0, padding=7)
        style.map("TButton", background=[("active", "#1a2c32")])

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        side = tk.Frame(self.root, bg=PANEL, width=230)
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_propagate(False)

        brand = tk.Frame(side, bg=PANEL)
        brand.pack(fill="x", padx=16, pady=(18, 12))
        tk.Label(brand, text="C", bg="#12302d", fg="#8ef3e6", width=3,
                 font=("Segoe UI", 15, "bold")).pack(side="left")
        b2 = tk.Frame(brand, bg=PANEL)
        b2.pack(side="left", padx=9)
        tk.Label(b2, text="Christus", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(b2, text="本地持续主体", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w")

        self.status_var = tk.StringVar(value="正在连接主体…")
        st = tk.Frame(side, bg="#0b1216", highlightbackground=LINE,
                      highlightthickness=1)
        st.pack(fill="x", padx=12, pady=(0, 10))
        self.dot = tk.Label(st, text="●", bg="#0b1216", fg="#f59e0b")
        self.dot.pack(side="left", padx=(8, 4), pady=6)
        tk.Label(st, textvariable=self.status_var, bg="#0b1216", fg=MUTED,
                 font=("Segoe UI", 8)).pack(side="left")

        tk.Button(side, text="＋  新对话", bg="#102221", fg="#bff8ef",
                  activebackground="#15302e", activeforeground="#ffffff",
                  bd=0, relief="flat", anchor="w", padx=14, pady=9,
                  command=self.new_chat).pack(fill="x", padx=12, pady=(0, 12))

        self.nav = {}
        for key, label in (("chat", "◌   交流"), ("state", "◈   主体状态"),
                           ("history", "⌁   历史与记忆"), ("learning", "▤   学习中心"),
                           ("system", "⚙   系统管理")):
            b = tk.Button(side, text=label, bg=PANEL, fg="#9aabae",
                          activebackground="#142126", activeforeground="#eafffb",
                          bd=0, relief="flat", anchor="w", padx=14, pady=8,
                          command=lambda k=key: self.show(k))
            b.pack(fill="x", padx=8)
            self.nav[key] = b

        tk.Label(side, text="会话", bg=PANEL, fg="#5f7073",
                 font=("Segoe UI", 8)).pack(anchor="w", padx=18, pady=(16, 4))
        self.session_list = tk.Listbox(side, bg=PANEL, fg="#93a5a8",
                                       selectbackground="#102026",
                                       selectforeground="#d8f7f1",
                                       bd=0, highlightthickness=0,
                                       font=("Segoe UI", 9), activestyle="none")
        self.session_list.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.session_list.bind("<<ListboxSelect>>", self._pick_session)
        tk.Label(side, text="本地运行 · 无浏览器 · 一个 Subject", bg=PANEL,
                 fg="#587074", font=("Segoe UI", 7)).pack(side="bottom",
                                                          fill="x", padx=12, pady=10)

        main = tk.Frame(self.root, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)
        header = tk.Frame(main, bg="#0d1418", height=66,
                          highlightbackground=LINE, highlightthickness=1)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        self.title = tk.Label(header, text="交流", bg="#0d1418", fg=TEXT,
                              font=("Segoe UI", 15, "bold"))
        self.title.pack(side="left", padx=22, pady=(12, 0))
        self.subtitle = tk.Label(header, text="和同一个持续主体对话",
                                 bg="#0d1418", fg=MUTED, font=("Segoe UI", 8))
        self.subtitle.pack(side="left", padx=(10, 0), pady=(18, 0))
        self.subject_pill = tk.Label(header, text="subject —", bg="#0c1418",
                                     fg="#8da3a4", font=("Consolas", 8),
                                     padx=9, pady=5)
        self.subject_pill.pack(side="right", padx=18)

        self.pages = tk.Frame(main, bg=BG)
        self.pages.grid(row=1, column=0, sticky="nsew")
        self.pages.grid_rowconfigure(0, weight=1)
        self.pages.grid_columnconfigure(0, weight=1)
        self.views = {}
        self._chat_view()
        self._state_view()
        self._history_view()
        self._learning_view()
        self._system_view()
        self.show("chat")

    def _chat_view(self):
        f = tk.Frame(self.pages, bg=BG)
        self.views["chat"] = f
        f.grid_rowconfigure(0, weight=1)
        f.grid_columnconfigure(0, weight=1)
        wrap = tk.Frame(f, bg=BG)
        wrap.grid(row=0, column=0, sticky="nsew")
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)
        self.chat = tk.Text(wrap, bg=BG, fg=TEXT, bd=0, highlightthickness=0,
                            wrap="word", padx=34, pady=20,
                            font=("Segoe UI", 10), state="disabled",
                            spacing1=4, spacing3=6)
        self.chat.grid(row=0, column=0, sticky="nsew")
        sb = tk.Scrollbar(wrap, command=self.chat.yview, bg=PANEL2)
        sb.grid(row=0, column=1, sticky="ns")
        self.chat.config(yscrollcommand=sb.set)
        self.chat.tag_config("user", foreground="#dff8f4", background=USER_BG,
                             lmargin1=60, lmargin2=60, rmargin=12,
                             spacing1=8, spacing3=8)
        self.chat.tag_config("assistant", foreground="#d5e6e3",
                             lmargin1=12, lmargin2=12, rmargin=100,
                             spacing1=8, spacing3=8)
        self.chat.tag_config("meta", foreground="#5e777a", font=("Segoe UI", 7))
        self.chat.tag_config("source", foreground="#68c8ba", font=("Segoe UI", 7))
        self.chat.config(state="normal")
        self.chat.insert("end", "Christus\n", "assistant")
        self.chat.insert("end", "消息会经过现有 Subject、SM-CTRN、记忆、情绪和概念模块。"
                               "外部知识会标注来源，不会写成亲身经历。\n\n", "meta")
        self.chat.config(state="disabled")
        self.chat.see("1.0")

        composer = tk.Frame(f, bg="#0a1114", highlightbackground=LINE,
                            highlightthickness=1)
        composer.grid(row=1, column=0, sticky="ew", padx=22, pady=(6, 14))
        composer.grid_columnconfigure(0, weight=1)
        self.input = tk.Text(composer, height=3, bg="#0d171b", fg=TEXT,
                             insertbackground=TEAL, bd=0, highlightthickness=0,
                             wrap="word", font=("Segoe UI", 10), padx=10, pady=8)
        self.input.grid(row=0, column=0, sticky="ew", padx=(8, 6), pady=8)
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Shift-Return>", lambda e: None)
        btns = tk.Frame(composer, bg="#0a1114")
        btns.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        self.send_btn = tk.Button(btns, text="发送 ↑", bg="#0e3d38", fg="#8ef3e6",
                                  activebackground="#15544c", bd=0, padx=12, pady=7,
                                  command=self.send)
        self.send_btn.pack(fill="x")
        tk.Button(btns, text="中断", bg="#151f23", fg="#9eb4b5", bd=0,
                  padx=12, pady=4, command=self.request_stop).pack(fill="x", pady=(4, 0))
        tk.Label(composer, text="Enter 发送 · Shift+Enter 换行 · 互联网查询自动识别",
                 bg="#0a1114", fg="#4f6669", font=("Segoe UI", 7)).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 6))

    def _state_view(self):
        f = tk.Frame(self.pages, bg=BG)
        self.views["state"] = f
        f.grid_columnconfigure(0, weight=1)
        f.grid_columnconfigure(1, weight=1)
        self.state_labels = {}
        cards = [
            ("subject", "SUBJECT", ["subject_id", "step", "turns", "action", "reward"]),
            ("core", "COGNITIVE CORE", ["h_norm", "core_weights_unchanged", "language_source"]),
            ("memory", "MEMORY", ["semantic_contexts", "semantic_rules", "working_items",
                                  "records_written"]),
            ("concept", "CONCEPT / LANGUAGE", ["concept_nodes", "interventions",
                                               "revisions", "words"]),
            ("self", "SELF MODEL / TRANSFORMER", ["available", "mode", "source_probe",
                                                   "temporal", "transformer_loss",
                                                   "transformer_docs"]),
            ("session", "SESSION / CHECKPOINT", ["paused", "last_checkpoint"]),
        ]
        for i, (key, title, fields) in enumerate(cards):
            card = tk.Frame(f, bg=PANEL, highlightbackground=LINE,
                            highlightthickness=1)
            card.grid(row=i // 2, column=i % 2, sticky="nsew",
                      padx=(18 if i % 2 == 0 else 8, 18 if i % 2 else 8),
                      pady=(16 if i < 2 else 8, 8))
            tk.Label(card, text=title, bg=PANEL, fg="#6e8b8c",
                     font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(12, 4))
            for field in fields:
                row = tk.Frame(card, bg=PANEL)
                row.pack(fill="x", padx=14, pady=2)
                tk.Label(row, text=field, bg=PANEL, fg="#73898b",
                         font=("Consolas", 8), width=22, anchor="w").pack(side="left")
                val = tk.Label(row, text="—", bg=PANEL, fg=TEXT,
                               font=("Consolas", 8), anchor="w")
                val.pack(side="left", fill="x", expand=True)
                self.state_labels[field] = val
        card = tk.Frame(f, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        card.grid(row=3, column=0, columnspan=2, sticky="ew", padx=18, pady=8)
        tk.Label(card, text="AFFECT", bg=PANEL, fg="#6e8b8c",
                 font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(10, 2))
        self.affect_box = tk.Frame(card, bg=PANEL)
        self.affect_box.pack(fill="x", padx=14, pady=(0, 10))

    def _history_view(self):
        f = tk.Frame(self.pages, bg=BG)
        self.views["history"] = f
        f.grid_rowconfigure(0, weight=1)
        f.grid_columnconfigure(0, weight=1)
        self.history_text = tk.Text(f, bg=PANEL, fg=TEXT, bd=0,
                                    highlightbackground=LINE, highlightthickness=1,
                                    wrap="word", padx=18, pady=14,
                                    font=("Segoe UI", 9), state="disabled")
        self.history_text.grid(row=0, column=0, sticky="nsew", padx=18, pady=16)
        self.history_text.tag_config("h", foreground=TEAL,
                                     font=("Segoe UI", 10, "bold"))
        self.history_text.tag_config("m", foreground=MUTED, font=("Segoe UI", 8))

    def _learning_view(self):
        f = tk.Frame(self.pages, bg=BG)
        self.views["learning"] = f
        f.grid_rowconfigure(1, weight=1)
        f.grid_columnconfigure(0, weight=1)
        bar = tk.Frame(f, bg=BG)
        bar.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 6))
        tk.Label(bar, text="学习事件", bg=BG, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(side="left")
        self.pause_btn = tk.Button(bar, text="暂停学习", bg=PANEL2, fg=TEXT,
                                   bd=0, padx=12, pady=5, command=self.toggle_pause)
        self.pause_btn.pack(side="right")
        tk.Button(bar, text="训练本地 Transformer", bg=PANEL2, fg=TEAL,
                  bd=0, padx=12, pady=5, command=self.train_transformer).pack(
            side="right", padx=(0, 8))
        self.learning_text = tk.Text(f, bg=PANEL, fg=TEXT, bd=0,
                                     highlightbackground=LINE, highlightthickness=1,
                                     wrap="word", padx=18, pady=14,
                                     font=("Consolas", 8), state="disabled")
        self.learning_text.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 16))

    def _system_view(self):
        f = tk.Frame(self.pages, bg=BG)
        self.views["system"] = f
        f.grid_columnconfigure(0, weight=1)
        f.grid_columnconfigure(1, weight=1)
        card = tk.Frame(f, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        card.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=16)
        tk.Label(card, text="LOCAL RUNTIME", bg=PANEL, fg="#6e8b8c",
                 font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(12, 6))
        self.settings_text = tk.Text(card, height=9, bg=PANEL, fg=TEXT, bd=0,
                                     highlightthickness=0, font=("Consolas", 8),
                                     state="disabled", padx=10)
        self.settings_text.pack(fill="x", padx=8, pady=(0, 8))
        actions = tk.Frame(f, bg=BG)
        actions.grid(row=1, column=0, columnspan=2, sticky="w", padx=18)
        for text, cmd in (("保存检查点", self.checkpoint), ("备份数据", self.backup),
                          ("打开日志", self.open_logs), ("刷新", self.refresh_all)):
            tk.Button(actions, text=text, bg=PANEL2, fg=TEXT, bd=0,
                      padx=12, pady=6, command=cmd).pack(side="left", padx=(0, 8))
        self.system_state = tk.Text(f, bg=BG, fg=MUTED, bd=0, height=10,
                                    font=("Consolas", 8), state="disabled")
        self.system_state.grid(row=2, column=0, columnspan=2, sticky="ew",
                               padx=18, pady=12)

    def show(self, key):
        for v in self.views.values():
            v.grid_forget()
        self.views[key].grid(row=0, column=0, sticky="nsew")
        titles = {"chat": ("交流", "和同一个持续主体对话"),
                  "state": ("主体状态", "真实 Subject 运行值"),
                  "history": ("历史与记忆", "会话、诊断与来源"),
                  "learning": ("学习中心", "互动、知识与本地 Transformer"),
                  "system": ("系统管理", "停止、备份、检查点")}
        self.title.config(text=titles[key][0])
        self.subtitle.config(text=titles[key][1])
        for k, b in self.nav.items():
            b.config(bg="#142126" if k == key else PANEL,
                     fg="#eafffb" if k == key else "#9aabae")
        if key == "state":
            self._refresh_state()
        elif key == "history":
            self._render_history()
        elif key == "learning":
            self._refresh_learning()
        elif key == "system":
            self._refresh_system()

    def _append(self, text, tag):
        self.chat.config(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.see("end")
        self.chat.config(state="disabled")

    def _on_enter(self, event):
        if event.state & 0x0001:
            return None
        self.send()
        return "break"

    def new_chat(self):
        self.session_id = None
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.insert("end", "Christus\n", "assistant")
        self.chat.insert("end", "新的会话。主体会继续使用已有的记忆和检查点。\n\n", "meta")
        self.chat.config(state="disabled")

    def _pick_session(self, event):
        sel = self.session_list.curselection()
        if sel and sel[0] < len(self._sessions):
            self.open_session(self._sessions[sel[0]]["id"])

    def _load_sessions(self):
        try:
            self._sessions = self.runtime.sessions("local-user")
            self.session_list.delete(0, "end")
            for s in self._sessions:
                self.session_list.insert("end", (s.get("title") or "新的对话")[:34])
        except Exception:
            self._sessions = []

    def open_session(self, sid):
        self.session_id = sid
        msgs = self.runtime.history(sid)
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        for m in msgs:
            tag = "user" if m["role"] == "user" else "assistant"
            self.chat.insert("end", ("你: " if tag == "user" else "Christus: ") + m["text"] + "\n", tag)
            if m.get("citations"):
                self.chat.insert("end", " | ".join(c.get("title", "") for c in m["citations"]) + "\n", "source")
        self.chat.config(state="disabled")
        self.chat.see("end")

    def send(self):
        if self.busy:
            return
        text = self.input.get("1.0", "end").strip()
        if not text:
            return
        self.input.delete("1.0", "end")
        self.busy = True
        self.stop_requested = False
        self.send_btn.config(state="disabled")
        self._append(f"\n你\n{text}\n", "user")
        self._append("Christus 正在通过主体处理…\n", "meta")

        def worker():
            try:
                out = self.runtime.chat(text, session_id=self.session_id,
                                        user_id="local-user")
                self.queue.put(("reply", out))
            except Exception as exc:
                self.queue.put(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))

        threading.Thread(target=worker, daemon=True).start()
        self._poll_queue()

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "reply":
                    self._render_reply(payload)
                elif kind == "error":
                    self._append("\n[错误] " + payload + "\n", "meta")
                    self.busy = False
                    self.send_btn.config(state="normal")
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _render_reply(self, out):
        self.session_id = out.get("session_id")
        reply = out.get("reply", "")
        self.chat.config(state="normal")
        self.chat.insert("end", "\nChristus\n", "assistant")
        self.chat.config(state="disabled")
        for i in range(0, len(reply), 24):
            self._append(reply[i:i + 24], "assistant")
            self.root.update_idletasks()
            time.sleep(0.006)
        cites = out.get("citations") or []
        meta = f"\n来源: {out.get('source', 'local')}"
        if cites:
            meta += " · " + " | ".join(f"{c.get('title')} <{c.get('url')}>" for c in cites[:2])
        self._append(meta + "\n", "meta")
        self.busy = False
        self.send_btn.config(state="normal")
        self._load_sessions()
        self._refresh_state()

    def request_stop(self):
        self.stop_requested = True
        self._append("\n[中断请求] 当前主体 turn 已开始；不会把未完成回答伪装成已完成。\n", "meta")

    def _refresh_state(self):
        try:
            s = self.runtime.public_state()
        except Exception:
            self.status_var.set("状态读取失败")
            return
        self.status_var.set("主体运行中")
        self.dot.config(fg=TEAL)
        self.subject_pill.config(text="subject " + (s.get("subject_id", "—") or "—")[:12])
        flat = {
            "subject_id": s.get("subject_id"), "step": s.get("step"),
            "turns": s.get("turns"), "action": s.get("action"),
            "reward": s.get("reward"), "h_norm": round(float(s.get("h_norm", 0)), 4),
            "core_weights_unchanged": s.get("core_weights_unchanged"),
            "language_source": s.get("language_source"),
        }
        flat.update(s.get("memory", {}))
        flat.update(s.get("concept", {}))
        sm = s.get("self_model", {})
        tr = s.get("transformer", {})
        flat.update({"available": sm.get("available"), "mode": sm.get("mode"),
                     "source_probe": sm.get("source_probe"), "temporal": sm.get("temporal"),
                     "transformer_loss": tr.get("final_loss"),
                     "transformer_docs": tr.get("documents")})
        flat.update({"paused": s.get("paused"), "last_checkpoint": bool(s.get("last_checkpoint"))})
        for k, v in self.state_labels.items():
            if k in flat:
                v.config(text=str(flat[k]))
        for w in self.affect_box.winfo_children():
            w.destroy()
        for k, v in (s.get("affect") or {}).items():
            row = tk.Frame(self.affect_box, bg=PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=k, bg=PANEL, fg="#8fa3a4", width=20,
                     anchor="w", font=("Consolas", 8)).pack(side="left")
            bar = tk.Frame(row, bg="#15262b", height=5, width=220)
            bar.pack(side="left", padx=6)
            bar.pack_propagate(False)
            fill = tk.Frame(bar, bg=TEAL, height=5, width=max(1, int(220 * float(v))))
            fill.place(x=0, y=0, relheight=1)
            tk.Label(row, text=f"{float(v):.3f}", bg=PANEL, fg=MUTED,
                     font=("Consolas", 8)).pack(side="left")

    def _render_history(self):
        try:
            sessions = self.runtime.sessions("local-user")
            self._sessions = sessions
            lines = ["自身历史\n\n"]
            for s in sessions:
                lines.append(f"[会话] {s.get('title')}\n  id={s.get('id')}  "
                             f"updated={time.strftime('%Y-%m-%d %H:%M', time.localtime(s.get('updated_at', 0)))}\n")
            lines.append("\n学习 / 纠正 / 网络事件\n\n")
            for ev in self.runtime.learning()[:80]:
                c = ev.get("content") or {}
                text = c.get("text") or c.get("query") or c.get("reply") or ""
                lines.append(f"[{ev.get('kind')}] {text[:160]}\n")
            self.history_text.config(state="normal")
            self.history_text.delete("1.0", "end")
            for line in lines:
                self.history_text.insert("end", line, "h" if line.startswith("[") else "m")
            self.history_text.config(state="disabled")
        except Exception as exc:
            self.history_text.config(state="normal")
            self.history_text.delete("1.0", "end")
            self.history_text.insert("end", f"读取失败: {exc}")
            self.history_text.config(state="disabled")

    def _refresh_learning(self):
        try:
            self.learning_text.config(state="normal")
            self.learning_text.delete("1.0", "end")
            for ev in self.runtime.learning()[:120]:
                self.learning_text.insert("end",
                    f"[{ev.get('kind')}] {json.dumps(ev.get('content') or {}, ensure_ascii=False)[:500]}\n")
            tr = self.runtime.semantic.status()
            self.learning_text.insert("end", f"\n[local-transformer] {json.dumps(tr, ensure_ascii=False)}\n")
            self.learning_text.config(state="disabled")
            self.pause_btn.config(text="恢复学习" if self.runtime.paused else "暂停学习")
        except Exception as exc:
            self.learning_text.config(state="normal")
            self.learning_text.insert("end", str(exc))
            self.learning_text.config(state="disabled")

    def train_transformer(self):
        try:
            import subprocess
            py = sys.executable
            script = PROJECT / "christus_desktop" / "training" / "train_local_transformer.py"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(PROJECT / "CHRISTUS-DATA-01") + os.pathsep + \
                r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR\.venv\Lib\site-packages"
            subprocess.Popen([py, str(script)], cwd=str(script.parent), env=env,
                             stdout=open(LOGS / "train_transformer.out.log", "a", encoding="utf-8"),
                             stderr=open(LOGS / "train_transformer.err.log", "a", encoding="utf-8"))
            messagebox.showinfo("本地 Transformer", "训练已在后台启动；完成后刷新状态。")
        except Exception as exc:
            messagebox.showerror("训练失败", str(exc))

    def toggle_pause(self):
        self.runtime.pause(not self.runtime.paused)
        self._refresh_learning()
        self._refresh_state()

    def _refresh_system(self):
        s = self.runtime.public_state()
        tr = s.get("transformer", {})
        self.settings_text.config(state="normal")
        self.settings_text.delete("1.0", "end")
        self.settings_text.insert("end", json.dumps({
            "subject_id": s.get("subject_id"), "step": s.get("step"),
            "language": s.get("language_source"),
            "external_bridge": self.runtime.external_language.enabled,
            "internet": "Wikipedia / DuckDuckGo",
            "port_used": False, "desktop": "Tkinter",
            "checkpoint": s.get("last_checkpoint"),
            "local_transformer": tr,
        }, ensure_ascii=False, indent=2))
        self.settings_text.config(state="disabled")
        self.system_state.config(state="normal")
        self.system_state.delete("1.0", "end")
        self.system_state.insert("end",
            "数据目录: christus_platform/state\n检查点: platform checkpoint\n"
            "日志: christus_platform/logs + christus_desktop/logs\n"
            "本地绑定: 无 Web 服务器，桌面直接调用 PlatformRuntime\n"
            "Transformer: 仅用于公开知识语义重排，不决定身份、记忆或情绪\n")
        self.system_state.config(state="disabled")

    def checkpoint(self):
        try:
            ck = self.runtime.checkpoint()
            messagebox.showinfo("检查点", "已保存\n" + str(ck)[:500])
            self._refresh_state()
        except Exception as exc:
            messagebox.showerror("检查点", str(exc))

    def backup(self):
        try:
            import shutil
            dest = PROJECT / "backups" / time.strftime("desktop-%Y%m%d-%H%M%S")
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("platform.db", "runtime_state.json"):
                src = PROJECT / "christus_platform" / "state" / name
                if src.exists():
                    shutil.copy2(src, dest / name)
            subj = PROJECT / "christus_platform" / "state" / "subject"
            if subj.exists():
                shutil.copytree(subj, dest / "subject", dirs_exist_ok=True)
            messagebox.showinfo("备份", "已备份到\n" + str(dest))
        except Exception as exc:
            messagebox.showerror("备份", str(exc))

    def open_logs(self):
        try:
            os.startfile(str(PROJECT / "christus_platform" / "logs"))
        except Exception as exc:
            messagebox.showerror("日志", str(exc))

    def refresh_all(self):
        self._load_sessions()
        self._refresh_state()
        self._refresh_system()

    def close(self):
        try:
            self.runtime.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    (STATE / "desktop.pid").write_text(str(os.getpid()), encoding="ascii")
    root = tk.Tk()
    app = DesktopApp(root)
    if "--smoke" in sys.argv:
        root.update()
        root.after(1200, app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
