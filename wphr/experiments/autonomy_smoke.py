#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AUTONOMY-01 冒烟：权限门控、文件沙箱、进程、网络、循环、审计、摄像头枚举。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR")

from wphr.tools.computer import ComputerTool
from wphr.tools.network import NetworkTool
from wphr.tools.policy import ApprovalRequired, Audit, PermissionDenied, Policy
from wphr.runtime.autonomous_loop import AutonomousLoop

OUT = Path(r"C:\Users\13394\Documents\deepseek-harness\default-workspace\.tmp\autonomy")
OUT.mkdir(parents=True, exist_ok=True)
res = {}

pol = Policy()
ct = ComputerTool(pol, Audit(OUT / "audit.jsonl"))
net = NetworkTool(pol, Audit(OUT / "audit.jsonl"))
res["workspace"] = str(pol.workspace)

# 1) fs sandbox
p = ct.write("notes/smoke.txt", "hello")
res["fs_write_in_workspace"] = p.exists()
try:
    ct.write(str(Path(r"C:\Windows\Temp\wphr_escape.txt")), "x")
    res["fs_write_outside"] = "ALLOWED(bad)"
except (ApprovalRequired, PermissionDenied) as e:
    res["fs_write_outside"] = f"blocked({type(e).__name__})"

# 2) processes
res["python"] = ct.run_python("print(2+2)")["stdout"].strip()
res["powershell"] = ct.run_powershell("Write-Output ok")["stdout"].strip()

# 3) network
res["dns"] = net.dns("pypi.org")
g = net.get("https://example.com")
res["http_get"] = {"ok": g.get("ok"), "status": g.get("status"), "bytes": g.get("bytes")}
res["download"] = net.download("https://example.com", str(pol.workspace / "dl" / "example.html"))

# 4) denied action
try:
    pol.check("net.http_post")
    res["post_gate"] = "ALLOWED(bad)"
except ApprovalRequired:
    res["post_gate"] = "approval_required"
try:
    pol.check("credential.theft")
    res["denied_gate"] = "ALLOWED(bad)"
except PermissionDenied:
    res["denied_gate"] = "denied"

# 5) loop
def decide(state, obs):
    return {"note": "count", "done": state.step >= 3}

loop = AutonomousLoop(decide, max_steps=10, state_path=OUT / "loop_state.json")
recs = loop.run(lambda: {"tick": True})
loop.pause()
recs2 = loop.run(lambda: {"tick": True})
loop.load()
res["loop"] = {"rounds": len(recs), "paused_extra": len(recs2),
               "state_step": loop.state.step, "budget": loop.state.budget_left}

# 6) camera enumeration (real hardware check)
from wphr.sensors.camera import CameraAdapter
res["cameras"] = [c.index for c in CameraAdapter().enumerate()]

# 7) audit
audit_lines = (OUT / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
res["audit_records"] = len(audit_lines)
res["audit_has_sha"] = "sha256" in audit_lines[-1]

print(json.dumps(res, ensure_ascii=False, indent=1))
(OUT / "smoke.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
