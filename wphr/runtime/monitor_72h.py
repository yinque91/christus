"""Long-run monitor: samples health, writes JSONL, detects growth/corruption."""
import json, time, sys
from pathlib import Path
WS = Path(r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR_workspace")
OUT = WS / "logs" / "monitor72.jsonl"
sys.path.insert(0, r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR")
from wphr.runtime.service_manager import ServiceManager
sm = ServiceManager(WS / "service")
start = time.time()
OUT.parent.mkdir(parents=True, exist_ok=True)
while True:
    st = sm.read_status()
    rec = {"ts": time.time(), "uptime_s": round(time.time() - start, 1),
           "pid": st.get("pid"), "rss_mb": st.get("rss_mb"),
           "disk_free_gb": (st.get("disk") or {}).get("free_gb"),
           "step": st.get("step")}
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    time.sleep(60)
