"""Start Christus Autonomy from the repository without requiring pip install."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CHRISTUS_AUTONOMY_STATE", str(ROOT / ".christus_autonomy"))
from christus_autonomy.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
