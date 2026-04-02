import time
from datetime import datetime
from typing import Dict


def trace_entry(agent: str, t0: float, inp: str, out: str,
                status: str = "success") -> Dict:
    return {
        "agent":          agent,
        "status":         status,
        "duration_ms":    round((time.time() - t0) * 1000, 1),
        "input_summary":  (inp or "")[:80],
        "output_summary": (out or "")[:80],
        "timestamp":      datetime.now().isoformat(),
    }