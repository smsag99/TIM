"""Flag source IPs with a high number of refused/failed connections.

This is deliberately rule-based, not LLM-based: counting refused
events is a deterministic job, so we parse the log lines directly.
This is detection only — it flags, it does not block anything.
"""

import re
from collections import defaultdict

from app import config

# Matches lines like:
# 2026-05-12 09:03:02 WARN  conn  src=203.0.113.55 dst=10.0.1.5 port=22 status=REFUSED bytes=0
_LINE_RE = re.compile(
    r"src=(?P<src>\S+).*?status=(?P<status>\S+)"
)

_BAD_STATUSES = {"REFUSED", "TIMEOUT", "SLOW"}


def get_flagged_ips() -> list[dict]:
    """Return source IPs whose failed-event count meets the threshold."""
    refused = defaultdict(int)
    total = defaultdict(int)

    for path in sorted(config.LOGS_DIR.glob("*.log")):
        for line in path.read_text(encoding="utf-8").splitlines():
            m = _LINE_RE.search(line)
            if not m:
                continue
            src = m.group("src")
            total[src] += 1
            if m.group("status").upper() in _BAD_STATUSES:
                refused[src] += 1

    flagged = []
    for ip, bad_count in refused.items():
        if bad_count >= config.FLAG_THRESHOLD:
            flagged.append({
                "ip": ip,
                "failed_events": bad_count,
                "total_events": total[ip],
            })

    flagged.sort(key=lambda x: x["failed_events"], reverse=True)
    return flagged
