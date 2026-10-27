"""Hash-commit forecasts before the window opens so results are publicly verifiable."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def canonical_json(obj: dict) -> bytes:
    """Serialise a dict deterministically so the same forecast always hashes the same."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(obj: dict) -> str:
    """SHA-256 hex digest of the canonical JSON form of a dict."""
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def commit_forecast(forecast: dict, out_dir: str | Path = "commitments") -> tuple[Path, str]:
    """Write a timestamped forecast record and return its path and hash."""
    record = {
        "committed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "forecast": forecast,
    }
    digest = sha256(record)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = record["committed_utc"].replace(":", "").replace("-", "")
    path = out_dir / f"{stamp}_{digest[:12]}.json"
    path.write_text(json.dumps({"sha256": digest, "record": record}, indent=2, sort_keys=True))
    return path, digest


def verify(path: str | Path) -> bool:
    """Check that a committed forecast file has not been altered since it was hashed."""
    data = json.loads(Path(path).read_text())
    return sha256(data["record"]) == data["sha256"]
