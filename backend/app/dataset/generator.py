from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from backend.app.config import DATA_DIR

ASSET_TYPES = ["WORKSTATION", "SERVER", "DATABASE", "CLOUD_SERVICE"]
ASSET_CRITICALITY = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
DEFAULT_DATASET_SIZE = 20_000


def _source_ip(rng: np.random.Generator) -> str:
    octets = [str(rng.integers(1, 255)) for _ in range(4)]
    return ".".join(octets)


def _risk_from_event(event: Dict[str, Any]) -> float:
    crit_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}[event["asset_criticality"]]
    source_risk = 0.0 if event["source_ip"].startswith("10.") or event["source_ip"].startswith("192.168.") else 0.4
    return (
        event["failed_logins"] / 20
        + event["threat_intel_score"]
        + event["previous_alerts"] / 10
        + source_risk
        + (0.2 * crit_rank)
    ) / 4.2


def build_rule_decisions(event: Dict[str, Any]) -> Dict[str, str]:
    failed = event["failed_logins"]
    threat = event["threat_intel_score"]
    geo = int(event["geo_anomaly"])
    prev = event["previous_alerts"]
    crit_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}[event["asset_criticality"]]
    d1 = "BLOCK" if (failed > 9 or (geo and event["login_hour"] in {0, 1, 2, 3, 4, 22, 23}) or (failed > 5 and not event["source_ip"].startswith("10."))) else "ALLOW"
    if failed > 12 or threat > 0.8 or prev > 7:
        d2 = "CRITICAL"
    elif failed > 8 or threat > 0.6 or prev > 5:
        d2 = "HIGH"
    elif failed > 4 or threat > 0.35 or prev > 2:
        d2 = "MEDIUM"
    else:
        d2 = "LOW"
    danger = failed > 6 or threat > 0.5
    d3 = "PROTECT" if (crit_rank >= 2 or threat > 0.7 or prev > 4) and (crit_rank >= 1 or danger) else "NORMAL"
    d4 = "ESCALATE" if (crit_rank >= 2 and d2 in {"HIGH", "CRITICAL"}) or (threat > 0.68 and geo and d2 in {"HIGH", "CRITICAL"}) else "MONITOR"
    if d2 in {"HIGH", "CRITICAL"} and crit_rank >= 2 and (geo or failed > 9):
        d5 = "ISOLATE"
    elif d2 in {"HIGH", "CRITICAL"} and failed > 7:
        d5 = "MONITOR"
    elif d2 == "MEDIUM":
        d5 = "MONITOR"
    else:
        d5 = "ALLOW"
    return {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5}


def generate_dataset(
    size: int = DEFAULT_DATASET_SIZE,
    seed: int = 42,
    include_demo_event: bool = True,
    include_labels: bool = True,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    events = []
    for index in range(size):
        generated = {
            "event_id": f"SEC-{index + 1:06d}",
            "timestamp": (datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)).isoformat(),
            "user_id": f"USR-{rng.integers(1000, 9999)}",
            "source_ip": _source_ip(rng),
            "failed_logins": int(np.clip(rng.normal(7.5, 4.5), 0, 20)),
            "login_hour": int(rng.integers(0, 24)),
            "geo_anomaly": bool(rng.random() < 0.18),
            "asset_id": f"ASSET-{rng.integers(1, 500):04d}",
            "asset_type": rng.choice(ASSET_TYPES, p=[0.42, 0.3, 0.18, 0.10]).item(),
            "asset_criticality": rng.choice(ASSET_CRITICALITY, p=[0.48, 0.30, 0.15, 0.07]).item(),
            "threat_intel_score": float(np.clip(rng.normal(0.45, 0.28), 0.0, 1.0)),
            "previous_alerts": int(np.clip(rng.normal(3.0, 2.8), 0, 10)),
        }
        if include_labels:
            generated.update({"decision_outputs": build_rule_decisions(generated)})
        events.append(generated)

    if include_demo_event and size >= 123:
        demo_event = {
            "event_id": "SEC-000123",
            "timestamp": "2024-01-02T14:28:00+00:00",
            "user_id": "USR-4207",
            "source_ip": "10.0.0.21",
            "failed_logins": 11,
            "login_hour": 2,
            "geo_anomaly": False,
            "asset_id": "ASSET-0213",
            "asset_type": "SERVER",
            "asset_criticality": "LOW",
            "threat_intel_score": 0.69,
            "previous_alerts": 2,
        }
        if include_labels:
            demo_event["decision_outputs"] = build_rule_decisions(demo_event)
        for i, item in enumerate(events):
            if item["event_id"] == "SEC-000123":
                events[i] = demo_event
                break
        else:
            if len(events) > 122:
                events[122] = demo_event
            else:
                events.append(demo_event)
    return events


def save_dataset(events: List[Dict[str, Any]], folder: str | Path = DATA_DIR) -> Dict[str, Path]:
    base = Path(folder)
    base.mkdir(parents=True, exist_ok=True)
    clean_path = base / "clean_events.json"
    corrupted_path = base / "corrupted_events.json"
    corrections_path = base / "corrections.json"
    clean_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
    corrupted_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
    corrections_path.write_text("[]", encoding="utf-8")
    return {"clean": clean_path, "corrupted": corrupted_path, "corrections": corrections_path}


def bootstrap_datasets() -> Dict[str, List[Dict[str, Any]]]:
    events = generate_dataset(size=DEFAULT_DATASET_SIZE, seed=42)
    save_dataset(events, DATA_DIR / "generated")
    return {"clean": events, "corrupted": events}
