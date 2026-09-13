"""
M6 backend — serves the M1-M5 pipeline outputs as one combined
'investigation case' per slick_id, for the dashboard frontend.

Run from anywhere with:
    OILSPILL_ROOT=/path/to/OilSpill_SIH uvicorn main:app --reload

If OILSPILL_ROOT isn't set, it assumes this file lives at
<repo_root>/dashboard/backend/main.py and walks up two levels.
"""

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(os.environ.get("OILSPILL_ROOT", Path(__file__).resolve().parents[2]))
OUTPUTS = REPO_ROOT / "outputs"

SPILL_FILE = OUTPUTS / "spill" / "all_detections.json"
GEOSPATIAL_FILE = OUTPUTS / "geospatial" / "spill_metadata.json"
DRIFT_FILE = OUTPUTS / "drift" / "drift_simulation_result.json"
AIS_FILE = OUTPUTS / "AIS" / "m4_evidence.json"
SCORING_FILE = OUTPUTS / "scoring" / "ranked_vessels.json"


def _load(path: Path) -> Any:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_list(data: Any) -> list:
    return data if isinstance(data, list) else [data]


def _clean_nan(value: Any) -> Any:
    """Recursively replace NaN/Infinity floats with None (raw AIS track data
    has literal NaN for the first point's prev_lat/prev_lon, which strict
    JSON — and FastAPI's encoder — rejects)."""
    if isinstance(value, float):
        return None if value != value or value in (float("inf"), float("-inf")) else value
    if isinstance(value, dict):
        return {k: _clean_nan(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_nan(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Maritime Pollution Forensics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "repo_root": str(REPO_ROOT),
        "found": {
            "spill": SPILL_FILE.exists(),
            "geospatial": GEOSPATIAL_FILE.exists(),
            "drift": DRIFT_FILE.exists(),
            "ais": AIS_FILE.exists(),
            "scoring": SCORING_FILE.exists(),
        },
    }


@app.get("/api/cases")
def list_cases():
    """One entry per slick_id that has ranked vessels (the fully-connected chain)."""
    scoring = _as_list(_load(SCORING_FILE))
    drift = {d["simulation"]["slick_id"]: d["simulation"] for d in _as_list(_load(DRIFT_FILE))}

    seen: dict[str, dict] = {}
    for row in scoring:
        sid = row.get("slick_id")
        if sid is None:
            continue
        entry = seen.setdefault(sid, {"slick_id": sid, "vessel_count": 0, "top_score": 0})
        entry["vessel_count"] += 1
        entry["top_score"] = max(entry["top_score"], row.get("overall_score", 0))
        sim = drift.get(sid)
        if sim:
            entry["observation_time"] = sim.get("observation_time")
            entry["origin"] = sim.get("origin_estimation", {}).get("centroid")

    return sorted(seen.values(), key=lambda e: -e["top_score"])


@app.get("/api/case/{slick_id}")
def get_case(slick_id: str):
    scoring = [r for r in _as_list(_load(SCORING_FILE)) if r.get("slick_id") == slick_id]
    if not scoring:
        raise HTTPException(status_code=404, detail=f"No ranked vessels for slick_id '{slick_id}'")

    scoring.sort(key=lambda r: r.get("rank", 999))

    drift_all = _as_list(_load(DRIFT_FILE))
    drift_entry = next(
        (d for d in drift_all if d.get("simulation", {}).get("slick_id") == slick_id), None
    )

    ais_all = _as_list(_load(AIS_FILE))
    ais_by_mmsi = {
        str(a.get("mmsi")): a for a in ais_all if a.get("slick_id") == slick_id
    }

    # geospatial polygon may or may not exist for this id (known pipeline gap
    # between M2's current run and M3-M5's run) — degrade gracefully.
    geospatial_all = _as_list(_load(GEOSPATIAL_FILE))
    spill_entry = next(
        (g for g in geospatial_all if g.get("image_id") == slick_id), None
    )

    vessels = []
    for row in scoring:
        mmsi = str(row.get("vessel_id"))
        ais = ais_by_mmsi.get(mmsi)
        vessels.append(
            {
                "rank": row.get("rank"),
                "vessel_id": row.get("vessel_id"),
                "vessel_name": row.get("vessel_name"),
                "overall_score": row.get("overall_score"),
                "confidence": row.get("confidence"),
                "evidence": row.get("evidence"),
                "explanation": row.get("explanation"),
                "track": (ais or {}).get("trajectory", {}).get("track", []),
                "gaps": (ais or {}).get("gaps", []),
                "vessel_type": (ais or {}).get("vessel_type"),
            }
        )

    origin = None
    hindcast_path = None
    forecast_path = None
    observation_time = None
    if drift_entry:
        sim = drift_entry["simulation"]
        observation_time = sim.get("observation_time")
        origin = sim.get("origin_estimation")
        hindcast_path = sim.get("hindcast_trajectories", {}).get("centroid_path")
        forecast_path = sim.get("forecast_trajectories", {}).get("centroid_path")

    return _clean_nan(
        {
            "slick_id": slick_id,
            "observation_time": observation_time,
            "spill": {
                "geometry": (spill_entry or {}).get("geometry"),
                "area_km2": (spill_entry or {}).get("area_km2"),
                "confidence": (spill_entry or {}).get("confidence"),
            },
            "origin": origin,
            "hindcast_path": hindcast_path,
            "forecast_path": forecast_path,
            "vessels": vessels,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)