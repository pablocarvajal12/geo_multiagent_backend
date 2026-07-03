"""
agents/planner.py - Planner Agent

Receives the raw user query and produces a structured execution plan:
  - Geographic location (bounding box)
  - Date range
  - Analysis type
  - Required spectral indices
  - Recommended satellites
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage

from state import GeoAgentState

logger = logging.getLogger(__name__)

from langchain_groq import ChatGroq

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Prompt ────────────────────────────────────────────────────────────────────
PLANNER_SYSTEM = """
You are the Planner Agent of a geospatial multi-agent system.
Your role is to interpret a natural-language Earth Observation query and
return a STRICT JSON execution plan — no extra text, only the JSON object.

## Output schema (all fields required)

{{
  "location": {{
    "name": "Human-readable place name",
    "bbox": [min_lon, min_lat, max_lon, max_lat],   // WGS-84
    "centroid": [lon, lat]
  }},
  "date_range": {{
    "start": "YYYY-MM-DD",
    "end":   "YYYY-MM-DD"
  }},
  "analysis_type": "<one of: vegetation | water | urban | fire | snow | flood | soil | general>",
  "required_indices": ["NDVI", "..."],   // list of spectral indices to compute
  "satellites": ["Sentinel-2", "..."],   // preferred sources in priority order
  "cloud_cover_max": 20,                 // maximum acceptable cloud cover %
  "resolution_m": 10,                    // target spatial resolution in metres
  "summary": "One sentence describing what will be done"
}}

## Index catalogue
- Vegetation : NDVI, EVI, SAVI, LAI
- Water       : NDWI, MNDWI, AWEI
- Urban/built : NDBI, NBI, BUI
- Fire/burn   : NBR, BAIS2
- Snow/ice    : NDSI
- Soil        : BSI, GSAVI

## Satellite catalogue
- Sentinel-2  (10-60 m, 5-day revisit, free via Copernicus)
- Landsat-8/9 (30 m,  16-day revisit, free via USGS/NASA)
- MODIS       (250-1000 m, daily, free via NASA)

## Rules
1. If no date is specified, use the last 30 days from today ({today}).
2. Choose the satellite with the best trade-off of resolution and revisit for
   the analysis type.
3. Always include at least one index.
4. bbox must be reasonable (≤ 5° × 5° for Sentinel, ≤ 10° × 10° for MODIS).
5. Return ONLY the JSON — no markdown fences, no explanation.
6. Your response must be written in Spanish.

## CRITICAL BBOX RULES for flood/water analysis near coasts
- For FLOOD or WATER analysis of coastal cities, bbox MUST cover INLAND areas, NOT the sea.
- Keep bbox TIGHT (≤ 0.5° × 0.5°) to target the specific flood zone, not a whole region.
- General rule: for coastal flood queries, keep max_lon ≤ centroid_lon + 0.2°.

## DANA Valencia October 2024 — hardcoded parameters
- If the query mentions: DANA, Valencia floods, inundaciones Valencia, octubre 2024, L'Horta Sud, Paiporta, Catarroja, Alfafar, Massanassa, or similar:
  - location.name: USE THE NAME THE USER SAID (e.g. "Valencia" if they said "Valencia") — do NOT replace it with "L'Horta Sud"
  - bbox: [-0.55, 39.30, -0.25, 39.55]   ← targets the flood zone near Valencia
  - date_range: 2024-10-30 to 2024-11-10  ← post-DANA window with clearing skies
  - cloud_cover_max: 60
  - required_indices: ["NDWI", "MNDWI", "AWEI"]
  - analysis_type: "flood"

## Date range rules for known events
- If the query mentions the DANA Valencia 2024 or Valencia floods October 2024: use date_range 2024-10-30 to 2024-11-10.
- For event-based flood queries, prefer the post-event window (event_date to event_date + 7 days) over the full month.
"""


# ── Agent node ────────────────────────────────────────────────────────────────
class PlannerAgent:
    """LangGraph node: parse user query → structured plan."""

    def __init__(self, model: str | None = None):
        self.llm = ChatGroq(
            model=model or GROQ_MODEL,
            temperature=0.0,
            max_tokens=1024,
        )

    # ── public interface ──────────────────────────────────────────────────────
    def __call__(self, state: GeoAgentState) -> dict:
        logger.info("[Planner] Processing query: %s", state["user_query"])

        system_prompt = PLANNER_SYSTEM.format(
            today=datetime.utcnow().strftime("%Y-%m-%d")
        )

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["user_query"]),
            ]
        )

        raw = response.content.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        plan = json.loads(raw)

        # Validate minimal required fields
        _validate_plan(plan)

        # Override hardcoded para eventos conocidos (seguridad si el LLM ignora el prompt)
        _apply_known_event_overrides(plan, state["user_query"])

        # Post-process: para análisis de inundaciones/agua en ciudades costeras,
        # recortar el bbox para que no se extienda más de 0.3° al este del centroide
        _clip_bbox_inland(plan)

        logger.info("[Planner] Plan created: %s", plan.get("summary", ""))

        return {
            "plan": plan,
            "location": plan["location"],
            "date_range": plan["date_range"],
            "analysis_type": plan["analysis_type"],
            "required_indices": plan["required_indices"],
            "satellites": plan["satellites"],
            "current_agent": "data_acquisition",
            "messages": [
                {
                    "agent": "planner",
                    "type": "plan",
                    "content": plan.get("summary", "Plan created"),
                    "data": plan,
                }
            ],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────
DANA_KEYWORDS = {
    "dana", "paiporta", "catarroja", "alfafar", "massanassa", "benetússer",
    "l'horta", "horta sud", "inundacion", "inundación", "riada", "flood",
}

def _apply_known_event_overrides(plan: dict, query: str) -> None:
    """
    Fuerza parámetros correctos para eventos conocidos con independencia de lo
    que genere el LLM. Actualmente cubre: DANA Valencia octubre 2024.
    """
    q = query.lower()
    is_dana = any(kw in q for kw in DANA_KEYWORDS) and (
        "valencia" in q or "dana" in q or "2024" in q
    )
    if not is_dana:
        return

    DANA_BBOX      = [-0.55, 39.30, -0.25, 39.55]
    DANA_START     = "2024-10-30"
    DANA_END       = "2024-11-10"
    DANA_INDICES   = ["NDWI", "MNDWI", "AWEI"]
    DANA_CLOUD_MAX = 60

    loc = plan.setdefault("location", {})
    loc["bbox"] = DANA_BBOX
    if loc.get("centroid"):
        loc["centroid"] = [-0.40, 39.42]

    dr = plan.setdefault("date_range", {})
    dr["start"] = DANA_START
    dr["end"]   = DANA_END

    plan["analysis_type"]    = "flood"
    plan["cloud_cover_max"]  = DANA_CLOUD_MAX

    existing = {i.upper() for i in plan.get("required_indices", [])}
    for idx in DANA_INDICES:
        if idx not in existing:
            plan.setdefault("required_indices", []).append(idx)

    logger.info(
        "[Planner] DANA override aplicado: bbox=%s  dates=%s→%s  indices=%s",
        DANA_BBOX, DANA_START, DANA_END, plan["required_indices"],
    )


def _clip_bbox_inland(plan: dict) -> None:
    """
    Para análisis de inundaciones/agua, recorta el bbox para que no se extienda
    más de 0.3° al este del centroide, evitando tiles que cubren sólo mar.
    """
    WATER_TYPES = {"flood", "water"}
    analysis_type = plan.get("analysis_type", "")
    if analysis_type not in WATER_TYPES:
        return

    loc = plan.get("location", {})
    bbox = loc.get("bbox")
    centroid = loc.get("centroid")
    if not bbox or not centroid or len(bbox) != 4 or len(centroid) != 2:
        return

    centroid_lon = centroid[0]
    max_east = centroid_lon + 0.3  # no más de 0.3° al este del centro

    if bbox[2] > max_east:
        original = list(bbox)
        bbox[2] = round(max_east, 4)
        logger.info(
            "[Planner] bbox recortado para evitar mar: %s → %s (análisis=%s)",
            original, bbox, analysis_type,
        )
        loc["bbox"] = bbox


def _validate_plan(plan: dict) -> None:
    required = [
        "location", "date_range", "analysis_type",
        "required_indices", "satellites",
    ]
    missing = [k for k in required if k not in plan]
    if missing:
        raise ValueError(f"Planner returned incomplete plan. Missing: {missing}")

    loc = plan["location"]
    if "bbox" not in loc or len(loc["bbox"]) != 4:
        raise ValueError("location.bbox must be [min_lon, min_lat, max_lon, max_lat]")

    dr = plan["date_range"]
    if "start" not in dr or "end" not in dr:
        raise ValueError("date_range must have 'start' and 'end'")