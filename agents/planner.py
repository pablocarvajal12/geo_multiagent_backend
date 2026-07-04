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

## Event-based flood queries
- If the query names a specific storm/flood event, use your own knowledge of that event's
  real date to set date_range, preferring the post-event window (event_date to event_date + 7-10
  days) over a whole month, so the imagery is close to the event itself.
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