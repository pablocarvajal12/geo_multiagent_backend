"""
agents/reporter.py - Report Synthesis Agent

Receives all analysis results and generates:
  1. A rich Markdown report in natural language
  2. An interactive Folium map (HTML)
  3. Summary statistics charts
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Optional

import os
from PIL import Image

import folium
from folium import plugins as folium_plugins
from langchain_core.messages import HumanMessage, SystemMessage

from state import GeoAgentState

logger = logging.getLogger(__name__)

from langchain_groq import ChatGroq

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "./outputs"))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  System prompt
# ─────────────────────────────────────────────────────────────────────────────
REPORTER_SYSTEM = """
You are the Report Synthesis Agent of a geospatial multi-agent system.
Your job is to translate raw analysis results into a clear, well-structured
report that a non-technical user can understand.

## Report structure
1. **Executive Summary** – 2-3 sentence overview of findings
2. **Study Area** – Location, date, satellite used
3. **Methodology** – Brief explanation of the analysis approach (no jargon)
4. **Results** – One section per computed index with:
   - Interpretation of values
   - Spatial patterns observed
   - Comparison to typical ranges
5. **Conclusions** – Key takeaways and recommendations
6. **Technical Notes** – Brief mention of data source, resolution, cloud cover

## Style rules
- Write for a general audience. Avoid technical jargon without explanation.
- Use concrete language: "70% of the area shows healthy vegetation" not "NDVI > 0.5".
- Include specific numbers from the statistics provided.
- Keep each section concise but informative.
- Use markdown formatting (headers, bold, bullet lists).

## NDVI interpretation guide
< 0.1  : Bare soil, rock, urban, water
0.1–0.2: Sparse vegetation, degraded land
0.2–0.4: Moderate vegetation (crops, grassland)
0.4–0.6: Dense vegetation, healthy crops
> 0.6  : Very dense vegetation (tropical forest)

## NDWI interpretation (standard: clear water)
< -0.1 : Dry land / dense vegetation — no flooding
-0.1–0 : Possibly saturated soil or turbid flood water (mud, sediment)
0–0.2  : Probable open water or very wet soil / flooded fields
> 0.2  : Confirmed open water

## NDWI interpretation for FLOOD events with TURBID water (e.g. flash floods, DANA storms)
IMPORTANT: Flash flood water carries mud and sediment, raising SWIR reflectance and lowering NDWI.
For these events use LOWER thresholds:
  NDWI > -0.1   → possible turbid flood water
  NDWI > 0.0    → likely flooded (turbid or mixed water/soil)
  NDWI > 0.15   → confirmed standing water
If analysis_type is "flood", use these lower thresholds instead of the standard ones.

## MNDWI interpretation (better for turbid/muddy water)
< -0.1 : Dry land
-0.1–0 : Saturated soil or muddy flood water
0–0.3  : Water-covered areas (flooded fields, rivers overflowing)
> 0.3  : Clear open water

## AWEI interpretation (Automated Water Extraction Index — best for turbid water)
< 0    : No water
> 0    : Water present (works better than NDWI for turbid flood conditions)

## NDBI interpretation
< 0    : Vegetation-dominated
0–0.2  : Mixed urban/vegetation
> 0.2  : Built-up area dominated

## CRITICAL for flood analysis
When analysis_type is "flood" and NDWI mean is between -0.15 and 0.1:
- Do NOT conclude "no flooding detected"
- Instead, report that TURBID FLOOD WATER is likely present (mud suppresses NDWI signal)
- Compare NDWI to expected baseline for the region (typically -0.3 to -0.2 for dry agricultural land in Valencia)
- A shift of +0.1 or more in NDWI from the dry baseline IS evidence of flooding
- Use MNDWI and AWEI results to confirm if available

Your response must be written in Spanish.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────────────────────────────────────
class ReporterAgent:
    """LangGraph node: synthesize results → Markdown report + Folium map."""

    def __init__(self, model: str | None = None):
        self.llm = ChatGroq(
            model=model or GROQ_MODEL,
            temperature=0.3,
            max_tokens=2048,
        )

    def __call__(self, state: GeoAgentState) -> dict:
        # ── Gather all context ─────────────────────────────────────────────
        plan            = state["plan"]
        location        = state["location"]
        date_range      = state["date_range"]
        analysis_type   = state["analysis_type"]
        computed_indices = state.get("computed_indices") or {}
        selected_scene  = state.get("selected_scene") or {}
        output_files    = state.get("output_files") or []
        code_iterations = state.get("code_iterations", 1)

        # ── Build report context for the LLM ──────────────────────────────
        context = {
            "user_query":    state["user_query"],
            "location":      location,
            "date_range":    date_range,
            "analysis_type": analysis_type,
            "indices":       computed_indices,
            "scene_info":    {
                "id":           selected_scene.get("id", "N/A"),
                "date":         selected_scene.get("date", "N/A"),
                "cloud_cover":  selected_scene.get("cloud_cover", "N/A"),
                "collection":   selected_scene.get("collection", "N/A"),
            },
            "plan_summary":  plan.get("summary", ""),
            "output_files":  output_files,
        }

        flood_note = ""
        if analysis_type == "flood":
            sar_available    = state.get("sar_available") or False
            pre_scene_files  = state.get("pre_scene_files") or []
            sar_stats        = computed_indices.get("SAR_CHANGE", {})

            if sar_available and sar_stats:
                sar_block = (
                    f"\n\nSentinel-1 SAR Change Detection (penetra nubes — detecta agua turbia):\n"
                    f"  · Cambio medio de retrodispersión: {sar_stats.get('mean_change_dB', '?')} dB\n"
                    f"  · Área con posible inundación (Δ < -3 dB): {sar_stats.get('pct_possible_flood', '?')}%\n"
                    f"  · Área con inundación confirmada (Δ < -5 dB): {sar_stats.get('pct_confirmed_flood', '?')}%\n"
                    "Incluye una sección dedicada SAR en el informe explicando qué significa cada umbral. "
                    "Destaca que SAR es inmune a nubes y agua turbia — supera las limitaciones del NDWI óptico."
                )
            else:
                sar_block = (
                    "\n\nNota: Datos Sentinel-1 SAR no disponibles para este evento. "
                    "La extensión de la inundación se basa únicamente en índices ópticos."
                )

            flood_note = (
                "\n\nIMPORTANT: This is a FLOOD analysis. "
                "Flash flood water is turbid (mud/sediment), which suppresses NDWI. "
                "Use the TURBID WATER thresholds from your system prompt. "
                "A typical dry Valencia agricultural baseline is NDWI ≈ -0.25. "
                "Any shift above -0.10 is significant evidence of flooding."
                + sar_block
            )

        prompt = (
            f"Generate a report based on the following analysis results:\n\n"
            f"```json\n{json.dumps(context, indent=2, ensure_ascii=False)}\n```\n\n"
            f"Original user question: \"{state['user_query']}\"\n"
            f"{flood_note}\n\n"
            f"Respond with a complete Markdown report."
        )

        response = self.llm.invoke(
            [
                SystemMessage(content=REPORTER_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        report_md = response.content.strip()

        # ── Generate interactive Folium map ────────────────────────────────
        map_html = _build_folium_map(
            location=location,
            date_range=date_range,
            computed_indices=computed_indices,
            output_files=output_files,
        )

        # Save map to file
        map_path = OUTPUTS_DIR / f"map_{state['session_id']}.html"
        map_path.write_text(map_html, encoding="utf-8")

        # Save report to file
        report_path = OUTPUTS_DIR / f"report_{state['session_id']}.md"
        report_path.write_text(report_md, encoding="utf-8")

        logger.info("[Reporter] Report and map generated.")

        return {
            "report_markdown": report_md,
            "map_html": map_html,
            "chart_paths": [str(p) for p in OUTPUTS_DIR.glob("*.png")],
            "status": "completed",
            "current_agent": "done",
            "messages": [
                {
                    "agent": "reporter",
                    "type": "completed",
                    "content": "Report and map generated successfully.",
                }
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Folium map builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_folium_map(
    location: dict,
    date_range: dict,
    computed_indices: dict,
    output_files: list[str],
) -> str:
    """Build an interactive Folium map for the analysis area."""
    bbox = location.get("bbox", [-10, 35, 5, 45])
    centroid = location.get("centroid") or [
        (bbox[0] + bbox[2]) / 2,
        (bbox[1] + bbox[3]) / 2,
    ]

    m = folium.Map(
        location=[centroid[1], centroid[0]],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    # Add alternative tile layers
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
    ).add_to(m)

    # Draw study area bounding box
    folium.Rectangle(
        bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
        color="#e63946",
        fill=True,
        fill_opacity=0.05,
        weight=2,
        tooltip=f"Study area: {location.get('name', '')}",
    ).add_to(m)

    # Add a marker at centroid with popup summary
    stats_html = "<b>Analysis Results</b><br>"
    for idx_name, stats in computed_indices.items():
        if isinstance(stats, dict):
            mean = stats.get("mean", "N/A")
            stats_html += f"<b>{idx_name}</b>: mean={mean:.3f}<br>" if isinstance(mean, float) else f"<b>{idx_name}</b>: {stats}<br>"

    folium.Marker(
        location=[centroid[1], centroid[0]],
        popup=folium.Popup(stats_html, max_width=300),
        tooltip=location.get("name", "Study Area"),
        icon=folium.Icon(color="red", icon="satellite", prefix="fa"),
    ).add_to(m)

    # Add PNG overlays for index images if they exist (thumbnailed to avoid MemoryError)
    for fpath in output_files:
        p = Path(fpath)
        if p.suffix.lower() == ".png" and p.exists():
            try:
                img = Image.open(p).convert("RGBA")
                img.thumbnail((512, 512), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                folium.raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{img_b64}",
                    bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
                    opacity=0.7,
                    name=p.stem,
                ).add_to(m)
            except Exception as exc:
                logger.warning("[Reporter] Could not add image overlay %s: %s", p, exc)

    folium.LayerControl().add_to(m)

    # Mini-map
    folium_plugins.MiniMap(toggle_display=True).add_to(m)

    # Scale bar
    folium_plugins.MousePosition().add_to(m)

    return m._repr_html_()