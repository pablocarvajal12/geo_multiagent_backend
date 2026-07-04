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

## General principle: pixel-based evidence over area averages
This applies to EVERY analysis type, not just floods. A real localized phenomenon (a flood, a burn
scar, an urban expansion, snow melt) is usually a fraction of a larger, mostly-unaffected bounding
box — a single area-wide mean dilutes it away. Whenever INDEX_EXTENT / INDEX_CHANGE / SAR_CHANGE
statistics are present in the data (see below), treat their pixel percentages as the primary
evidence, and treat the plain area-wide mean/std of each index as secondary, supporting context.

## NDVI interpretation guide
< 0.1  : Bare soil, rock, urban, water
0.1–0.2: Sparse vegetation, degraded land
0.2–0.4: Moderate vegetation (crops, grassland)
0.4–0.6: Dense vegetation, healthy crops
> 0.6  : Very dense vegetation (tropical forest)

## NDWI interpretation (standard: clear water)
< -0.1 : Dry land / dense vegetation — no water signal
-0.1–0 : Possibly saturated soil or turbid water (mud, sediment) — AMBIGUOUS, needs confirmation
0–0.2  : Probable open water or very wet soil / flooded fields
> 0.2  : Confirmed open water

## Context: turbid flood water (flood analyses only)
Flash flood water (e.g. DANA-type storms) carries mud and sediment, which raises SWIR reflectance
and can suppress the NDWI signal compared to clear water. This means a moderately negative NDWI
(between -0.1 and 0) is AMBIGUOUS on its own — it could be dry soil OR turbid flood water. Do NOT
resolve this ambiguity by assumption or by picking the more dramatic conclusion. Instead, rely on
the INDEX_EXTENT, SAR_CHANGE and INDEX_CHANGE statistics described below — these report the actual
measured percentage of pixels with a water/flood signature, which is the reliable evidence.

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

## NBR / burn severity (fire analyses)
NBR alone is not diagnostic — burn severity needs the CHANGE in NBR (pre-event minus post-event,
i.e. NBR drops after a fire). If INDEX_CHANGE includes nbr_pct_low_severity_burn /
nbr_pct_high_severity_burn, use those percentages, not the plain NBR mean.

## NDSI (snow analyses)
> 0.4  : Likely snow/ice covered.
Note: NDSI uses the same band ratio as MNDWI (green/SWIR1) — snow and turbid water can look similar
without extra context (e.g. temperature, season). Treat with some caution in mixed scenes.

## INDEX_EXTENT / INDEX_CHANGE / SAR_CHANGE (pixel-based evidence, any analysis type)
When present in the data, these report the ACTUAL MEASURED percentage of pixels in the study area
crossing a meaningful threshold for the relevant index — far more reliable than a single area-wide
mean, because the real phenomenon is usually localized within a larger, mostly-unaffected bounding
box and gets diluted away in a simple average.
- If these percentages are low (a few % or less) and the area-wide means are ambiguous, the correct
  conclusion is that there is little to no evidence of the phenomenon in the analyzed scene — say so
  plainly.
- If these percentages are substantial (tens of %), report that as clear evidence.
- If sources disagree (e.g. SAR shows change but optical indices don't, or vice versa), report that
  disagreement transparently instead of silently picking the conclusion that seems more dramatic.
- Any stat named "pct_area_valid" tells you what fraction of the AOI actually has usable data (the
  rest was cloud/shadow and is unknown, not "unaffected") — always mention this coverage limit.
- Any stat suffixed "_excl_urban" excludes pixels with a built-up spectral signature, which is a
  known source of false water/flood positives in NDWI/MNDWI — prefer it over the unfiltered version.

## CRITICAL RULE
Never state a conclusion the numbers do not support. Base your conclusion strictly on the computed
statistics actually provided (area means AND pixel percentages), and say explicitly when the
evidence is weak, absent, or contradictory. Do not invent or assume a baseline value that was not
measured from the data provided.

Your response must be written in Spanish.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers de evidencia — genéricos, no dependen de analysis_type
# ─────────────────────────────────────────────────────────────────────────────

def _format_evidence_block(title: str, stats: dict, note: str = "") -> str:
    """Convierte un dict de estadísticas en texto legible para inyectar en el
    prompt del LLM. Genérico: funciona para cualquier índice o analysis_type,
    no solo inundación."""
    if not stats:
        return ""
    lines = [f"\n\n{title}:"]
    pct_area_valid = stats.get("pct_area_valid")
    if pct_area_valid is not None:
        lines.append(
            f"  · IMPORTANTE — cobertura: solo el {pct_area_valid}% del área pudo evaluarse "
            "(el resto quedó fuera por nubes/sombras u otra falta de dato). Los demás porcentajes "
            "de este bloque son relativos a esa área válida, NO al área total."
        )
    for key, value in stats.items():
        if key == "pct_area_valid":
            continue
        suffix = "%" if "pct" in key else ""
        lines.append(f"  · {key}: {value}{suffix}")
    if note:
        lines.append(note)
    return "\n".join(lines)


def _build_evidence_table_md(computed_indices: dict) -> str:
    """Tabla con TODOS los valores calculados, generada directamente desde los
    datos (no por el LLM) para que ningún número se pierda en el resumen —
    válido para cualquier tipo de análisis."""
    if not computed_indices:
        return ""
    lines = [
        "\n\n## Tabla de Evidencia (datos completos)\n",
        "| Fuente | Estadística | Valor |",
        "|---|---|---|",
    ]
    for source_name, stats in computed_indices.items():
        if not isinstance(stats, dict):
            continue
        for stat_name, value in stats.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
            else:
                value_str = str(value)
            lines.append(f"| {source_name} | {stat_name} | {value_str} |")
    return "\n".join(lines) + "\n"


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

        # Evidencia por píxel — genérica, se aplica a CUALQUIER analysis_type,
        # no solo inundación (INDEX_EXTENT/INDEX_CHANGE se calculan siempre que
        # haya bandas ópticas legibles; SAR_CHANGE solo existe para inundación).
        index_extent  = computed_indices.get("INDEX_EXTENT", {})
        index_change  = computed_indices.get("INDEX_CHANGE", {})
        sar_stats     = computed_indices.get("SAR_CHANGE", {})
        sar_available = state.get("sar_available") or False

        evidence_note = ""
        if index_extent:
            evidence_note += _format_evidence_block(
                "Evidencia por píxel en la escena analizada (NO es una media de área — evita "
                "que una señal localizada se diluya)",
                index_extent,
                note=(
                    "Los stats '_excl_urban' descuentan confusión con zonas edificadas en "
                    "NDWI/MNDWI (limitación conocida) — son más fiables que la versión sin filtrar."
                ),
            )
        if index_change:
            evidence_note += _format_evidence_block(
                "Cambio real pre/post evento (comparación con una escena Sentinel-2 previa "
                "real, NO un baseline asumido)",
                index_change,
            )
        if sar_available and sar_stats:
            evidence_note += (
                "\n\nSentinel-1 SAR Change Detection (penetra nubes — detecta agua turbia):\n"
                f"  · Cambio medio de retrodispersión: {sar_stats.get('mean_change_dB', '?')} dB\n"
                f"  · Área con posible inundación (Δ < -3 dB): {sar_stats.get('pct_possible_flood', '?')}%\n"
                f"  · Área con inundación confirmada (Δ < -5 dB): {sar_stats.get('pct_confirmed_flood', '?')}%\n"
                "Incluye una sección dedicada SAR en el informe explicando qué significa cada umbral."
            )
        elif analysis_type == "flood":
            evidence_note += (
                "\n\nNota: Datos Sentinel-1 SAR no disponibles para este evento. "
                "La extensión de la inundación se basa en los índices ópticos disponibles."
            )

        if evidence_note:
            evidence_note = (
                "\n\nEvidencia cuantitativa adicional (calculada directamente sobre los datos, "
                "independiente del código generado por el LLM analista). Basa tu conclusión "
                "principalmente en los porcentajes de píxeles de abajo, no solo en la media de "
                "área de cada índice. Si la evidencia es débil, inexistente o contradictoria "
                "entre fuentes, dilo explícitamente en vez de forzar una conclusión."
                + evidence_note
            )

        prompt = (
            f"Generate a report based on the following analysis results:\n\n"
            f"```json\n{json.dumps(context, indent=2, ensure_ascii=False)}\n```\n\n"
            f"Original user question: \"{state['user_query']}\"\n"
            f"{evidence_note}\n\n"
            f"Respond with a complete Markdown report."
        )

        response = self.llm.invoke(
            [
                SystemMessage(content=REPORTER_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        report_md = response.content.strip()
        # Tabla determinista con TODOS los valores calculados — no depende de que
        # el LLM decida mencionarlos (ver _build_evidence_table_md).
        report_md += _build_evidence_table_md(computed_indices)

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