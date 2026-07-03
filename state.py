"""
state.py - Shared state definitions for the GeoMultiAgent LangGraph workflow.

Each agent reads and writes to this shared TypedDict state, which LangGraph
passes between nodes as the conversation progresses.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
import operator


# ─────────────────────────────────────────────
#  Message accumulator helper (LangGraph style)
# ─────────────────────────────────────────────
def _append(existing: list, new: list) -> list:
    """Reducer: append new items to the existing list."""
    return existing + new


# ─────────────────────────────────────────────
#  Main workflow state
# ─────────────────────────────────────────────
class GeoAgentState(TypedDict):

    # ── Entrada ──────────────────────────────────────────────
    user_query: str                           # Consulta original del usuario
    session_id: str                           # Identificador único de sesión

    # ── Salidas del Planificador ─────────────────────────────
    plan: Optional[dict]                      # JSON con bbox, fechas, índices, satélites
    location: Optional[dict]                  # Localización geográfica (bbox, nombre, centroide)
    date_range: Optional[dict]                # {"start": ..., "end": ...}
    analysis_type: Optional[str]              # "vegetation", "flood", "urban"...
    required_indices: Optional[list[str]]     # ["NDVI", "NDWI"...]
    satellites: Optional[list[str]]           # ["Sentinel-2", "Landsat-8"]

    # ── Salidas del Agente de Adquisición ────────────────────
    available_scenes: Optional[list[dict]]    # Escenas encontradas en el catálogo
    selected_scene: Optional[dict]            # Escena seleccionada
    downloaded_files: Optional[list[str]]     # Rutas locales de los .tif descargados (S2 + SAR post)
    pre_scene_files: Optional[list[str]]      # Rutas SAR pre-evento para change detection
    sar_available: Optional[bool]             # True cuando SAR change detection está listo
    acquisition_error: Optional[str]

    # ── Salidas del Agente Analista ──────────────────────────
    generated_code: Optional[str]             # Script Python generado
    execution_result: Optional[dict]          # stdout, stderr y excepción capturada
    computed_indices: Optional[dict]          # {"NDVI": {"mean": ..., "std": ...}}
    output_files: Optional[list[str]]         # Rutas de PNG y GeoTIFF generados
    code_iterations: int                      # Iteraciones de depuración usadas
    analysis_error: Optional[str]

    # ── Salidas del Reporter ─────────────────────────────────
    report_markdown: Optional[str]            # Informe final en lenguaje natural
    map_html: Optional[str]                   # Mapa Folium como HTML
    chart_paths: Optional[list[str]]          # Rutas de las visualizaciones

    # ── Control del flujo ────────────────────────────────────
    messages: Annotated[list[dict], _append]  # Log de eventos (acumulativo)
    current_agent: Optional[str]              # Agente activo en cada momento
    status: str                               # "running" | "completed" | "error"
    error_message: Optional[str]              # Error irrecuperable si lo hay
