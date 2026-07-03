"""
backend.py – FastAPI Backend (corregido)
- workflow.stream() corre en ThreadPoolExecutor para no bloquear el event loop
- El estado final se guarda en disco como state_{session_id}.json
- Endpoint GET /api/cesium-data/{session_id} devuelve datos listos para CesiumJS
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "./outputs"))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Thread pool dedicado para correr el pipeline síncrono de LangGraph
_executor = ThreadPoolExecutor(max_workers=4)

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GeoMultiAgent API",
    description="AI-powered Earth Observation — optimizado para CesiumJS",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_state(session_id: str, state: dict) -> None:
    """Persiste el estado final del pipeline en disco."""
    out = OUTPUTS_DIR / f"state_{session_id}.json"
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(state, f, default=str, ensure_ascii=False)
    except Exception as exc:
        logger.warning("[save_state] No se pudo guardar: %s", exc)


def _load_state(session_id: str) -> dict | None:
    """Carga el estado guardado para una sesión."""
    path = OUTPUTS_DIR / f"state_{session_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Rutas básicas
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoint CesiumJS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_bbox(state: dict) -> list[float] | None:
    """Extrae bounding box [W, S, E, N] del estado del pipeline."""
    plan = state.get("plan") or {}

    loc = state.get("location") or plan.get("location")
    if isinstance(loc, dict):
        # Formato del Planner: location.bbox = [min_lon, min_lat, max_lon, max_lat]
        bbox = loc.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return [float(x) for x in bbox]

        # Fallback: location con lat/lon directos
        lat = loc.get("lat") or loc.get("latitude")
        lon = loc.get("lon") or loc.get("longitude")
        if lat and lon:
            delta = 0.8
            return [lon - delta, lat - delta, lon + delta, lat + delta]

        # Fallback: centroid [lon, lat]
        centroid = loc.get("centroid")
        if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
            lon, lat = float(centroid[0]), float(centroid[1])
            delta = 0.8
            return [lon - delta, lat - delta, lon + delta, lat + delta]

    # bbox directa en el plan (formato alternativo)
    bbox = plan.get("bbox") or plan.get("bounding_box")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [float(x) for x in bbox]

    # coordenadas como string "lat, lon"
    if isinstance(loc, str):
        import re
        nums = re.findall(r"[-+]?\d+\.?\d*", loc)
        if len(nums) >= 2:
            lat, lon = float(nums[0]), float(nums[1])
            delta = 0.8
            return [lon - delta, lat - delta, lon + delta, lat + delta]

    return None


@app.get("/api/cesium-data/{session_id}")
async def get_cesium_data(session_id: str) -> JSONResponse:
    """
    Devuelve todo lo que CesiumJS necesita para renderizar los resultados:
    - bbox [W, S, E, N]
    - center {lat, lon}
    - layers: lista de capas (imagery PNG, GeoJSON, etc.)
    - computed_indices, report, generated_code
    """
    state = _load_state(session_id)
    if state is None:
        return JSONResponse(
            {"error": f"Sesión '{session_id}' no encontrada"},
            status_code=404,
        )

    bbox = _parse_bbox(state)
    center = None
    if bbox:
        center = {
            "lon": (bbox[0] + bbox[2]) / 2,
            "lat": (bbox[1] + bbox[3]) / 2,
        }

    # Capas: buscar PNGs de índices en outputs/
    _INDEX_COLORMAP = {
        "ndvi": "ndvi", "evi": "ndvi", "evi2": "ndvi",
        "ndwi": "ndwi", "mndwi": "ndwi", "ndsi": "ndwi",
        "ndbi": "thermal", "nbr": "thermal", "bsi": "grayscale",
    }
    layers = []
    for idx, cmap in _INDEX_COLORMAP.items():
        png = OUTPUTS_DIR / f"{idx}.png"
        if png.exists():
            layers.append({
                "id": f"index_{idx}",
                "name": idx.upper(),
                "type": "imagery_url",
                "url": f"/outputs/{idx}.png",
                "colormap": cmap,
            })

    # GeoJSON del área de estudio
    study_geojson = None
    if bbox:
        w, s, e, n = bbox
        study_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[w,s],[e,s],[e,n],[w,n],[w,s]]],
                },
                "properties": {
                    "name": "Área de análisis",
                    "session_id": session_id,
                    "analysis_type": str(state.get("analysis_type") or ""),
                },
            }],
        }

    return JSONResponse({
        "session_id": session_id,
        "status": state.get("status", "unknown"),
        "bbox": bbox,
        "center": center,
        "layers": layers,
        "study_geojson": study_geojson,
        "computed_indices": state.get("computed_indices"),
        "report": state.get("report_markdown"),
        "generated_code": state.get("generated_code"),
        "error": state.get("error_message"),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  WebSocket principal
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_sync(query: str, session_id: str) -> tuple[dict, list[dict]]:
    """
    Corre el workflow de LangGraph de forma SÍNCRONA en un hilo separado.
    Retorna (final_state, list_of_ws_events).
    Los eventos se acumulan aquí y se envían al WebSocket desde el hilo async.
    """
    from workflow import build_workflow
    from state import GeoAgentState

    workflow = build_workflow()
    initial: GeoAgentState = {
        "user_query": query,
        "session_id": session_id,
        "plan": None, "location": None, "date_range": None,
        "analysis_type": None, "required_indices": None,
        "satellites": None, "available_scenes": None,
        "selected_scene": None, "downloaded_files": None,
        "acquisition_error": None, "generated_code": None,
        "execution_result": None, "computed_indices": None,
        "output_files": None, "code_iterations": 0,
        "analysis_error": None, "report_markdown": None,
        "map_html": None, "chart_paths": None,
        "messages": [], "current_agent": "planner",
        "status": "running", "error_message": None,
    }

    events: list[dict] = []
    state = dict(initial)
    prev_msgs: list = []

    for event in workflow.stream(initial):
        for node_name, partial in event.items():
            events.append({
                "type": "agent_start",
                "agent": node_name,
                "content": f"Agente '{node_name}' en ejecución…",
            })
            state.update(partial)
            new_msgs = state.get("messages", [])
            for msg in new_msgs[len(prev_msgs):]:
                events.append({
                    "type": "agent_log",
                    "agent": msg.get("agent", node_name),
                    "content": msg.get("content", ""),
                    "data": msg.get("data"),
                })
            prev_msgs = list(new_msgs)

    return state, events


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info("[WS] Conectado — sesión %s", session_id)

    try:
        raw = await websocket.receive_text()
        payload = json.loads(raw)
        query = payload.get("query", "").strip()
        if payload.get("session_id"):
            session_id = payload["session_id"]

        if not query:
            await websocket.send_json({"type": "error", "content": "La consulta está vacía."})
            return

        await websocket.send_json({
            "type": "status",
            "agent": "system",
            "content": "Pipeline iniciado…",
            "session_id": session_id,
        })

        # ── Correr el pipeline en un hilo para no bloquear asyncio ────────
        loop = asyncio.get_event_loop()
        final_state, events = await loop.run_in_executor(
            _executor,
            _run_pipeline_sync,
            query,
            session_id,
        )

        # Enviar eventos acumulados al frontend
        for evt in events:
            try:
                await websocket.send_json(evt)
            except Exception:
                break

        # Persistir estado para que /api/cesium-data pueda leerlo
        _save_state(session_id, final_state)

        # Construir URLs de mapa si existen
        map_url = None
        map_path = OUTPUTS_DIR / f"map_{session_id}.html"
        if map_path.exists():
            map_url = f"/outputs/map_{session_id}.html"

        await websocket.send_json({
            "type": "completed",
            "agent": "system",
            "content": "Análisis finalizado.",
            "data": {
                "session_id":       session_id,
                "status":           final_state.get("status"),
                "report":           final_state.get("report_markdown"),
                "map_url":          map_url,
                "computed_indices": final_state.get("computed_indices"),
                "plan":             final_state.get("plan"),
                "generated_code":   final_state.get("generated_code"),
                "location":         final_state.get("location"),
                "error":            final_state.get("error_message"),
                # URL para que el frontend pida los datos Cesium
                "cesium_data_url":  f"/api/cesium-data/{session_id}",
            },
        })

    except WebSocketDisconnect:
        logger.info("[WS] Desconectado — sesión %s", session_id)
    except Exception as exc:
        logger.exception("[WS] Error crítico — sesión %s: %s", session_id, exc)
        try:
            await websocket.send_json({"type": "error", "content": str(exc)})
        except Exception:
            pass