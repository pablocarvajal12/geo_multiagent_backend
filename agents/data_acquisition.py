"""
agents/data_acquisition.py - Data Acquisition Agent

Autonomously searches satellite data catalogues (STAC / Copernicus / NASA),
selects the best available scene and downloads the required bands.

Supported backends (tried in order):
  1. Microsoft Planetary Computer  (STAC, free, no auth for most datasets)
  2. Element84 Earth Search        (STAC, free, AWS Open Data)
  3. Copernicus SciHub / CDSE      (Sentinel-2, requires account)
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from state import GeoAgentState

logger = logging.getLogger(__name__)

from langchain_groq import ChatGroq

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LangChain Tools  (callable by the agent via tool calling)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_planetary_computer(
    bbox: list[float],
    start_date: str,
    end_date: str,
    collections: list[str],
    cloud_cover_max: int = 20,
    limit: int = 10,
) -> dict:
    """
    Search Microsoft Planetary Computer STAC catalog for satellite imagery.

    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat] in WGS-84
        start_date: ISO date string e.g. "2024-01-01"
        end_date:   ISO date string e.g. "2024-03-31"
        collections: STAC collection IDs, e.g. ["sentinel-2-l2a", "landsat-c2-l2"]
        cloud_cover_max: maximum cloud cover percentage (0-100)
        limit: max number of items to return

    Returns:
        Dictionary with 'items' list and 'total_found' count.
    """
    results = {"items": [], "total_found": 0, "source": "planetary_computer"}

    for collection in collections:
        url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
        payload = {
            "collections": [collection],
            "bbox": bbox,
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": cloud_cover_max}},
            "limit": limit,
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        }
        try:
            resp = httpx.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("features", [])
            for item in items:
                results["items"].append(
                    {
                        "id": item["id"],
                        "collection": collection,
                        "date": item["properties"].get("datetime", ""),
                        "cloud_cover": item["properties"].get("eo:cloud_cover", -1),
                        # % de píxeles clasificados como agua (clave para detectar
                        # la escena de una inundación dentro de la ventana)
                        "water_percentage": item["properties"].get("s2:water_percentage"),
                        "bbox": item.get("bbox", []),
                        "assets": {
                            k: v.get("href", "")
                            for k, v in item.get("assets", {}).items()
                            if "href" in v
                        },
                        "source": "planetary_computer",
                    }
                )
            results["total_found"] += data.get("numberMatched", len(items))
        except Exception as exc:
            logger.warning("[DataAcquisition] PC search failed for %s: %s", collection, exc)

    return results


@tool
def search_earth_search(
    bbox: list[float],
    start_date: str,
    end_date: str,
    collections: list[str],
    cloud_cover_max: int = 20,
    limit: int = 10,
) -> dict:
    """
    Search Element84 Earth Search STAC catalog (AWS Open Data).

    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat] in WGS-84
        start_date: ISO date string
        end_date:   ISO date string
        collections: STAC collection IDs e.g. ["sentinel-2-l2a"]
        cloud_cover_max: maximum cloud cover %
        limit: max number of items

    Returns:
        Dictionary with 'items' list and metadata.
    """
    results = {"items": [], "total_found": 0, "source": "earth_search"}

    url = "https://earth-search.aws.element84.com/v1/search"
    payload = {
        "collections": collections,
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": cloud_cover_max}},
        "limit": limit,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }
    try:
        resp = httpx.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("features", [])
        for item in items:
            results["items"].append(
                {
                    "id": item["id"],
                    "collection": item.get("collection", collections[0]),
                    "date": item["properties"].get("datetime", ""),
                    "cloud_cover": item["properties"].get("eo:cloud_cover", -1),
                    "water_percentage": item["properties"].get("s2:water_percentage"),
                    "bbox": item.get("bbox", []),
                    "assets": {
                        k: v.get("href", "")
                        for k, v in item.get("assets", {}).items()
                        if "href" in v
                    },
                    "source": "earth_search",
                }
            )
        results["total_found"] = data.get("numberMatched", len(items))
    except Exception as exc:
        logger.warning("[DataAcquisition] Earth Search failed: %s", exc)

    return results


@tool
def select_best_scene(scenes: list[dict], preference: str = "lowest_cloud") -> dict:
    """
    Select the best scene from a list of available scenes.

    Args:
        scenes: list of scene dicts (from search tools)
        preference: "lowest_cloud" | "most_recent" | "earliest" | "best_coverage"

    Returns:
        The selected scene dict.
    """
    if not scenes:
        return {"error": "No scenes available to select from."}

    if preference == "most_recent":
        scenes_sorted = sorted(
            scenes, key=lambda s: s.get("date", ""), reverse=True
        )
    elif preference == "earliest":
        scenes_sorted = sorted(
            scenes, key=lambda s: s.get("date", "")
        )
    else:  # lowest_cloud (default)
        scenes_sorted = sorted(
            scenes,
            key=lambda s: (s.get("cloud_cover", 100), s.get("date", "")),
        )

    best = scenes_sorted[0]
    logger.info(
        "[DataAcquisition] Selected scene: %s  cloud=%.1f%%  date=%s",
        best["id"],
        best.get("cloud_cover", -1),
        best.get("date", ""),
    )
    return best


@tool
def download_scene_bands(
    scene: dict,
    bands: list[str],
    output_dir: str = str(DATA_DIR),
) -> dict:
    """
    Download specific spectral bands from a selected STAC scene.

    Args:
        scene: Scene dict as returned by select_best_scene
        bands: List of band asset keys to download, e.g. ["B04", "B08", "B02"]
        output_dir: Local directory to save files

    Returns:
        Dict mapping band name → local file path (or error message).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    assets = scene.get("assets", {})
    source = scene.get("source", "") 

    for band in bands:
        url = assets.get(band) or assets.get(band.lower()) or assets.get(band.upper())
        if not url:
            downloaded[band] = f"ERROR: asset '{band}' not found in scene"
            continue

        # Sign URL if from Planetary Computer (free anonymous SAS token) 
        if source == "planetary_computer" or "blob.core.windows.net" in url:
            try:
                import planetary_computer
                url = planetary_computer.sign(url)
                logger.info("[DataAcquisition] URL signed via Planetary Computer SAS token")
            except Exception as sign_exc:
                logger.warning("[DataAcquisition] Could not sign URL: %s", sign_exc)

        local_path = out_dir / f"{scene['id']}_{band}.tif"
        if local_path.exists():
            logger.info("[DataAcquisition] Already cached: %s", local_path)
            downloaded[band] = str(local_path)
            continue
        try:
            with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1024 * 256):
                        f.write(chunk)
            downloaded[band] = str(local_path)
            logger.info("[DataAcquisition] Downloaded %s → %s", band, local_path)
        except Exception as exc:
            # Sin este log, una descarga fallida desaparecía en silencio y solo se
            # detectaba muchos pasos despues (p.ej. "banda insuficiente" en el
            # analista), sin ninguna pista de la causa real.
            logger.warning("[DataAcquisition] Descarga fallida para banda %s: %s", band, exc)
            # Si la descarga fallo a mitad de escritura, el fichero parcial se queda
            # en disco y el chequeo "if local_path.exists()" de arriba lo trataria
            # como valido (ya cacheado) en el siguiente intento, propagando un
            # archivo corrupto indefinidamente.
            if local_path.exists():
                try:
                    local_path.unlink()
                except OSError:
                    pass
            downloaded[band] = f"ERROR: {exc}"

    return downloaded


# ─────────────────────────────────────────────────────────────────────────────
#  Agent class
# ─────────────────────────────────────────────────────────────────────────────

ACQUISITION_SYSTEM = """
You are the Data Acquisition Agent of a geospatial multi-agent system.
Your job is to find and download the best available satellite imagery for
a given analysis plan.

## Workflow
1. Use `search_planetary_computer` first (best reliability).
2. If fewer than 3 scenes found, also try `search_earth_search`.
3. Call `select_best_scene` with the combined results.
4. Identify which spectral bands are needed for the requested indices
   and call `download_scene_bands`.
5. Return a JSON summary with:
   - selected_scene: the chosen scene metadata
   - downloaded_files: dict of band → local path
   - notes: any important observations

## Band naming conventions
Sentinel-2 L2A (STAC):  B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
Landsat-8/9 C2 L2:      blue green red nir08 swir16 swir22

## Index → bands mapping
NDVI  → red (B04/red) + nir (B08/nir08)
EVI   → blue + red + nir
NDWI  → green (B03/green) + nir
MNDWI → green + swir1 (B11/swir16)
NDBI  → swir1 + nir
NBR   → nir + swir2 (B12/swir22)
NDSI  → green + swir1

Always use the best available source.  Prefer Sentinel-2 for ≤ 100 km² areas,
Landsat for larger ones, MODIS for continental scale.
Your response must be written in Spanish.
"""

SENTINEL2_COLLECTIONS = ["sentinel-2-l2a"]
LANDSAT_COLLECTIONS   = ["landsat-c2-l2"]

_MGRS_TILE_RE = re.compile(r"_T(\d{2}[A-Z]{3})_")


def _extract_mgrs_tile(scene_id: str) -> Optional[str]:
    """
    Extrae el tile MGRS (p.ej. '29TPG') del ID de una escena Sentinel-2
    (formato S2B_MSIL2A_..._R137_T29TPG_...). Se usa para exigir que la
    escena pre-evento caiga en el MISMO tile que la post-evento: regiones
    grandes (p.ej. Galicia) pueden abarcar varios tiles, y comparar píxel a
    píxel dos escenas de tiles distintos compara zonas geográficas distintas
    aunque los arrays tengan las mismas dimensiones.
    """
    m = _MGRS_TILE_RE.search(scene_id or "")
    return m.group(1) if m else None


def _select_flood_scene_by_water_anomaly(scenes: list) -> Optional[dict]:
    """
    Escena de inundación por ANOMALÍA de clasificación de agua: dentro de cada
    tile MGRS, mide cuánto se desvía el s2:water_percentage de cada escena
    respecto a la MEDIANA de su tile en la ventana, en CUALQUIER dirección.

    Validado con las escenas reales de la DANA de Valencia (oct-2024): en la
    escena del evento el water%% se DESPLOMA (17.7%% frente a ~45%% normal del
    tile), porque el clasificador SCL no reconoce como agua la riada turbia y
    las nubes de la tormenta reducen los píxeles clasificados. Una inundación
    de agua limpia lo sube. Por eso cuenta la desviación absoluta, no el signo,
    y nunca el porcentaje absoluto (un tile costero siempre ronda el 50%% de mar).

    Entre las escenas anómalas (a menos de 5 puntos de la desviación máxima) se
    elige la MÁS TEMPRANA: es la más cercana al pico del evento.

    Devuelve None si ningún tile tiene >= 3 escenas con metadato de agua o si la
    desviación máxima no llega a 10 puntos (el llamante cae a 'earliest').
    """
    from collections import defaultdict
    from statistics import median

    by_tile: dict = defaultdict(list)
    for s in scenes or []:
        tile = _extract_mgrs_tile(s.get("id", ""))
        wp = s.get("water_percentage")
        if tile and isinstance(wp, (int, float)):
            by_tile[tile].append(s)

    MIN_DEVIATION_PCT_POINTS = 10.0
    FINALIST_MARGIN = 5.0

    candidates: list = []  # (desviación, escena)
    for tile, tile_scenes in by_tile.items():
        if len(tile_scenes) < 3:
            continue  # con menos escenas la mediana no define un "normal" del tile
        baseline = median(s["water_percentage"] for s in tile_scenes)
        for s in tile_scenes:
            deviation = abs(s["water_percentage"] - baseline)
            if deviation >= MIN_DEVIATION_PCT_POINTS:
                candidates.append((deviation, s))

    if not candidates:
        return None

    top = max(dev for dev, _s in candidates)
    finalists = [s for dev, s in candidates if dev >= top - FINALIST_MARGIN]
    best = min(finalists, key=lambda s: s.get("date", ""))
    logger.info(
        "[DataAcquisition] Escena por anomalía de agua: %s (water%%=%.2f, desviación máx=%.1f pts, %d finalistas)",
        best.get("id"), best.get("water_percentage", -1), top, len(finalists),
    )
    return best


class DataAcquisitionAgent:
    """LangGraph node: find and download satellite imagery."""

    TOOLS = [
        search_planetary_computer,
        search_earth_search,
        select_best_scene,
        download_scene_bands,
    ]

    def __init__(self, model: str | None = None):
        self.llm = ChatGroq(
            model=model or GROQ_MODEL,
            temperature=0.0,
            max_tokens=2048,
        )

    def __call__(self, state: GeoAgentState) -> dict:
        plan     = state["plan"]
        location = state["location"]
        date_range = state["date_range"]
        indices  = state["required_indices"]
        satellites = state["satellites"]

        # Determinar colecciones STAC según satélite preferido
        collections = []
        for sat in satellites:
            if "Sentinel" in sat:
                collections.append("sentinel-2-l2a")
            # Ignoramos Landsat por ahora, nombres de banda incompatibles con Landsat-7
        if not collections:
            collections = ["sentinel-2-l2a"]  # siempre Sentinel-2 por defecto

        bbox       = location["bbox"]
        start_date = date_range["start"]
        end_date   = date_range["end"]
        cloud_max  = plan.get("cloud_cover_max", 20)

        # Para consultas de inundación/agua, usar cloud_max mínimo de 60%
        # (los eventos de inundación ocurren con nubes y lluvia)
        WATER_INDICES = {"NDWI", "MNDWI", "AWEI", "NDWI2", "WRI"}
        is_flood_query = bool(WATER_INDICES.intersection({i.upper() for i in indices}))
        if is_flood_query and cloud_max < 60:
            logger.info(
                "[DataAcquisition] Flood query: bumping cloud_max %d%% → 60%% to find storm-era scenes",
                cloud_max,
            )
            cloud_max = 60

        # Para inundaciones, ampliar la VENTANA DE BÚSQUEDA 15 días más allá del
        # plan (sin tocar el plan): si el LLM eligió un subrango pre-evento
        # (no puede conocer eventos posteriores a su entrenamiento), las escenas
        # post-evento siguen entrando en la búsqueda y la selección por anomalía
        # de agua puede encontrarlas.
        search_end = end_date
        search_limit = 10
        if is_flood_query:
            try:
                end_dt_ = datetime.strptime(end_date, "%Y-%m-%d")
                search_end = min(
                    end_dt_ + timedelta(days=15), datetime.utcnow()
                ).strftime("%Y-%m-%d")
            except ValueError:
                pass
            search_limit = 25  # más escenas por tile → mejor anomalía de agua

        # 1. Buscar en Planetary Computer
        logger.info("[DataAcquisition] Searching Planetary Computer...")
        results = search_planetary_computer.func(
            bbox=bbox,
            start_date=start_date,
            end_date=search_end,
            collections=collections,
            cloud_cover_max=cloud_max,
            limit=search_limit,
        )
        scenes = results.get("items", [])

        # 2. Fallback a Earth Search si hay pocos resultados
        if len(scenes) < 3:
            logger.info("[DataAcquisition] Trying Earth Search fallback...")
            results2 = search_earth_search.func(
                bbox=bbox,
                start_date=start_date,
                end_date=search_end,
                collections=collections,
                cloud_cover_max=cloud_max,
                limit=search_limit,
            )
            scenes += results2.get("items", [])

        # 3. Si no hay escenas con cloud_max bajo, reintenta con 80% (eventos de lluvia/inundación)
        if not scenes and cloud_max < 80:
            logger.warning(
                "[DataAcquisition] No scenes with cloud_max=%d%%. Retrying with 80%% (flood/storm event).",
                cloud_max,
            )
            for retry_source, retry_fn in [
                ("Planetary Computer", search_planetary_computer.func),
                ("Earth Search",       search_earth_search.func),
            ]:
                retry_results = retry_fn(
                    bbox=bbox,
                    start_date=start_date,
                    end_date=search_end,
                    collections=collections,
                    cloud_cover_max=80,
                    limit=search_limit,
                )
                scenes += retry_results.get("items", [])
                if scenes:
                    logger.info("[DataAcquisition] %s encontró %d escenas con cloud_max=80%%",
                                retry_source, len(scenes))
                    break

        if not scenes:
            return {
                "available_scenes": [],
                "selected_scene": None,
                "downloaded_files": [],
                "acquisition_error": "No scenes found for the given parameters.",
                "current_agent": "analyst",
                "messages": [{"agent": "data_acquisition", "type": "warning",
                            "content": "No scenes found, analyst will use demo data."}],
            }

        # 3. Seleccionar mejor escena
        # Para inundaciones: primero intentar la escena con mayor ANOMALÍA de
        # porcentaje de agua dentro de su tile (detecta la escena del evento sin
        # depender de que el LLM conozca la fecha exacta). Si no hay metadatos
        # suficientes, caer a la MÁS TEMPRANA de la ventana (cercana a la
        # tormenta; el agua superficial drena en pocos días).
        selected = None
        scene_preference = "lowest_cloud"
        if is_flood_query:
            selected = _select_flood_scene_by_water_anomaly(scenes)
            scene_preference = "water_anomaly" if selected else "earliest"
        if selected is None:
            selected = select_best_scene.func(scenes=scenes, preference=scene_preference)
        logger.info(
            "[DataAcquisition] Selected scene: %s  preference=%s  cloud=%.1f%%  date=%s  water%%=%s",
            selected.get("id"), scene_preference,
            selected.get("cloud_cover", -1), selected.get("date", ""),
            selected.get("water_percentage"),
        )

        # 4. Determinar bandas necesarias para los índices
        band_map = {
            "NDVI": ["B04", "B08"], "EVI": ["B02", "B04", "B08"],
            "NDWI": ["B03", "B08"], "MNDWI": ["B03", "B11"],
            "AWEI": ["B02", "B03", "B08", "B11", "B12"],
            "NDBI": ["B08", "B11"], "NBR":   ["B08", "B12"],
            "NDSI": ["B03", "B11"], "SAVI":  ["B04", "B08"],
        }
        bands_needed = set()
        for idx in indices:
            bands_needed.update(band_map.get(idx.upper(), ["B04", "B08"]))

        # Descargar también SCL (Scene Classification Layer) SIEMPRE — permite
        # enmascarar nubes/sombras antes de calcular CUALQUIER índice, no solo
        # los de agua (las nubes/sombras distorsionan NDVI, NBR, NDSI, etc. igual).
        bands_needed.add("SCL")

        # 5. Descargar bandas
        logger.info("[DataAcquisition] Downloading bands: %s", bands_needed)
        downloaded = download_scene_bands.func(
            scene=selected,
            bands=list(bands_needed),
        )

        files = [v for v in downloaded.values() if not v.startswith("ERROR")]

        # ── Sentinel-1 SAR (solo inundación — es una técnica específica de agua) ──
        pre_scene_files: list[str] = []
        sar_available: bool = False
        sar_note: str = "no solicitado"
        pre_optical_files: list[str] = []

        # La fecha "evento" de referencia es la de la ESCENA seleccionada (que la
        # selección por anomalía de agua sitúa justo tras la inundación), no el
        # inicio del plan: si el LLM eligió un subrango equivocado, anclar el SAR
        # y el pre-óptico al plan compararía dos fechas sin evento entre medias.
        scene_date_str = (selected.get("date") or start_date)[:10]

        if state.get("analysis_type") == "flood":
            logger.info("[DataAcquisition] Flood query — iniciando adquisición Sentinel-1 SAR")
            try:
                scene_dt = datetime.strptime(scene_date_str, "%Y-%m-%d")
                sar_post_end = min(scene_dt + timedelta(days=7), datetime.utcnow()).strftime("%Y-%m-%d")
            except ValueError:
                sar_post_end = end_date
            sar_result = _acquire_sentinel1_sar(
                bbox=bbox,
                event_date_str=scene_date_str,
                post_end_str=sar_post_end,
            )
            pre_scene_files = sar_result["pre_files"]
            sar_available   = sar_result["sar_available"]
            sar_note        = sar_result["note"]
            files.extend(sar_result["post_files"])

        # ── Escena óptica pre-evento (para cambio real, no baseline asumido) ──
        # Válido para CUALQUIER análisis con una fecha concreta, no solo inundación
        # (deforestación, crecimiento urbano, deshielo, cicatriz de incendio...).
        # Se omite solo para "general", donde no hay un evento claro que comparar.
        if state.get("analysis_type") not in (None, "general"):
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt   = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                start_dt = end_dt = None

            pre_target_dt = None  # objetivo estacional para elegir la escena pre (solo interanual)
            if start_dt is None:
                pre_start = pre_end = None
            elif state.get("analysis_type") == "flood":
                # Comparar contra una escena despejada de las semanas justo antes
                # del evento, anclado a la fecha de la ESCENA seleccionada (la
                # selección por anomalía la sitúa justo tras la inundación).
                try:
                    anchor_dt = datetime.strptime(scene_date_str, "%Y-%m-%d")
                except ValueError:
                    anchor_dt = start_dt
                pre_start = (anchor_dt - timedelta(days=40)).strftime("%Y-%m-%d")
                pre_end   = (anchor_dt - timedelta(days=8)).strftime("%Y-%m-%d")
            else:
                # Sin evento puntual (vegetación, urbano, deshielo, cicatriz…):
                # comparar contra el MISMO momento del año anterior para controlar
                # la estacionalidad. Se ancla a la fecha de la ESCENA post (no al
                # rango de la consulta) con margen estrecho (±30 días), y luego la
                # escena pre se elige por PROXIMIDAD estacional (ver target_date),
                # no solo por nubosidad. Sin este anclaje, una ventana ancha (toda
                # la temporada anterior) elegía la más despejada aunque cayera en
                # otra fase fenológica — p.ej. octubre-2023 ya reverdecido tras las
                # lluvias frente a un septiembre-2024 seco — recreando el artefacto
                # estacional que precisamente se quería eliminar.
                try:
                    anchor_dt = datetime.strptime(scene_date_str, "%Y-%m-%d")
                except (ValueError, TypeError):
                    anchor_dt = end_dt
                pre_target_dt = anchor_dt - timedelta(days=365)
                pre_start = (pre_target_dt - timedelta(days=30)).strftime("%Y-%m-%d")
                pre_end   = (pre_target_dt + timedelta(days=30)).strftime("%Y-%m-%d")

            if pre_start and pre_end:
                pre_bands_needed = bands_needed & {"B02", "B03", "B04", "B08", "B11", "B12"}
                pre_bands_needed.add("SCL")
                logger.info(
                    "[DataAcquisition] Adquiriendo escena óptica pre-evento (cambio real): "
                    "ventana=%s→%s bandas=%s", pre_start, pre_end, pre_bands_needed,
                )
                pre_optical_result = _acquire_pre_event_optical(
                    bbox=bbox, pre_start=pre_start, pre_end=pre_end, bands=list(pre_bands_needed),
                    preferred_tile=_extract_mgrs_tile(selected.get("id", "")),
                    target_date=pre_target_dt,
                )
                pre_optical_files = pre_optical_result["pre_optical_files"]
                if pre_optical_result["pre_optical_available"]:
                    sar_note += f" | {pre_optical_result['note']}"

        return {
            "available_scenes":   scenes,
            "selected_scene":     selected,
            "downloaded_files":   files,
            "pre_scene_files":    pre_scene_files,
            "sar_available":      sar_available,
            "pre_optical_files":  pre_optical_files,
            "current_agent":      "analyst",
            "messages": [{"agent": "data_acquisition", "type": "status",
                        "content": (
                            f"Found {len(scenes)} scenes. Downloaded {len(files)} band files. "
                            f"SAR: {sar_note}"
                        )}],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Sentinel-1 SAR acquisition helper
# ─────────────────────────────────────────────────────────────────────────────

def _acquire_sentinel1_sar(
    bbox: list[float],
    event_date_str: str,
    post_end_str: str,
) -> dict:
    """
    Busca y descarga la banda VV de Sentinel-1 GRD para periodos pre y post evento.
    El radar SAR ve a través de nubes — ideal para inundaciones con tormenta.

    Returns dict con claves: pre_files, post_files, sar_available, note.
    """
    result: dict = {"pre_files": [], "post_files": [], "sar_available": False, "note": ""}

    try:
        event_dt = datetime.strptime(event_date_str, "%Y-%m-%d")
    except ValueError:
        result["note"] = f"Fecha de evento inválida: {event_date_str}"
        return result

    pre_start = (event_dt - timedelta(days=28)).strftime("%Y-%m-%d")
    pre_end   = (event_dt - timedelta(days=5)).strftime("%Y-%m-%d")

    PC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

    def _search_s1(start: str, end: str) -> list[dict]:
        payload = {
            "collections": ["sentinel-1-grd"],
            "bbox": bbox,
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "limit": 5,
            "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        }
        try:
            resp = httpx.post(PC_URL, json=payload, timeout=30)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            return [
                {
                    "id": f["id"],
                    "date": f["properties"].get("datetime", ""),
                    "assets": {k: v.get("href", "") for k, v in f.get("assets", {}).items() if "href" in v},
                    "source": "planetary_computer",
                }
                for f in features
            ]
        except Exception as exc:
            logger.warning("[SAR] Búsqueda S1 fallida (%s→%s): %s", start, end, exc)
            return []

    def _download_vv(scene: dict, prefix: str) -> Optional[str]:
        assets = scene.get("assets", {})
        url = assets.get("vv") or assets.get("VV")
        if not url:
            logger.warning("[SAR] Sin asset VV en escena %s", scene.get("id"))
            return None
        try:
            import planetary_computer as pc
            url = pc.sign(url)
        except Exception as e:
            logger.warning("[SAR] No se pudo firmar URL S1: %s", e)
        local_path = DATA_DIR / f"{scene['id']}_{prefix}_VV.tif"
        if local_path.exists():
            logger.info("[SAR] Cacheado: %s", local_path)
            return str(local_path)
        try:
            with httpx.stream("GET", url, timeout=300, follow_redirects=True) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1024 * 256):
                        f.write(chunk)
            logger.info("[SAR] Descargado %s VV → %s", prefix, local_path)
            return str(local_path)
        except Exception as exc:
            logger.warning("[SAR] Descarga fallida (%s): %s", scene.get("id"), exc)
            return None

    # Pre-evento
    pre_scenes = _search_s1(pre_start, pre_end)
    if not pre_scenes:
        result["note"] = f"Sin escenas S1 pre-evento ({pre_start}→{pre_end})"
        logger.warning("[SAR] %s", result["note"])
        return result
    pre_path = _download_vv(pre_scenes[0], "pre")
    if not pre_path:
        result["note"] = "Descarga VV pre-evento fallida"
        return result
    result["pre_files"].append(pre_path)

    # Post-evento: la MÁS TEMPRANA de la ventana (la búsqueda ordena descendente,
    # así que es el último elemento). Es la más cercana al pico de la inundación;
    # la más tardía captaría el agua ya drenada.
    post_scenes = _search_s1(event_date_str, post_end_str)
    if not post_scenes:
        result["note"] = f"Sin escenas S1 post-evento ({event_date_str}→{post_end_str})"
        logger.warning("[SAR] %s", result["note"])
        return result
    post_path = _download_vv(post_scenes[-1], "post")
    if not post_path:
        result["note"] = "Descarga VV post-evento fallida"
        return result
    result["post_files"].append(post_path)

    result["sar_available"] = True
    result["note"] = (
        f"pre: {pre_scenes[0]['id'][:30]}… ({pre_scenes[0]['date'][:10]}), "
        f"post: {post_scenes[0]['id'][:30]}… ({post_scenes[0]['date'][:10]})"
    )
    logger.info("[SAR] Adquisición completa — %s", result["note"])
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Sentinel-2 óptico pre-evento (para cambio NDWI real, sin baseline asumido)
# ─────────────────────────────────────────────────────────────────────────────

def _acquire_pre_event_optical(
    bbox: list[float],
    pre_start: str,
    pre_end: str,
    bands: list[str],
    preferred_tile: Optional[str] = None,
    target_date: Optional[datetime] = None,
) -> dict:
    """
    Busca y descarga las bandas indicadas (las que necesiten los índices
    solicitados, más SCL) de la escena Sentinel-2 dentro de la ventana
    [pre_start, pre_end], para poder calcular un cambio real pre/post en vez de
    comparar la media del área contra un baseline asumido.

    La elección de esa ventana (corta antes de un evento puntual, o alrededor
    del mismo momento del año anterior para controlar estacionalidad) se decide
    en el llamador, que es quien conoce el analysis_type — ver __call__.

    Si se indica target_date, la escena se elige por PROXIMIDAD de calendario a
    esa fecha (misma fase fenológica que la post, usando la nubosidad solo como
    desempate) en vez de por menor nubosidad a secas. Esto evita que, en una
    comparación interanual, se elija una escena de otra fase estacional solo por
    estar más despejada. Sin target_date se mantiene el criterio de menor nube.

    Si se indica preferred_tile, se EXIGE que la escena pre-evento pertenezca
    al mismo tile MGRS que la escena post-evento. Regiones grandes (p.ej.
    Galicia) pueden abarcar varios tiles Sentinel-2; sin esta restricción, la
    búsqueda podría devolver una escena de un tile distinto (simplemente por
    tener menos nubes en esa ventana), y comparar píxel a píxel dos escenas de
    tiles distintos compara zonas geográficas que no tienen relación entre sí,
    aunque los arrays resultantes tengan las mismas dimensiones.

    Returns dict con claves: pre_optical_files, pre_optical_available, note.
    """
    result: dict = {"pre_optical_files": [], "pre_optical_available": False, "note": ""}

    # limit alto: la búsqueda pagina por fecha, y con un límite bajo una ventana
    # amplia sobre varios tiles solo devuelve las fechas más tardías — dejaría
    # fuera la escena estacionalmente más cercana (la que luego elige target_date)
    # y forzaría a coger una de otra fase (p.ej. el rebrote de finales de mes).
    search = search_planetary_computer.func(
        bbox=bbox,
        start_date=pre_start,
        end_date=pre_end,
        collections=["sentinel-2-l2a"],
        cloud_cover_max=20,
        limit=200,
    )
    scenes = search.get("items", [])
    if not scenes:
        result["note"] = f"Sin escenas S2 pre-evento despejadas ({pre_start}→{pre_end})"
        logger.warning("[PreOptical] %s", result["note"])
        return result

    if preferred_tile:
        same_tile = [s for s in scenes if _extract_mgrs_tile(s.get("id", "")) == preferred_tile]
        if not same_tile:
            result["note"] = (
                f"Sin escena pre-evento en el mismo tile ({preferred_tile}) dentro de "
                f"{pre_start}→{pre_end} — se omite la comparación para no comparar zonas distintas"
            )
            logger.warning("[PreOptical] %s", result["note"])
            return result
        scenes = same_tile

    if target_date is not None:
        # Elegir la escena pre más cercana en el calendario a target_date (misma
        # fase fenológica que la post); la nubosidad solo desempata. Así una
        # comparación interanual no acaba mezclando estaciones (ver docstring).
        def _season_gap(s: dict):
            d = (s.get("date", "") or "")[:10]
            try:
                gap = abs((datetime.strptime(d, "%Y-%m-%d") - target_date).days)
            except ValueError:
                gap = 10 ** 6
            return (gap, s.get("cloud_cover", 100))

        best = sorted(scenes, key=_season_gap)[0]
        logger.info(
            "[PreOptical] Escena pre por proximidad estacional: %s (objetivo %s, cloud=%.1f%%)",
            (best.get("date", "") or "")[:10],
            target_date.strftime("%Y-%m-%d"),
            best.get("cloud_cover", -1),
        )
    else:
        best = select_best_scene.func(scenes=scenes, preference="lowest_cloud")
    downloaded = download_scene_bands.func(scene=best, bands=bands)
    files = [v for v in downloaded.values() if not v.startswith("ERROR")]

    if not files:
        result["note"] = "Descarga de bandas ópticas pre-evento fallida"
        logger.warning("[PreOptical] %s", result["note"])
        return result

    result["pre_optical_files"] = files
    result["pre_optical_available"] = True
    result["note"] = (
        f"pre-óptico: {best['id'][:30]}… "
        f"({best.get('date', '')[:10]}, nubes={best.get('cloud_cover', '?')}%)"
    )
    logger.info("[PreOptical] Adquisición completa — %s", result["note"])
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────
def _harvest(content, scenes: list, files: list) -> None:
    """Best-effort extraction of scene/file data from tool output strings."""
    import json as _json
    if isinstance(content, str):
        try:
            data = _json.loads(content)
            if isinstance(data, dict):
                if "id" in data and "assets" in data:
                    scenes.append(data)
                if "items" in data:
                    scenes.extend(data["items"])
                # downloaded files map band → path
                for v in data.values():
                    if isinstance(v, str) and v.endswith(".tif"):
                        files.append(v)
        except Exception:
            pass