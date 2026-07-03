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
        preference: "lowest_cloud" | "most_recent" | "best_coverage"

    Returns:
        The selected scene dict.
    """
    if not scenes:
        return {"error": "No scenes available to select from."}

    if preference == "most_recent":
        scenes_sorted = sorted(
            scenes, key=lambda s: s.get("date", ""), reverse=True
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

        # 1. Buscar en Planetary Computer
        logger.info("[DataAcquisition] Searching Planetary Computer...")
        results = search_planetary_computer.func(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            collections=collections,
            cloud_cover_max=cloud_max,
            limit=10,
        )
        scenes = results.get("items", [])

        # 2. Fallback a Earth Search si hay pocos resultados
        if len(scenes) < 3:
            logger.info("[DataAcquisition] Trying Earth Search fallback...")
            results2 = search_earth_search.func(
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                collections=collections,
                cloud_cover_max=cloud_max,
                limit=10,
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
                    end_date=end_date,
                    collections=collections,
                    cloud_cover_max=80,
                    limit=10,
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
        # Para índices de agua/inundación preferir la escena más reciente (captura el evento)
        WATER_INDICES = {"NDWI", "MNDWI", "AWEI", "NDWI2", "WRI", "NDSI"}
        is_flood_query = bool(WATER_INDICES.intersection({i.upper() for i in indices}))
        scene_preference = "most_recent" if is_flood_query else "lowest_cloud"
        selected = select_best_scene.func(scenes=scenes, preference=scene_preference)
        logger.info(
            "[DataAcquisition] Selected scene: %s  preference=%s  cloud=%.1f%%  date=%s",
            selected.get("id"), scene_preference,
            selected.get("cloud_cover", -1), selected.get("date", ""),
        )

        # 4. Determinar bandas necesarias para los índices
        band_map = {
            "NDVI": ["B04", "B08"], "EVI": ["B02", "B04", "B08"],
            "NDWI": ["B03", "B08"], "MNDWI": ["B03", "B11"],
            "NDBI": ["B08", "B11"], "NBR":   ["B08", "B12"],
            "NDSI": ["B03", "B11"], "SAVI":  ["B04", "B08"],
        }
        bands_needed = set()
        for idx in indices:
            bands_needed.update(band_map.get(idx.upper(), ["B04", "B08"]))

        # 5. Descargar bandas
        logger.info("[DataAcquisition] Downloading bands: %s", bands_needed)
        downloaded = download_scene_bands.func(
            scene=selected,
            bands=list(bands_needed),
        )

        files = [v for v in downloaded.values() if not v.startswith("ERROR")]

        # ── Sentinel-1 SAR (solo para análisis de inundación) ─────────────
        pre_scene_files: list[str] = []
        sar_available: bool = False
        sar_note: str = "no solicitado"

        if state.get("analysis_type") == "flood":
            logger.info("[DataAcquisition] Flood query — iniciando adquisición Sentinel-1 SAR")
            sar_result = _acquire_sentinel1_sar(
                bbox=bbox,
                event_date_str=start_date,
                post_end_str=end_date,
            )
            pre_scene_files = sar_result["pre_files"]
            sar_available   = sar_result["sar_available"]
            sar_note        = sar_result["note"]
            files.extend(sar_result["post_files"])

        return {
            "available_scenes":  scenes,
            "selected_scene":    selected,
            "downloaded_files":  files,
            "pre_scene_files":   pre_scene_files,
            "sar_available":     sar_available,
            "current_agent":     "analyst",
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

    # Post-evento
    post_scenes = _search_s1(event_date_str, post_end_str)
    if not post_scenes:
        result["note"] = f"Sin escenas S1 post-evento ({event_date_str}→{post_end_str})"
        logger.warning("[SAR] %s", result["note"])
        return result
    post_path = _download_vv(post_scenes[0], "post")
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