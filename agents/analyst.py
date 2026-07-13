"""
agents/analyst.py – Geospatial Analyst Agent (corregido)

Cambios principales:
- Prompt más estricto: exige siempre datos sintéticos si no hay archivos reales
- _extract_code mejorado: maneja bloques con/sin lenguaje, strips de texto extra
- Timeout en ejecución para no colgar el pipeline
- Mejor log de errores para diagnóstico
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import signal
import textwrap
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from state import GeoAgentState

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "./outputs"))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_CODE_ITERATIONS = 3

# ─────────────────────────────────────────────────────────────────────────────
#  System prompt — muy explícito para evitar código que falle
# ─────────────────────────────────────────────────────────────────────────────

ANALYST_SYSTEM = """\
You are the Geospatial Analyst Agent of a multi-agent Earth Observation system.
Your ONLY output must be a single Python code block: ```python ... ```
Do NOT write any explanation, markdown, or text outside the code block.

## Available libraries
numpy, rasterio, matplotlib, PIL (Pillow), scipy, pandas, pathlib, json, os

## CRITICAL RULES (follow every single one)
1. ALWAYS start with: import numpy as np, import matplotlib, matplotlib.use('Agg'), import matplotlib.pyplot as plt, import rasterio, import json, import os
2. Define OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs") at the top.

3. READING REAL SENTINEL-2 BAND FILES (when band_files is not empty):
   - Sentinel-2 L2A bands are uint16 with DN values 0-10000 (surface reflectance × 10000).
   - Fill/nodata pixels have value 0. You MUST mask them as NaN.
   - EXACT reading pattern — copy verbatim, do NOT modify:
     ```python
     import rasterio
     from rasterio.enums import Resampling
     TARGET = 512
     with rasterio.open(path) as src:
         h = min(TARGET, src.height)
         w = min(TARGET, src.width)
         arr = src.read(1, out_shape=(h, w), resampling=Resampling.average).astype(np.float32)
     arr[arr <= 0] = np.nan
     arr = arr / 10000.0
     if np.nanmin(arr) > 0.04:  # BOA offset (processing baseline >= 04.00, scenes from 2022+)
         arr = np.clip(arr - 0.1, 0.0001, None)
     ```
   - IMPORTANT: `src.read(1, out_shape=(h, w))` uses a 2D out_shape when reading a single band by scalar index. Never use (1, h, w) with a scalar band index.
   - All file reading MUST happen INSIDE the `with rasterio.open(...) as src:` block.
   - Detect band names from filenames: _B04.tif → RED, _B08.tif → NIR, _B02.tif → BLUE, _B03.tif → GREEN, _B11.tif → SWIR1, _B12.tif → SWIR2.

4. If band_files list is empty OR all files are missing/unreadable:
   - Generate SYNTHETIC reflectance data with REALISTIC per-band values (do NOT use the same range for all bands):
     np.random.seed(42)
     NIR   = np.random.uniform(0.30, 0.60, (512, 512)).astype(np.float32)  # high in vegetation
     RED   = np.random.uniform(0.03, 0.15, (512, 512)).astype(np.float32)  # low (absorbed by chlorophyll)
     GREEN = np.random.uniform(0.05, 0.18, (512, 512)).astype(np.float32)
     BLUE  = np.random.uniform(0.02, 0.10, (512, 512)).astype(np.float32)
     SWIR1 = np.random.uniform(0.08, 0.25, (512, 512)).astype(np.float32)
     SWIR2 = np.random.uniform(0.05, 0.18, (512, 512)).astype(np.float32)
   - Do NOT raise errors.

5. For each requested index:
   a. Compute using float32 arrays (NaN-safe: use np.nanmean, np.nanstd, np.nanpercentile)
   b. Clip to valid range
   c. Save colour-mapped PNG: plt.imsave(f"{OUTPUT_DIR}/{index_name.lower()}.png", valid_array, cmap='RdYlGn', vmin=-1, vmax=1)
      where valid_array = np.nan_to_num(index_array, nan=0.0)
   d. Compute stats over VALID (non-NaN) pixels only:
      valid = index_array[~np.isnan(index_array)]
      stats = {"mean": float(np.mean(valid)), "std": float(np.std(valid)),
               "min": float(np.min(valid)), "max": float(np.max(valid)),
               "p25": float(np.percentile(valid, 25)), "p50": float(np.percentile(valid, 50)),
               "p75": float(np.percentile(valid, 75))}

6. Save a summary figure (all indices as subplots) to: {OUTPUT_DIR}/summary.png
   Use plt.savefig(..., dpi=100, bbox_inches='tight'). NEVER call plt.show().

7. At the very END print a single JSON line (no extra text):
   {"status": "success", "indices": {"NDVI": {"mean": 0.35, "std": 0.18, "min": -0.05, "max": 0.82, "p25": 0.22, "p50": 0.34, "p75": 0.51}}, "output_files": ["./outputs/ndvi.png"]}

8. Wrap ALL file operations in try/except. Never crash.
9. Do NOT use plt.show(), display(), or any interactive function.

## Index formulas (inputs are float32 reflectance 0.0–1.0, NaN where nodata)
NDVI  = (NIR - RED) / (NIR + RED + 1e-10)                        → clip [-1, 1]
EVI   = 2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1+1e-10)               → clip [-1, 1]
SAVI  = 1.5*(NIR-RED)/(NIR+RED+0.5)                               → clip [-1, 1]
NDWI  = (GREEN - NIR) / (GREEN + NIR + 1e-10)                     → clip [-1, 1]
MNDWI = (GREEN - SWIR1) / (GREEN + SWIR1 + 1e-10)                 → clip [-1, 1]
AWEI  = 4*(GREEN - SWIR1) - (0.25*NIR + 2.75*SWIR2)               → clip [-1, 1]  (AWEInsh, no-shadow variant)
NDBI  = (SWIR1 - NIR) / (SWIR1 + NIR + 1e-10)                     → clip [-1, 1]
NBR   = (NIR - SWIR2) / (NIR + SWIR2 + 1e-10)                     → clip [-1, 1]
LAI   = 3.618 * EVI - 0.118                                        → clip [0, 8]

## Band → file mapping (detect from filename suffix)
_B02.tif → BLUE   _B03.tif → GREEN  _B04.tif → RED
_B08.tif → NIR    _B11.tif → SWIR1  _B12.tif → SWIR2

Respond in Spanish in comments only. Code must be valid Python 3.10+.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Ejecutor con timeout
# ─────────────────────────────────────────────────────────────────────────────

def _execute_code(code: str) -> dict:
    """
    Ejecuta código Python en un namespace aislado.
    Captura stdout/stderr y cualquier excepción.
    Timeout de 120 segundos vía SIGALRM (solo Unix).
    """
    namespace: dict = {
        "__builtins__": __builtins__,
        "OUTPUT_DIR": str(OUTPUTS_DIR),
    }
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    error: Optional[str] = None

    # Timeout handler (Unix only — en Windows se ignora)
    def _timeout(signum, frame):
        raise TimeoutError("Ejecución superó 120 segundos")

    try:
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(120)
    except (AttributeError, OSError):
        pass  # Windows: SIGALRM no disponible

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compile(code, "<analyst>", "exec"), namespace)  # noqa: S102
    except Exception:
        error = traceback.format_exc()
    finally:
        try:
            signal.alarm(0)
        except (AttributeError, OSError):
            pass

    return {
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "error": error,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Extractor de código — robusto
# ─────────────────────────────────────────────────────────────────────────────

def _extract_code(text: str) -> Optional[str]:
    """
    Extrae el primer bloque de código Python de la respuesta del LLM.
    Maneja: ```python, ```py, ``` (sin lenguaje), y código plano.
    """
    # 1. Bloque con etiqueta de lenguaje
    m = re.search(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2. Bloque genérico (solo ```)
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        # Verificar que parezca código Python
        if any(kw in candidate for kw in ["import", "def ", "np.", "plt.", "="]):
            return candidate

    # 3. Respuesta completa que empieza como código
    stripped = text.strip()
    if stripped.startswith(("import ", "# ", "from ", "numpy", "OUTPUT_DIR")):
        return stripped

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Parser del JSON de resumen
# ─────────────────────────────────────────────────────────────────────────────

def _parse_summary(stdout: str) -> tuple[dict, list[str]]:
    """Extrae índices computados y archivos de salida del stdout del script."""
    computed: dict = {}
    files: list[str] = []

    # Buscar objetos JSON completos en el stdout
    depth = 0
    start = None
    for i, ch in enumerate(stdout):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = stdout[start : i + 1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict) and data.get("status") == "success":
                        for idx_name, stats in data.get("indices", {}).items():
                            flat = {}
                            for k, v in (stats or {}).items():
                                if isinstance(v, dict):
                                    flat.update({pk: round(float(pv), 4) for pk, pv in v.items()})
                                elif isinstance(v, (int, float)):
                                    flat[k] = round(float(v), 4)
                            computed[idx_name] = flat
                        files = data.get("output_files", [])
                        break
                except (json.JSONDecodeError, ValueError):
                    pass

    # Recoger PNGs existentes en disco como fallback
    for match in re.findall(r"[\w./\\-]+\.(?:png|tif|tiff)", stdout):
        if match not in files and Path(match).exists():
            files.append(match)

    return computed, files


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback Python directo — no depende del código generado por el LLM
# ─────────────────────────────────────────────────────────────────────────────

DEBUG_IMAGES_DIR = OUTPUTS_DIR / "debug_images"
DEBUG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _save_rgb_composite(bands: dict, is_synthetic: bool, scene_info: dict | None = None) -> None:
    """Guarda imágenes de debug (true color, false color, NDWI) e info.txt."""
    try:
        import numpy as np
        from PIL import Image
        import matplotlib

        def _to_uint8(arr: "np.ndarray", lo: float = 0.02, hi: float = 0.25) -> "np.ndarray":
            clipped = np.clip(arr, lo, hi)
            return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)

        # True color: R=RED, G=GREEN, B=BLUE
        if all(b in bands for b in ("RED", "GREEN", "BLUE")):
            r = _to_uint8(np.nan_to_num(bands["RED"],   nan=0.0))
            g = _to_uint8(np.nan_to_num(bands["GREEN"], nan=0.0))
            b = _to_uint8(np.nan_to_num(bands["BLUE"],  nan=0.0))
            Image.fromarray(np.stack([r, g, b], axis=-1), "RGB").save(
                str(DEBUG_IMAGES_DIR / "true_color.png"))
            logger.info("[Debug] true_color.png guardado")

        # False color: R=NIR, G=RED, B=GREEN (vegetación=rojo, agua=negro)
        if all(b in bands for b in ("NIR", "RED", "GREEN")):
            r = _to_uint8(np.nan_to_num(bands["NIR"],   nan=0.0), lo=0.05, hi=0.60)
            g = _to_uint8(np.nan_to_num(bands["RED"],   nan=0.0))
            b = _to_uint8(np.nan_to_num(bands["GREEN"], nan=0.0))
            Image.fromarray(np.stack([r, g, b], axis=-1), "RGB").save(
                str(DEBUG_IMAGES_DIR / "false_color_nir.png"))
            logger.info("[Debug] false_color_nir.png guardado")

        # NDWI (azul > 0.2 = agua / inundación)
        if all(b in bands for b in ("GREEN", "NIR")):
            ndwi = (bands["GREEN"] - bands["NIR"]) / (bands["GREEN"] + bands["NIR"] + 1e-10)
            ndwi = np.clip(ndwi, -1, 1)
            norm = (np.nan_to_num(ndwi, nan=0.0) + 1) / 2.0
            rgba = (matplotlib.colormaps["RdYlBu"](norm) * 255).astype(np.uint8)
            Image.fromarray(rgba, "RGBA").save(str(DEBUG_IMAGES_DIR / "ndwi_flood.png"))
            water_frac = float(np.mean(ndwi[~np.isnan(ndwi)] > 0.2) * 100)
            logger.info("[Debug] ndwi_flood.png — %.1f%% píxeles agua (NDWI>0.2)", water_frac)

        # info.txt con metadatos de escena y estadísticas
        meta_lines = [
            f"Datos: {'SINTÉTICOS (fallback)' if is_synthetic else 'REALES (Sentinel-2)'}",
        ]
        if scene_info:
            meta_lines += [
                f"Escena ID: {scene_info.get('id', '?')}",
                f"Fecha adquisición: {scene_info.get('date', '?')}",
                f"Cobertura nubes: {scene_info.get('cloud_cover', '?')}%",
                f"Fuente: {scene_info.get('source', '?')}",
            ]
        meta_lines += [
            f"Bandas disponibles: {list(bands.keys())}",
            f"Resolución: {next(iter(bands.values())).shape}",
        ]
        if "NIR" in bands and "GREEN" in bands:
            ndwi_v = (bands["GREEN"] - bands["NIR"]) / (bands["GREEN"] + bands["NIR"] + 1e-10)
            valid = ndwi_v[~np.isnan(ndwi_v)]
            meta_lines += [
                f"NDWI medio: {float(np.mean(valid)):.4f}",
                f"% píxeles agua (NDWI>0.2): {float(np.mean(valid > 0.2)*100):.1f}%",
            ]
        (DEBUG_IMAGES_DIR / "info.txt").write_text("\n".join(meta_lines), encoding="utf-8")

    except Exception as exc:
        logger.warning("[Debug] Error guardando imágenes de depuración: %s", exc)


def _python_fallback_compute(
    downloaded_files: list,
    required_indices: list,
    location: Optional[dict],
) -> tuple[dict, list]:
    """
    Computa índices geoespaciales directamente en Python sin LLM.
    Usa PIL directamente para guardar PNGs, evitando bugs de plt.imsave.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.enums import Resampling
        from PIL import Image
        import matplotlib
    except ImportError as e:
        logger.warning("[Fallback] Librerías no disponibles: %s", e)
        return {}, []

    TARGET = 512

    # ── Leer bandas desde TIF (lectura compartida, con corrección de offset BOA) ──
    bands: dict = _read_optical_bands(downloaded_files, target=TARGET)
    if bands:
        logger.info("[Fallback] Bandas leídas: %s", sorted(bands.keys()))

    is_synthetic = len(bands) == 0

    # ── Datos sintéticos realistas para bandas que falten ─────────────────
    rng = np.random.default_rng(42)
    H = W = TARGET
    defaults = {
        "NIR":   (0.30, 0.60), "RED":   (0.03, 0.15),
        "GREEN": (0.05, 0.18), "BLUE":  (0.02, 0.10),
        "SWIR1": (0.08, 0.25), "SWIR2": (0.05, 0.18),
    }
    for b, (lo, hi) in defaults.items():
        if b not in bands:
            bands[b] = rng.uniform(lo, hi, (H, W)).astype(np.float32)

    # ── Guardar imágenes de depuración siempre ────────────────────────────
    _save_rgb_composite(bands, is_synthetic=is_synthetic, scene_info=None)

    NIR, RED, GREEN = bands["NIR"], bands["RED"], bands["GREEN"]
    BLUE, SWIR1, SWIR2 = bands["BLUE"], bands["SWIR1"], bands["SWIR2"]

    INDEX_FORMULAS = {
        "NDVI":  (NIR - RED)   / (NIR + RED   + 1e-10),
        "EVI":   2.5*(NIR-RED) / (NIR + 6*RED - 7.5*BLUE + 1 + 1e-10),
        "SAVI":  1.5*(NIR-RED) / (NIR + RED   + 0.5),
        "NDWI":  (GREEN - NIR) / (GREEN + NIR  + 1e-10),
        "MNDWI": (GREEN - SWIR1) / (GREEN + SWIR1 + 1e-10),
        "AWEI":  4*(GREEN-SWIR1) - (0.25*NIR + 2.75*SWIR2),
        "NDBI":  (SWIR1 - NIR) / (SWIR1 + NIR  + 1e-10),
        "NBR":   (NIR - SWIR2) / (NIR + SWIR2 + 1e-10),
        "NDSI":  (GREEN - SWIR1) / (GREEN + SWIR1 + 1e-10),
    }
    CMAPS = {"NDWI": "RdYlBu", "NDBI": "RdYlBu"}

    computed: dict = {}
    files: list = []

    def _save_png(array: "np.ndarray", path: Path, cmap_name: str) -> bool:
        """Guarda array [-1,1] como PNG usando PIL (sin plt.imsave)."""
        try:
            norm = np.clip((array + 1) / 2.0, 0.0, 1.0).astype(np.float64)
            cmap_fn = matplotlib.colormaps[cmap_name]
            rgba = (cmap_fn(norm) * 255).astype(np.uint8)
            Image.fromarray(rgba, mode="RGBA").save(str(path))
            return True
        except Exception as exc:
            logger.warning("[Fallback] PIL save falló para %s: %s", path.name, exc)
            return False

    for idx_name in required_indices:
        key = idx_name.upper()
        raw = INDEX_FORMULAS.get(key)
        if raw is None:
            continue

        arr = np.clip(raw, -1, 1).astype(np.float32)
        display = np.nan_to_num(arr, nan=0.0)
        png_path = OUTPUTS_DIR / f"{key.lower()}.png"
        cmap_name = CMAPS.get(key, "RdYlGn")

        if _save_png(display, png_path, cmap_name):
            files.append(str(png_path))
            logger.info("[Fallback] PNG guardado: %s", png_path.name)

        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            continue
        computed[key] = {
            "mean": round(float(np.mean(valid)),           4),
            "std":  round(float(np.std(valid)),            4),
            "min":  round(float(np.min(valid)),            4),
            "max":  round(float(np.max(valid)),            4),
            "p25":  round(float(np.percentile(valid, 25)), 4),
            "p50":  round(float(np.percentile(valid, 50)), 4),
            "p75":  round(float(np.percentile(valid, 75)), 4),
        }

    return computed, files


# ─────────────────────────────────────────────────────────────────────────────
#  Sentinel-1 SAR change detection
# ─────────────────────────────────────────────────────────────────────────────

def _sar_flood_analysis(
    pre_files: list,
    post_files: list,
    outputs_dir: Path,
    bbox: Optional[list] = None,
) -> tuple[dict, list, list]:
    """
    Detección de inundaciones por cambio de retrodispersión SAR (Sentinel-1 GRD VV).

    Método:
      1. Reproyecta las bandas VV pre y post a una malla común EPSG:4326
         (los GRD vienen en geometría radar con GCPs, sin CRS: leerlos "tal
         cual" compara píxeles de sitios distintos si las órbitas difieren)
      2. Convierte a dB: σ° = 10 * log10(valor + ε)
      3. Imagen de cambio: Δσ° = post_dB - pre_dB
      4. Máscara: Δ < -3 dB = posible inundación, Δ < -5 dB = inundación confirmada

    Si se pasa `bbox` (área de estudio del plan), el análisis se recorta a esa
    zona (con margen), lo que además hace que los porcentajes se refieran al
    área de interés y no a toda la escena SAR (~250 km de swath).

    El SAR penetra nubes — detecta agua durante la tormenta misma.
    El agua tranquila refleja la señal lejos del sensor → muy baja retrodispersión.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
        from PIL import Image
        import matplotlib
    except ImportError as exc:
        logger.warning("[SAR] Dependencia no disponible: %s", exc)
        return {}, [], []

    if not pre_files or not post_files:
        logger.warning("[SAR] pre_files o post_files vacíos — saltando análisis SAR")
        return {}, [], []

    TARGET = 512

    def _open_warped(src):
        src_crs = src.crs
        if src_crs is None and src.gcps and src.gcps[1] is not None:
            src_crs = src.gcps[1]
        return WarpedVRT(src, src_crs=src_crs, crs="EPSG:4326",
                         resampling=Resampling.average)

    def _warped_extent(path: str) -> Optional[tuple]:
        try:
            with rasterio.open(path) as src, _open_warped(src) as vrt:
                return tuple(vrt.bounds)
        except Exception as exc:
            logger.warning("[SAR] No se pudo georreferenciar %s: %s", path, exc)
            return None

    def _read_vv_window(path: str, bounds: tuple) -> Optional["np.ndarray"]:
        try:
            with rasterio.open(path) as src, _open_warped(src) as vrt:
                window = vrt.window(*bounds)
                arr = vrt.read(1, window=window, out_shape=(TARGET, TARGET),
                               resampling=Resampling.average).astype(np.float32)
            arr[arr <= 0] = np.nan
            return arr
        except Exception as exc:
            logger.warning("[SAR] No se pudo leer %s: %s", path, exc)
            return None

    # Malla común: intersección de ambas escenas (y del bbox del plan con margen)
    pre_extent  = _warped_extent(pre_files[0])
    post_extent = _warped_extent(post_files[0])
    if pre_extent is None or post_extent is None:
        logger.warning("[SAR] Falta georreferenciación pre o post")
        return {}, [], []

    common = [max(pre_extent[0], post_extent[0]), max(pre_extent[1], post_extent[1]),
              min(pre_extent[2], post_extent[2]), min(pre_extent[3], post_extent[3])]
    if bbox and len(bbox) == 4:
        margin = 0.3  # grados de contexto alrededor del área de estudio
        focused = [max(common[0], bbox[0] - margin), max(common[1], bbox[1] - margin),
                   min(common[2], bbox[2] + margin), min(common[3], bbox[3] + margin)]
        if focused[0] < focused[2] and focused[1] < focused[3]:
            common = focused
    if not (common[0] < common[2] and common[1] < common[3]):
        logger.warning("[SAR] Las escenas pre y post no se solapan — se omite")
        return {}, [], []
    common_bounds = tuple(common)

    pre_vv  = _read_vv_window(pre_files[0], common_bounds)
    post_vv = _read_vv_window(post_files[0], common_bounds)

    if pre_vv is None or post_vv is None:
        logger.warning("[SAR] Falta pre o post VV")
        return {}, [], []

    # Conversión a dB
    eps = 1e-10
    pre_db  = 10.0 * np.log10(pre_vv  + eps)
    post_db = 10.0 * np.log10(post_vv + eps)
    change_db = post_db - pre_db

    nan_mask        = np.isnan(change_db)
    possible_flood  = (~nan_mask) & (change_db < -3.0)
    confirmed_flood = (~nan_mask) & (change_db < -5.0)
    total_valid     = int(np.sum(~nan_mask)) or 1

    pct_possible  = float(np.sum(possible_flood))  / total_valid * 100
    pct_confirmed = float(np.sum(confirmed_flood)) / total_valid * 100

    output_files: list = []

    # Imagen de cambio (colormap divergente: azul=caída, rojo=subida)
    change_path = outputs_dir / "sar_change.png"
    try:
        display = np.nan_to_num(change_db, nan=0.0)
        norm    = np.clip((display + 15.0) / 30.0, 0.0, 1.0)
        rgba    = (matplotlib.colormaps["RdBu"](norm) * 255).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(str(change_path))
        output_files.append(str(change_path))
        logger.info("[SAR] sar_change.png guardado")
    except Exception as exc:
        logger.warning("[SAR] No se pudo guardar sar_change.png: %s", exc)

    # Máscara categórica: gris=sin cambio, amarillo=posible, azul=confirmado
    mask_path = outputs_dir / "sar_flood_mask.png"
    try:
        mask_rgb = np.full((*change_db.shape, 3), 180, dtype=np.uint8)  # gris
        mask_rgb[possible_flood  & ~confirmed_flood] = [255, 200, 0]    # amarillo
        mask_rgb[confirmed_flood]                    = [0,   100, 200]   # azul
        mask_rgb[nan_mask]                           = [0,   0,   0]     # negro
        Image.fromarray(mask_rgb, "RGB").save(str(mask_path))
        output_files.append(str(mask_path))
        logger.info("[SAR] sar_flood_mask.png guardado — %.1f%% confirmado", pct_confirmed)
    except Exception as exc:
        logger.warning("[SAR] No se pudo guardar sar_flood_mask.png: %s", exc)

    # Copiar máscara a debug_images también
    try:
        from PIL import Image as _PIL
        _PIL.open(str(mask_path)).save(str(DEBUG_IMAGES_DIR / "sar_flood_mask.png"))
    except Exception:
        pass

    # Overlay transparente para el globo 3D: solo los píxeles inundados llevan color
    overlays: list = []
    if bool(np.any(possible_flood)):
        overlay_path = outputs_dir / "overlay_sar_flood.png"
        try:
            _save_overlay_png(change_db.shape, [
                (possible_flood & ~confirmed_flood, [255, 180, 40, 140]),
                (confirmed_flood,                   [0, 100, 200, 200]),
            ], overlay_path)
            overlays.append({
                "id": "sar_flood",
                "name": "Inundación (radar SAR)",
                "file": "overlay_sar_flood.png",
                "bounds": list(common_bounds),
                "legend": [
                    {"color": "#ffb428", "label": "Posible inundación (Δ < −3 dB)"},
                    {"color": "#0064c8", "label": "Inundación confirmada (Δ < −5 dB)"},
                ],
            })
            logger.info("[Overlay] overlay_sar_flood.png guardado")
        except Exception as exc:
            logger.warning("[Overlay] No se pudo guardar overlay_sar_flood.png: %s", exc)

    valid_change = change_db[~nan_mask]
    stats = {
        "mean_change_dB":      round(float(np.mean(valid_change)),  3),
        "std_change_dB":       round(float(np.std(valid_change)),   3),
        "min_change_dB":       round(float(np.min(valid_change)),   3),
        "max_change_dB":       round(float(np.max(valid_change)),   3),
        "pct_possible_flood":  round(pct_possible,  2),
        "pct_confirmed_flood": round(pct_confirmed, 2),
        "threshold_possible_dB":  -3.0,
        "threshold_confirmed_dB": -5.0,
    }

    logger.info("[SAR] Análisis completado — posible %.1f%%, confirmado %.1f%%",
                pct_possible, pct_confirmed)
    return {"SAR_CHANGE": stats}, output_files, overlays


# ─────────────────────────────────────────────────────────────────────────────
#  Máscara de nubes/sombras a partir de la banda SCL (Scene Classification Layer)
# ─────────────────────────────────────────────────────────────────────────────

# Clases SCL consideradas superficie válida para CUALQUIER índice espectral:
# 4=vegetación, 5=suelo desnudo, 6=agua, 7=sin clasificar, 11=nieve/hielo.
# Se excluyen: 0=nodata, 1=saturado/defectuoso, 2=área oscura (sombra ambigua),
# 3=sombra de nube, 8/9=nube prob. media/alta, 10=cirro fino — todas ellas
# distorsionan cualquier índice (no solo los de agua) si no se filtran.
_VALID_SCL_CLASSES = {4, 5, 6, 7, 11}


def _read_scl_valid_mask(path: str, target: int = 512) -> Optional["np.ndarray"]:
    """Lee la banda SCL y devuelve una máscara booleana: True = píxel de superficie
    válido, False = nube, sombra, nodata o saturado."""
    try:
        import numpy as np
        import rasterio
        from rasterio.enums import Resampling

        with rasterio.open(path) as src:
            h = min(target, src.height)
            w = min(target, src.width)
            # nearest: SCL es una capa categórica, promediar clases no tiene sentido
            scl = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
        return np.isin(scl, list(_VALID_SCL_CLASSES))
    except Exception as exc:
        logger.warning("[CloudMask] No se pudo leer SCL %s: %s", path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Lectura de bandas + fórmulas de índices — compartido por extent y change
# ─────────────────────────────────────────────────────────────────────────────

_INDEX_BAND_SUFFIXES = {"B02": "BLUE", "B03": "GREEN", "B04": "RED",
                         "B08": "NIR", "B11": "SWIR1", "B12": "SWIR2"}


def _read_optical_bands(files: list, target: int = 512) -> dict:
    """Lee todas las bandas ópticas reconocidas disponibles en una lista de
    ficheros descargados. Ignora SCL y bandas SAR (de otro sensor/fase)."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    bands: dict = {}
    for path in (files or []):
        if "_pre_VV" in path or "_post_VV" in path:
            continue  # bandas SAR, no ópticas
        for suffix, band_name in _INDEX_BAND_SUFFIXES.items():
            if path.endswith(f"_{suffix}.tif") and band_name not in bands:
                try:
                    with rasterio.open(path) as src:
                        h = min(target, src.height)
                        w = min(target, src.width)
                        arr = src.read(1, out_shape=(h, w),
                                       resampling=Resampling.average).astype(np.float32)
                    arr[arr <= 0] = np.nan
                    arr /= 10000.0
                    bands[band_name] = arr
                except Exception as exc:
                    logger.warning("No se pudo leer %s: %s", path, exc)

    # Corrección del offset radiométrico BOA (baseline >= 04.00, ene-2022):
    # la reflectancia real es (DN - 1000) / 10000, no DN / 10000. Sin la resta,
    # el agua queda en ~0.10 en todas las bandas y los umbrales absolutos de los
    # índices dejan de cumplirse. Detección por datos: en un tile completo
    # siempre hay píxeles oscuros (agua/sombras) con reflectancia ≈ 0, así que
    # un mínimo global > 0.04 delata el offset.
    if bands:
        global_min = min(float(np.nanmin(a)) for a in bands.values())
        if global_min > 0.04:
            for name in bands:
                bands[name] = np.clip(bands[name] - 0.1, 0.0001, None)
            logger.info("[Bands] Offset BOA -1000 detectado (min=%.4f) y corregido", global_min)
    return bands


def _compute_index_array(index_name: str, bands: dict) -> Optional["np.ndarray"]:
    """Calcula el array de un índice a partir de bandas ya leídas (GREEN, NIR,
    RED, BLUE, SWIR1, SWIR2). Devuelve None si faltan bandas necesarias."""
    import numpy as np
    eps = 1e-10
    try:
        if index_name == "NDVI":
            return np.clip((bands["NIR"] - bands["RED"]) / (bands["NIR"] + bands["RED"] + eps), -1, 1)
        if index_name == "EVI":
            return np.clip(2.5 * (bands["NIR"] - bands["RED"]) /
                            (bands["NIR"] + 6 * bands["RED"] - 7.5 * bands["BLUE"] + 1 + eps), -1, 1)
        if index_name == "SAVI":
            return np.clip(1.5 * (bands["NIR"] - bands["RED"]) / (bands["NIR"] + bands["RED"] + 0.5), -1, 1)
        if index_name == "NDWI":
            return np.clip((bands["GREEN"] - bands["NIR"]) / (bands["GREEN"] + bands["NIR"] + eps), -1, 1)
        if index_name == "MNDWI":
            return np.clip((bands["GREEN"] - bands["SWIR1"]) / (bands["GREEN"] + bands["SWIR1"] + eps), -1, 1)
        if index_name == "AWEI":
            return np.clip(4 * (bands["GREEN"] - bands["SWIR1"]) -
                            (0.25 * bands["NIR"] + 2.75 * bands["SWIR2"]), -2, 2)
        if index_name == "NDBI":
            return np.clip((bands["SWIR1"] - bands["NIR"]) / (bands["SWIR1"] + bands["NIR"] + eps), -1, 1)
        if index_name == "NBR":
            return np.clip((bands["NIR"] - bands["SWIR2"]) / (bands["NIR"] + bands["SWIR2"] + eps), -1, 1)
        if index_name == "NDSI":
            # Misma fórmula que MNDWI (verde/SWIR1) — limitación conocida: sin más
            # contexto, nieve y agua turbia pueden dar una señal parecida.
            return np.clip((bands["GREEN"] - bands["SWIR1"]) / (bands["GREEN"] + bands["SWIR1"] + eps), -1, 1)
    except KeyError:
        return None
    return None


# Umbrales de NIVEL (una sola escena): {índice: [(etiqueta, operador, valor), ...]}
INDEX_LEVEL_THRESHOLDS: dict = {
    "NDWI":  [("probable_water", ">", -0.10), ("confirmed_water", ">", 0.15)],
    "MNDWI": [("water", ">", 0.0)],
    "AWEI":  [("water", ">", 0.0)],
    "NDVI":  [("vegetated", ">", 0.2)],
    "NDBI":  [("built_up", ">", 0.2)],
    "NDSI":  [("snow", ">", 0.4)],
    # NBR bajo en una sola escena es solo indicativo (el agua también lo da);
    # el overlay de fuego excluye píxeles de agua vía NDWI cuando está disponible.
    "NBR":   [("possible_burn", "<", 0.0), ("burn_scar", "<", -0.15)],
}

# Umbrales de CAMBIO real pre/post evento. El signo importa: agua/urbano nuevo
# = aumento; pérdida de vegetación/nieve y severidad de quemado = descenso
# (el dNBR clásico se define como NBR-pre menos NBR-post, es decir cae tras un incendio).
INDEX_CHANGE_THRESHOLDS: dict = {
    "NDWI":  [("probable_new_water", ">", 0.10), ("confirmed_new_water", ">", 0.15)],
    "NDBI":  [("new_built_up", ">", 0.15)],
    "NDVI":  [("vegetation_loss", "<", -0.15), ("vegetation_gain", ">", 0.15)],
    "NBR":   [("low_severity_burn", "<", -0.10), ("high_severity_burn", "<", -0.27)],
    "NDSI":  [("snow_loss", "<", -0.20)],
}

_EXTENT_CANDIDATE_INDICES = tuple(INDEX_LEVEL_THRESHOLDS.keys())
_CHANGE_CANDIDATE_INDICES = tuple(INDEX_CHANGE_THRESHOLDS.keys())


def _apply_threshold(array: "np.ndarray", op: str, value: float):
    return array > value if op == ">" else array < value


# ─────────────────────────────────────────────────────────────────────────────
#  Overlays RGBA para el visor 3D (CesiumJS)
#  Solo los píxeles afectados llevan color; el resto queda transparente, de
#  modo que la capa puede proyectarse sobre el globo sin tapar el mapa base.
#  Cada overlay lleva los bounds REALES del raster en WGS-84 — el tile
#  descargado cubre mucho más que el bbox del plan, y proyectarlo sobre el
#  bbox desplazaría las zonas marcadas de su posición verdadera.
# ─────────────────────────────────────────────────────────────────────────────

def _raster_bounds_4326(files: list) -> Optional[list]:
    """Devuelve [W, S, E, N] en WGS-84 del primer GeoTIFF legible de la lista."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError:
        return None
    for path in files or []:
        if not str(path).endswith(".tif"):
            continue
        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    continue
                w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            return [round(w, 6), round(s, 6), round(e, 6), round(n, 6)]
        except Exception as exc:
            logger.warning("[Overlay] No se pudieron leer bounds de %s: %s", path, exc)
    return None


def _save_overlay_png(shape: tuple, classes: list, path: Path) -> None:
    """Guarda un PNG RGBA transparente. `classes` es [(máscara_bool, [r,g,b,a]), ...];
    se pintan en orden, de modo que la última clase queda encima."""
    import numpy as np
    from PIL import Image

    rgba = np.zeros((*shape, 4), dtype=np.uint8)
    for mask, color in classes:
        rgba[mask] = color
    Image.fromarray(rgba, "RGBA").save(str(path))


def _build_extent_overlays(
    analysis_type: Optional[str],
    arrays: dict,
    valid: "np.ndarray",
    outputs_dir: Path,
    bounds: Optional[list],
) -> list:
    """
    Genera los overlays de zonas afectadas relevantes para el tipo de análisis
    (agua→azul, fuego→rojo, vegetación→verde, urbano→morado, nieve→cian) y
    devuelve sus metadatos: [{id, name, file, bounds, legend}, ...].
    """
    import numpy as np

    atype = (analysis_type or "general").lower()

    # Firma de agua por SIGNO (NDWI>0: verde>NIR; MNDWI>0: verde>SWIR1), el
    # criterio estándar de la literatura. Los signos de las diferencias no
    # dependen del offset radiométrico, a diferencia de umbrales como 0.15.
    water_like = None
    if "NDWI" in arrays:
        water_like = valid & (arrays["NDWI"] > 0.0)
        if "MNDWI" in arrays:
            water_like = water_like & (arrays["MNDWI"] > 0.0)

    # (id, nombre de capa, [(máscara, rgba, color_css, etiqueta_leyenda), ...])
    specs: list = []

    if "NDWI" in arrays and atype in ("flood", "water", "general"):
        ndwi = arrays["NDWI"]
        if "MNDWI" in arrays:
            # Con SWIR disponible, MNDWI manda: NDWI solo da mucho falso
            # positivo en urbano/suelo desnudo/bruma.
            mndwi = arrays["MNDWI"]
            confirmed = valid & (mndwi > 0.0) & (ndwi > 0.0)
            probable  = valid & ((mndwi > 0.0) | (ndwi > 0.15)) & ~confirmed
        else:
            confirmed = valid & (ndwi > 0.15)
            probable  = valid & (ndwi > 0.05) & ~confirmed
        specs.append(("water", "Zonas de agua", [
            (probable,  [80, 180, 255, 110], "#50b4ff", "Agua probable"),
            (confirmed, [0, 100, 200, 190],  "#0064c8", "Agua confirmada"),
        ]))

    if "NBR" in arrays and atype == "fire":
        burned = valid & (arrays["NBR"] < 0.0)
        if water_like is not None:
            burned = burned & ~water_like  # el agua también da NBR bajo
        severe = burned & (arrays["NBR"] < -0.15)
        specs.append(("burn", "Zonas quemadas", [
            (burned & ~severe, [255, 140, 0, 140], "#ff8c00", "Posible zona quemada"),
            (severe,           [200, 30, 30, 200], "#c81e1e", "Zona quemada (severa)"),
        ]))

    if "NDVI" in arrays and atype in ("vegetation", "soil", "general"):
        dense    = valid & (arrays["NDVI"] > 0.5)
        moderate = valid & (arrays["NDVI"] > 0.2) & ~dense
        specs.append(("vegetation", "Vegetación", [
            (moderate, [150, 210, 130, 100], "#96d282", "Vegetación moderada"),
            (dense,    [20, 140, 60, 160],   "#148c3c", "Vegetación densa"),
        ]))

    if "NDBI" in arrays and atype == "urban":
        built = valid & (arrays["NDBI"] > 0.2)
        specs.append(("urban", "Superficie edificada", [
            (built, [155, 80, 210, 160], "#9b50d2", "Zona edificada"),
        ]))

    if "NDSI" in arrays and atype == "snow":
        snow = valid & (arrays["NDSI"] > 0.4)
        if water_like is not None:
            snow = snow & ~water_like  # NDSI y agua turbia se confunden
        specs.append(("snow", "Nieve / hielo", [
            (snow, [130, 220, 255, 180], "#82dcff", "Nieve o hielo"),
        ]))

    shape = next(iter(arrays.values())).shape
    overlays: list = []
    for overlay_id, name, classes in specs:
        if not any(bool(np.any(mask)) for mask, _rgba, _c, _l in classes):
            continue  # nada que marcar — no se genera capa vacía
        filename = f"overlay_{overlay_id}.png"
        try:
            _save_overlay_png(shape, [(m, rgba) for m, rgba, _c, _l in classes],
                              outputs_dir / filename)
        except Exception as exc:
            logger.warning("[Overlay] No se pudo guardar %s: %s", filename, exc)
            continue
        overlays.append({
            "id": overlay_id,
            "name": name,
            "file": filename,
            "bounds": bounds,
            "legend": [{"color": css, "label": label} for _m, _rgba, css, label in classes],
        })
        logger.info("[Overlay] %s guardado (%s)", filename, name)
    return overlays


# ─────────────────────────────────────────────────────────────────────────────
#  Extensión por píxel — evidencia independiente del código del LLM
#  Vale para cualquier tipo de análisis (agua, vegetación, urbano, fuego, nieve),
#  no solo inundaciones: evita que una señal localizada se diluya en la media
#  de un bbox grande, y enmascara nubes/sombras con SCL cuando está disponible.
# ─────────────────────────────────────────────────────────────────────────────

def _index_extent_analysis(
    downloaded_files: list,
    outputs_dir: Path,
    analysis_type: Optional[str] = None,
) -> tuple[dict, list, list]:
    try:
        import numpy as np
        from PIL import Image
        import matplotlib
    except ImportError as exc:
        logger.warning("[IndexExtent] Dependencia no disponible: %s", exc)
        return {}, [], []

    bands = _read_optical_bands(downloaded_files)
    if not bands:
        logger.warning("[IndexExtent] Sin bandas ópticas legibles — se omite")
        return {}, [], []

    total_scene_pixels = int(next(iter(bands.values())).size)

    arrays: dict = {}
    for idx_name in _EXTENT_CANDIDATE_INDICES:
        arr = _compute_index_array(idx_name, bands)
        if arr is not None:
            arrays[idx_name] = arr

    if not arrays:
        logger.warning("[IndexExtent] Ningún índice calculable con las bandas disponibles (%s)",
                        list(bands.keys()))
        return {}, [], []

    valid = np.ones(next(iter(arrays.values())).shape, dtype=bool)
    for arr in arrays.values():
        valid &= ~np.isnan(arr)

    scl_path = next((f for f in (downloaded_files or []) if f.endswith("_SCL.tif")), None)
    if scl_path:
        cloud_mask = _read_scl_valid_mask(scl_path)
        if cloud_mask is not None:
            h = min(valid.shape[0], cloud_mask.shape[0])
            w = min(valid.shape[1], cloud_mask.shape[1])
            valid = valid[:h, :w] & cloud_mask[:h, :w]
            arrays = {k: v[:h, :w] for k, v in arrays.items()}
            logger.info("[IndexExtent] Máscara SCL aplicada — %.1f%% píxeles válidos tras filtrar nubes/sombras",
                        float(np.mean(cloud_mask[:h, :w])) * 100)
    else:
        logger.warning("[IndexExtent] Banda SCL no disponible — no se filtran nubes/sombras")

    total = int(np.sum(valid)) or 1
    not_urban = None
    if "NDBI" in arrays:
        not_urban = ~_apply_threshold(arrays["NDBI"], ">", 0.0)  # NDBI>0 → superficie edificada

    stats: dict = {"pct_area_valid": round(total / total_scene_pixels * 100, 2)}
    for idx_name, thresholds in INDEX_LEVEL_THRESHOLDS.items():
        arr = arrays.get(idx_name)
        if arr is None:
            continue
        prefix = idx_name.lower()
        for label, op, value in thresholds:
            mask = valid & _apply_threshold(arr, op, value)
            stats[f"{prefix}_pct_{label}"] = round(float(np.sum(mask)) / total * 100, 2)
            # NDWI/MNDWI confunden superficie edificada con agua (limitación conocida)
            if idx_name in ("NDWI", "MNDWI") and not_urban is not None:
                stats[f"{prefix}_pct_{label}_excl_urban"] = round(
                    float(np.sum(mask & not_urban)) / total * 100, 2)
    if "NDBI" in arrays:
        stats["pct_urban_like"] = round(
            float(np.sum(valid & _apply_threshold(arrays["NDBI"], ">", 0.0))) / total * 100, 2)

    output_files: list = []
    if "NDWI" in arrays:
        mask_path = outputs_dir / "water_extent_mask.png"
        try:
            confirmed = valid & (arrays["NDWI"] > 0.15)
            if "MNDWI" in arrays:
                confirmed = confirmed & (arrays["MNDWI"] > 0.0)
            probable = valid & (arrays["NDWI"] > -0.10) & ~confirmed
            mask_rgb = np.full((*arrays["NDWI"].shape, 3), 180, dtype=np.uint8)  # gris = sin evidencia
            mask_rgb[probable]  = [255, 200, 0]    # amarillo = posible agua turbia
            mask_rgb[confirmed] = [0,   100, 200]  # azul = agua confirmada
            mask_rgb[~valid]    = [0,   0,   0]
            Image.fromarray(mask_rgb, "RGB").save(str(mask_path))
            output_files.append(str(mask_path))
        except Exception as exc:
            logger.warning("[IndexExtent] No se pudo guardar water_extent_mask.png: %s", exc)

    # Overlays transparentes para el globo 3D, con bounds reales del raster
    optical_files = [f for f in (downloaded_files or [])
                     if any(str(f).endswith(f"_{s}.tif") for s in _INDEX_BAND_SUFFIXES)]
    overlays = _build_extent_overlays(
        analysis_type, arrays, valid, outputs_dir,
        bounds=_raster_bounds_4326(optical_files),
    )

    logger.info("[IndexExtent] índices=%s stats=%s overlays=%s",
                list(arrays.keys()), stats, [o["id"] for o in overlays])
    return {"INDEX_EXTENT": stats}, output_files, overlays


# ─────────────────────────────────────────────────────────────────────────────
#  Cambio real pre/post evento (Sentinel-2, sin baseline asumido)
#  Generaliza el cambio óptico a cualquier índice calculable (no solo NDWI):
#  compara una escena real previa despejada con la escena post-evento, en vez
#  de asumir un valor de referencia fijo.
# ─────────────────────────────────────────────────────────────────────────────

def _index_change_analysis(
    pre_files: list,
    post_files: list,
    outputs_dir: Path,
) -> tuple[dict, list, list]:
    try:
        import numpy as np
        from PIL import Image
        import matplotlib
    except ImportError as exc:
        logger.warning("[IndexChange] Dependencia no disponible: %s", exc)
        return {}, [], []

    pre_bands  = _read_optical_bands(pre_files)
    post_bands = _read_optical_bands(post_files)
    if not pre_bands or not post_bands:
        logger.warning("[IndexChange] Bandas pre y/o post no disponibles — se omite")
        return {}, [], []

    # Se conservan también los arrays pre/post (no solo la diferencia) para
    # poder clasificar "agua nueva": firma de agua en post que no estaba en pre.
    # MNDWI se calcula aunque no tenga umbral de cambio, porque refuerza la firma.
    pre_idx: dict = {}
    post_idx: dict = {}
    changes: dict = {}
    for idx_name in (*_CHANGE_CANDIDATE_INDICES, "MNDWI"):
        pre_arr  = _compute_index_array(idx_name, pre_bands)
        post_arr = _compute_index_array(idx_name, post_bands)
        if pre_arr is None or post_arr is None:
            continue
        h = min(pre_arr.shape[0], post_arr.shape[0])
        w = min(pre_arr.shape[1], post_arr.shape[1])
        pre_idx[idx_name]  = pre_arr[:h, :w]
        post_idx[idx_name] = post_arr[:h, :w]
        if idx_name in _CHANGE_CANDIDATE_INDICES:
            changes[idx_name] = post_arr[:h, :w] - pre_arr[:h, :w]

    if not changes:
        logger.warning("[IndexChange] Ningún índice de cambio calculable con las bandas disponibles")
        return {}, [], []

    h = min(arr.shape[0] for arr in pre_idx.values())
    w = min(arr.shape[1] for arr in pre_idx.values())
    changes  = {k: v[:h, :w] for k, v in changes.items()}
    pre_idx  = {k: v[:h, :w] for k, v in pre_idx.items()}
    post_idx = {k: v[:h, :w] for k, v in post_idx.items()}
    total_scene_pixels = h * w

    valid = np.ones((h, w), dtype=bool)
    for arr in changes.values():
        valid &= ~np.isnan(arr)

    # Enmascarar nubes/sombras en AMBAS fechas — un píxel solo es válido si está
    # despejado tanto en la escena pre-evento como en la post-evento.
    pre_scl_path  = next((f for f in pre_files  if f.endswith("_SCL.tif")), None)
    post_scl_path = next((f for f in post_files if f.endswith("_SCL.tif")), None)
    if pre_scl_path and post_scl_path:
        pre_mask  = _read_scl_valid_mask(pre_scl_path)
        post_mask = _read_scl_valid_mask(post_scl_path)
        if pre_mask is not None and post_mask is not None:
            hh = min(valid.shape[0], pre_mask.shape[0], post_mask.shape[0])
            ww = min(valid.shape[1], pre_mask.shape[1], post_mask.shape[1])
            valid = valid[:hh, :ww] & pre_mask[:hh, :ww] & post_mask[:hh, :ww]
            changes  = {k: v[:hh, :ww] for k, v in changes.items()}
            pre_idx  = {k: v[:hh, :ww] for k, v in pre_idx.items()}
            post_idx = {k: v[:hh, :ww] for k, v in post_idx.items()}
    else:
        logger.warning("[IndexChange] Banda SCL no disponible en pre y/o post — no se filtran nubes/sombras")

    total = int(np.sum(valid)) or 1

    stats: dict = {"pct_area_valid": round(total / total_scene_pixels * 100, 2)}
    output_files: list = []
    for idx_name, change_arr in changes.items():
        prefix = idx_name.lower()
        stats[f"{prefix}_mean_change"] = round(float(np.mean(change_arr[valid])), 4)
        for label, op, value in INDEX_CHANGE_THRESHOLDS.get(idx_name, []):
            mask = valid & _apply_threshold(change_arr, op, value)
            stats[f"{prefix}_pct_{label}"] = round(float(np.sum(mask)) / total * 100, 2)

        if idx_name == "NDWI":
            change_path = outputs_dir / "optical_ndwi_change.png"
            try:
                display = np.nan_to_num(change_arr, nan=0.0)
                norm = np.clip((display + 0.5) / 1.0, 0.0, 1.0)
                rgba = (matplotlib.colormaps["RdBu"](norm) * 255).astype(np.uint8)
                Image.fromarray(rgba, "RGBA").save(str(change_path))
                output_files.append(str(change_path))
            except Exception as exc:
                logger.warning("[IndexChange] No se pudo guardar optical_ndwi_change.png: %s", exc)

    # ── Overlay "agua nueva": firma de agua en post que NO estaba en pre ──
    # Marca solo la inundación, no el mar ni los cauces permanentes. La firma
    # es por signo (NDWI>0, reforzada con MNDWI>0 si hay SWIR en ambas fechas)
    # y la clase confirmada exige además una subida clara de NDWI.
    overlays: list = []
    if "NDWI" in pre_idx and "NDWI" in post_idx:
        pre_water  = valid & (pre_idx["NDWI"] > 0.0)
        post_water = valid & (post_idx["NDWI"] > 0.0)
        if "MNDWI" in pre_idx and "MNDWI" in post_idx:
            pre_water  = pre_water  & (pre_idx["MNDWI"] > 0.0)
            post_water = post_water & (post_idx["MNDWI"] > 0.0)
        appeared  = post_water & ~pre_water
        confirmed = appeared & (changes["NDWI"] > 0.10)
        probable  = appeared & ~confirmed
        stats["pct_new_water"] = round(float(np.sum(appeared)) / total * 100, 2)
        if bool(np.any(appeared)):
            try:
                _save_overlay_png(appeared.shape, [
                    (probable,  [80, 180, 255, 130]),
                    (confirmed, [0, 100, 200, 200]),
                ], outputs_dir / "overlay_new_water.png")
                output_files.append(str(outputs_dir / "overlay_new_water.png"))
                overlays.append({
                    "id": "new_water",
                    "name": "Agua nueva (inundación)",
                    "file": "overlay_new_water.png",
                    "bounds": _raster_bounds_4326(post_files),
                    "legend": [
                        {"color": "#50b4ff", "label": "Agua nueva probable"},
                        {"color": "#0064c8", "label": "Agua nueva confirmada"},
                    ],
                })
                logger.info("[Overlay] overlay_new_water.png guardado — %.2f%% agua nueva",
                            stats["pct_new_water"])
            except Exception as exc:
                logger.warning("[Overlay] No se pudo guardar overlay_new_water.png: %s", exc)

    logger.info("[IndexChange] índices=%s stats=%s", list(changes.keys()), stats)
    return {"INDEX_CHANGE": stats}, output_files, overlays


# ─────────────────────────────────────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────────────────────────────────────

class AnalystAgent:
    """Nodo LangGraph: genera → ejecuta → depura código Python geoespacial."""

    def __init__(self, model: str | None = None):
        self.llm = ChatGroq(
            model=model or GROQ_MODEL,
            temperature=0.0,
            max_tokens=4096,
        )

    # ── Construcción del prompt inicial ──────────────────────────────────

    def _build_task_prompt(self, state: GeoAgentState) -> str:
        downloaded = state.get("downloaded_files") or []
        indices = state.get("required_indices") or ["NDVI"]
        analysis_type = state.get("analysis_type") or "vegetation"
        location = state.get("location") or {}
        location_name = (
            location.get("name") if isinstance(location, dict) else str(location)
        ) or "unknown location"
        bbox = location.get("bbox") if isinstance(location, dict) else None

        if not downloaded:
            data_section = (
                "Band files: NONE — generate all raster data synthetically using np.random."
            )
        else:
            file_lines = "\n".join(f"  {f}" for f in downloaded)
            bbox_line = (
                f"Bounding box (WGS-84, min_lon lat max_lon lat): {bbox}"
                if bbox else ""
            )
            data_section = (
                f"Band files (Sentinel-2 L2A, uint16 DN 0-10000):\n{file_lines}\n"
                f"{bbox_line}\n"
                "IMPORTANT: mask pixels <= 0 as NaN, divide by 10000 to get reflectance."
            )

        return textwrap.dedent(f"""
            Generate a complete Python analysis script for:
            - Analysis: {analysis_type}
            - Location: {location_name}
            - Indices : {indices}
            - Output  : {OUTPUTS_DIR}

            {data_section}

            Follow ALL rules in the system prompt exactly.
            Return ONLY the ```python ... ``` block, nothing else.
        """).strip()

    # ── Guardado de imágenes de depuración desde archivos TIF reales ─────

    @staticmethod
    def _save_debug_from_files(downloaded_files: list, scene_info: dict | None = None) -> None:
        """Lee las bandas de los TIF descargados y guarda imágenes de debug."""
        if not downloaded_files:
            return
        try:
            import numpy as np
            import rasterio
            from rasterio.enums import Resampling

            TARGET = 512
            BAND_MAP = {"B02": "BLUE", "B03": "GREEN", "B04": "RED",
                        "B08": "NIR",  "B11": "SWIR1", "B12": "SWIR2"}
            bands: dict = {}
            for path in downloaded_files:
                for suffix, band_name in BAND_MAP.items():
                    if path.endswith(f"_{suffix}.tif") and band_name not in bands:
                        try:
                            with rasterio.open(path) as src:
                                h = min(TARGET, src.height)
                                w = min(TARGET, src.width)
                                arr = src.read(1, out_shape=(h, w),
                                               resampling=Resampling.average).astype(np.float32)
                            arr[arr <= 0] = np.nan
                            arr /= 10000.0
                            bands[band_name] = arr
                        except Exception as exc:
                            logger.warning("[Debug] No se pudo leer %s: %s", path, exc)

            if bands:
                _save_rgb_composite(bands, is_synthetic=False, scene_info=scene_info)
        except Exception as exc:
            logger.warning("[Debug] _save_debug_from_files falló: %s", exc)

    # ── Loop principal ────────────────────────────────────────────────────

    def __call__(self, state: GeoAgentState) -> dict:
        # Guardar imágenes de debug desde TIFs reales antes de cualquier análisis
        self._save_debug_from_files(
            state.get("downloaded_files") or [],
            scene_info=state.get("selected_scene"),
        )

        task_prompt = self._build_task_prompt(state)

        messages = [
            SystemMessage(content=ANALYST_SYSTEM),
            HumanMessage(content=task_prompt),
        ]

        code: Optional[str] = None
        execution_result: Optional[dict] = None
        iterations = 0

        for iteration in range(MAX_CODE_ITERATIONS):
            iterations = iteration + 1

            try:
                response = self.llm.invoke(messages)
            except Exception as exc:
                logger.error("[Analyst] LLM invoke failed iter %d: %s", iterations, exc)
                break

            raw_content = response.content if hasattr(response, "content") else str(response)
            messages.append(AIMessage(content=raw_content))

            code = _extract_code(raw_content)
            if not code:
                logger.warning("[Analyst] Sin bloque de código en iter %d — reintentando", iterations)
                messages.append(HumanMessage(
                    content="Tu respuesta no contiene un bloque ```python ... ```. "
                            "Devuelve ÚNICAMENTE el bloque de código sin ningún texto adicional."
                ))
                continue

            logger.info("[Analyst] Ejecutando código (iter %d / %d)…", iterations, MAX_CODE_ITERATIONS)
            result = _execute_code(code)
            execution_result = result

            if result["error"] is None:
                logger.info("[Analyst] Ejecución exitosa en iter %d", iterations)
                break

            # Realimentar el error al LLM
            error_feedback = (
                f"El código falló. Corrige SOLO el error y devuelve el script completo corregido.\n\n"
                f"STDOUT (últimas 1500 chars):\n{result['stdout'][-1500:]}\n\n"
                f"STDERR:\n{result['stderr'][-500:]}\n\n"
                f"TRACEBACK:\n{result['error'][-2000:]}\n\n"
                f"Devuelve ÚNICAMENTE el bloque ```python ... ``` corregido."
            )
            logger.warning("[Analyst] Error iter %d: %s", iterations, result["error"][:200])
            messages.append(HumanMessage(content=error_feedback))

        # ── Parsear resultados ────────────────────────────────────────────
        stdout = execution_result.get("stdout", "") if execution_result else ""
        computed_indices, output_files = _parse_summary(stdout)

        # Fallback directo: si el código LLM falló o no generó PNGs, computar en Python
        downloaded = state.get("downloaded_files") or []
        required   = state.get("required_indices") or ["NDVI"]
        if not output_files or execution_result and execution_result.get("error"):
            fb_indices, fb_files = _python_fallback_compute(
                downloaded, required, state.get("location")
            )
            if fb_indices:
                computed_indices = fb_indices
                output_files = fb_files
                logger.info("[Analyst] Fallback Python completado: %s", list(fb_indices.keys()))

        if not computed_indices:
            for png in OUTPUTS_DIR.glob("*.png"):
                if png.stem.lower() in ["ndvi","evi","savi","ndwi","mndwi","ndbi","nbr","ndsi","bsi"]:
                    computed_indices.setdefault(png.stem.upper(), {})

        # ── Análisis SAR Sentinel-1 (solo cuando hay datos disponibles) ────
        map_overlays: list = []
        sar_available = state.get("sar_available") or False
        if sar_available:
            pre_files  = state.get("pre_scene_files") or []
            post_files = [f for f in (state.get("downloaded_files") or [])
                          if "_post_VV.tif" in f]
            location = state.get("location") or {}
            sar_indices, sar_files, sar_overlays = _sar_flood_analysis(
                pre_files=pre_files,
                post_files=post_files,
                outputs_dir=OUTPUTS_DIR,
                bbox=location.get("bbox") if isinstance(location, dict) else None,
            )
            if sar_indices:
                computed_indices.update(sar_indices)
                output_files.extend(sar_files)
                map_overlays.extend(sar_overlays)
                logger.info("[Analyst] SAR índices fusionados: %s", list(sar_indices.keys()))
            else:
                logger.warning("[Analyst] SAR no devolvió resultados (fallback óptico)")

        # ── Extensión por píxel — siempre que haya bandas ópticas, para
        # CUALQUIER tipo de análisis (no solo inundación): evita que una señal
        # localizada (una inundación, un incendio, una mancha urbana) se diluya
        # en la media de un bbox grande, y filtra nubes/sombras con SCL.
        ie_indices, ie_files, ie_overlays = _index_extent_analysis(
            downloaded, OUTPUTS_DIR, analysis_type=state.get("analysis_type")
        )
        if ie_indices:
            computed_indices.update(ie_indices)
            output_files.extend(ie_files)
            map_overlays.extend(ie_overlays)
            logger.info("[Analyst] INDEX_EXTENT calculado: %s", ie_indices["INDEX_EXTENT"])
        else:
            logger.warning("[Analyst] INDEX_EXTENT no disponible (bandas insuficientes)")

        # ── Cambio real pre/post (cualquier índice) ────────────────────────
        pre_optical_files = state.get("pre_optical_files") or []
        if pre_optical_files:
            _RECOGNIZED_SUFFIXES = ("B02", "B03", "B04", "B08", "B11", "B12", "SCL")
            post_optical_files = [
                f for f in downloaded if any(f.endswith(f"_{s}.tif") for s in _RECOGNIZED_SUFFIXES)
            ]
            ic_indices, ic_files, ic_overlays = _index_change_analysis(
                pre_files=pre_optical_files,
                post_files=post_optical_files,
                outputs_dir=OUTPUTS_DIR,
            )
            if ic_indices:
                computed_indices.update(ic_indices)
                output_files.extend(ic_files)
                map_overlays.extend(ic_overlays)
                logger.info("[Analyst] INDEX_CHANGE calculado: %s", ic_indices["INDEX_CHANGE"])
            else:
                logger.warning("[Analyst] INDEX_CHANGE no disponible")

        # Cuando hay detección de agua NUEVA (óptica pre/post o SAR), la capa
        # estática con TODAS las zonas de agua (incluido el mar y los cauces
        # permanentes) se deja oculta por defecto — sigue disponible desde la
        # leyenda del visor, pero lo que se muestra es solo la inundación.
        if any(o["id"] in ("new_water", "sar_flood") for o in map_overlays):
            for o in map_overlays:
                if o["id"] == "water":
                    o["visible"] = False

        logger.info(
            "[Analyst] computed_indices=%s  output_files=%d  sar=%s  error=%s",
            list(computed_indices.keys()),
            len(output_files),
            sar_available,
            bool(execution_result and execution_result.get("error")),
        )

        return {
            "generated_code": code,
            "execution_result": execution_result,
            "computed_indices": computed_indices,
            "output_files": output_files,
            "map_overlays": map_overlays or None,
            "code_iterations": iterations,
            "analysis_error": (
                execution_result.get("error") if execution_result else "No se generó código"
            ),
            "current_agent": "reporter",
            "messages": [{
                "agent": "analyst",
                "type": "status",
                "content": (
                    f"Código ejecutado en {iterations} iteración(es). "
                    f"Índices: {list(computed_indices.keys())}. "
                    f"Archivos: {len(output_files)}. "
                    f"SAR: {'disponible ✓' if sar_available else 'no disponible'}."
                ),
            }],
        }