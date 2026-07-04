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
    BAND_MAP = {"B02": "BLUE", "B03": "GREEN", "B04": "RED",
                "B08": "NIR",  "B11": "SWIR1", "B12": "SWIR2"}

    # ── Leer bandas desde TIF ─────────────────────────────────────────────
    bands: dict = {}
    for path in (downloaded_files or []):
        for suffix, band_name in BAND_MAP.items():
            if path.endswith(f"_{suffix}.tif") and band_name not in bands:
                try:
                    with rasterio.open(path) as src:
                        h = min(TARGET, src.height)
                        w = min(TARGET, src.width)
                        arr = src.read(
                            1,
                            out_shape=(h, w),
                            resampling=Resampling.average,
                        ).astype(np.float32)
                    arr[arr <= 0] = np.nan
                    arr /= 10000.0
                    bands[band_name] = arr
                    logger.info("[Fallback] Banda %s leída: shape=%s", band_name, arr.shape)
                except Exception as exc:
                    logger.warning("[Fallback] No se pudo leer %s: %s", path, exc)

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
) -> tuple[dict, list]:
    """
    Detección de inundaciones por cambio de retrodispersión SAR (Sentinel-1 GRD VV).

    Método:
      1. Lee bandas VV pre y post evento (float32 en unidades de potencia lineal)
      2. Convierte a dB: σ° = 10 * log10(valor + ε)
      3. Imagen de cambio: Δσ° = post_dB - pre_dB
      4. Máscara: Δ < -3 dB = posible inundación, Δ < -5 dB = inundación confirmada

    El SAR penetra nubes — detecta agua durante la tormenta misma.
    El agua tranquila refleja la señal lejos del sensor → muy baja retrodispersión.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.enums import Resampling
        from PIL import Image
        import matplotlib
    except ImportError as exc:
        logger.warning("[SAR] Dependencia no disponible: %s", exc)
        return {}, []

    if not pre_files or not post_files:
        logger.warning("[SAR] pre_files o post_files vacíos — saltando análisis SAR")
        return {}, []

    TARGET = 512

    def _read_vv(path: str) -> Optional["np.ndarray"]:
        try:
            with rasterio.open(path) as src:
                h = min(TARGET, src.height)
                w = min(TARGET, src.width)
                arr = src.read(1, out_shape=(h, w),
                               resampling=Resampling.average).astype(np.float32)
            arr[arr <= 0] = np.nan
            return arr
        except Exception as exc:
            logger.warning("[SAR] No se pudo leer %s: %s", path, exc)
            return None

    pre_vv  = _read_vv(pre_files[0])
    post_vv = _read_vv(post_files[0])

    if pre_vv is None or post_vv is None:
        logger.warning("[SAR] Falta pre o post VV")
        return {}, []

    # Alinear dimensiones
    h = min(pre_vv.shape[0], post_vv.shape[0])
    w = min(pre_vv.shape[1], post_vv.shape[1])
    pre_vv, post_vv = pre_vv[:h, :w], post_vv[:h, :w]

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
    return {"SAR_CHANGE": stats}, output_files


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
#  Extensión por píxel — evidencia independiente del código del LLM
#  Vale para cualquier tipo de análisis (agua, vegetación, urbano, fuego, nieve),
#  no solo inundaciones: evita que una señal localizada se diluya en la media
#  de un bbox grande, y enmascara nubes/sombras con SCL cuando está disponible.
# ─────────────────────────────────────────────────────────────────────────────

def _index_extent_analysis(
    downloaded_files: list,
    outputs_dir: Path,
) -> tuple[dict, list]:
    try:
        import numpy as np
        from PIL import Image
        import matplotlib
    except ImportError as exc:
        logger.warning("[IndexExtent] Dependencia no disponible: %s", exc)
        return {}, []

    bands = _read_optical_bands(downloaded_files)
    if not bands:
        logger.warning("[IndexExtent] Sin bandas ópticas legibles — se omite")
        return {}, []

    total_scene_pixels = int(next(iter(bands.values())).size)

    arrays: dict = {}
    for idx_name in _EXTENT_CANDIDATE_INDICES:
        arr = _compute_index_array(idx_name, bands)
        if arr is not None:
            arrays[idx_name] = arr

    if not arrays:
        logger.warning("[IndexExtent] Ningún índice calculable con las bandas disponibles (%s)",
                        list(bands.keys()))
        return {}, []

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

    logger.info("[IndexExtent] índices=%s stats=%s", list(arrays.keys()), stats)
    return {"INDEX_EXTENT": stats}, output_files


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
) -> tuple[dict, list]:
    try:
        import numpy as np
        from PIL import Image
        import matplotlib
    except ImportError as exc:
        logger.warning("[IndexChange] Dependencia no disponible: %s", exc)
        return {}, []

    pre_bands  = _read_optical_bands(pre_files)
    post_bands = _read_optical_bands(post_files)
    if not pre_bands or not post_bands:
        logger.warning("[IndexChange] Bandas pre y/o post no disponibles — se omite")
        return {}, []

    changes: dict = {}
    for idx_name in _CHANGE_CANDIDATE_INDICES:
        pre_arr  = _compute_index_array(idx_name, pre_bands)
        post_arr = _compute_index_array(idx_name, post_bands)
        if pre_arr is None or post_arr is None:
            continue
        h = min(pre_arr.shape[0], post_arr.shape[0])
        w = min(pre_arr.shape[1], post_arr.shape[1])
        changes[idx_name] = post_arr[:h, :w] - pre_arr[:h, :w]

    if not changes:
        logger.warning("[IndexChange] Ningún índice de cambio calculable con las bandas disponibles")
        return {}, []

    h = min(arr.shape[0] for arr in changes.values())
    w = min(arr.shape[1] for arr in changes.values())
    changes = {k: v[:h, :w] for k, v in changes.items()}
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
            changes = {k: v[:hh, :ww] for k, v in changes.items()}
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

    logger.info("[IndexChange] índices=%s stats=%s", list(changes.keys()), stats)
    return {"INDEX_CHANGE": stats}, output_files


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
        sar_available = state.get("sar_available") or False
        if sar_available:
            pre_files  = state.get("pre_scene_files") or []
            post_files = [f for f in (state.get("downloaded_files") or [])
                          if "_post_VV.tif" in f]
            sar_indices, sar_files = _sar_flood_analysis(
                pre_files=pre_files,
                post_files=post_files,
                outputs_dir=OUTPUTS_DIR,
            )
            if sar_indices:
                computed_indices.update(sar_indices)
                output_files.extend(sar_files)
                logger.info("[Analyst] SAR índices fusionados: %s", list(sar_indices.keys()))
            else:
                logger.warning("[Analyst] SAR no devolvió resultados (fallback óptico)")

        # ── Extensión por píxel — siempre que haya bandas ópticas, para
        # CUALQUIER tipo de análisis (no solo inundación): evita que una señal
        # localizada (una inundación, un incendio, una mancha urbana) se diluya
        # en la media de un bbox grande, y filtra nubes/sombras con SCL.
        ie_indices, ie_files = _index_extent_analysis(downloaded, OUTPUTS_DIR)
        if ie_indices:
            computed_indices.update(ie_indices)
            output_files.extend(ie_files)
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
            ic_indices, ic_files = _index_change_analysis(
                pre_files=pre_optical_files,
                post_files=post_optical_files,
                outputs_dir=OUTPUTS_DIR,
            )
            if ic_indices:
                computed_indices.update(ic_indices)
                output_files.extend(ic_files)
                logger.info("[Analyst] INDEX_CHANGE calculado: %s", ic_indices["INDEX_CHANGE"])
            else:
                logger.warning("[Analyst] INDEX_CHANGE no disponible")

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