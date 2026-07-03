# GeoMultiAgent 🛰️

Sistema multiagente basado en **LangGraph + Claude** para el análisis de datos de Observación de la Tierra mediante lenguaje natural.

## Arquitectura

```
Usuario (lenguaje natural)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                   FastAPI Backend                      │
│          REST API  +  WebSocket streaming              │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│              LangGraph Workflow                        │
│                                                        │
│  ┌──────────┐   ┌──────────────┐   ┌───────────┐     │
│  │ Planner  │──▶│ DataAcquisi- │──▶│  Analyst  │     │
│  │  Agent   │   │  tion Agent  │   │   Agent   │     │
│  └──────────┘   └──────────────┘   └─────┬─────┘     │
│                                          │            │
│                                    ┌─────▼──────┐    │
│                                    │  Reporter  │    │
│                                    │   Agent    │    │
│                                    └────────────┘    │
└───────────────────────────────────────────────────────┘
```

## Agentes

| Agente | Responsabilidad |
|--------|----------------|
| **Planner** | Interpreta la consulta natural → genera plan estructurado (bbox, fechas, índices, satélites) |
| **DataAcquisition** | Busca en catálogos STAC (Planetary Computer, Earth Search) y descarga bandas espectrales |
| **Analyst** | Genera, ejecuta y depura código Python para calcular índices (NDVI, EVI, NDWI…) |
| **Reporter** | Sintetiza resultados → informe en lenguaje natural + mapa Folium interactivo |

## Instalación

```bash
# 1. Clonar y entrar al proyecto
cd geo_multiagent

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar credenciales
cp .env.example .env
# Edita .env y añade tu ANTHROPIC_API_KEY
```

## Configuración `.env`

```dotenv
ANTHROPIC_API_KEY=sk-ant-...      # Obligatorio

# Opcional (amplía las fuentes de datos):
COPERNICUS_USER=...
COPERNICUS_PASSWORD=...
EARTHDATA_USERNAME=...
EARTHDATA_PASSWORD=...
```

> **Nota:** Sin credenciales de satélite, el Agente Analista usará datos sintéticos de demostración.

## Uso

### Interfaz Web

```bash
python main.py
# Abre http://localhost:8000
```

### CLI

```bash
# Demo integrado
python cli.py --demo

# Consulta personalizada
python cli.py "Analiza la deforestación en el estado de Pará, Brasil en 2023"

# Salida JSON completa
python cli.py --demo --json
```

### API REST

```bash
# Ejecutar análisis
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Estado de la vegetación en Madrid, verano 2024"}'

# WebSocket (streaming)
# Ver frontend/index.html para ejemplo completo
```

## Estructura del proyecto

```
geo_multiagent/
├── agents/
│   ├── __init__.py
│   ├── planner.py           # Agente planificador
│   ├── data_acquisition.py  # Agente adquisición de datos
│   ├── analyst.py           # Agente analista geoespacial
│   └── reporter.py          # Agente generador de informes
├── frontend/
│   └── index.html           # Interfaz web
├── data/                    # Bandas espectrales descargadas
├── outputs/                 # Resultados (mapas, imágenes, informes)
├── state.py                 # Definición del estado LangGraph
├── workflow.py              # Grafo LangGraph principal
├── backend.py               # API FastAPI + WebSocket
├── serve_frontend.py        # Sirve el frontend desde FastAPI
├── main.py                  # Punto de entrada principal
├── cli.py                   # Interfaz de línea de comandos
├── requirements.txt
└── .env.example
```

## Índices soportados

| Índice | Aplicación | Bandas |
|--------|-----------|--------|
| NDVI | Vegetación | NIR, Rojo |
| EVI  | Vegetación mejorado | NIR, Rojo, Azul |
| NDWI | Agua | Verde, NIR |
| MNDWI | Agua modificado | Verde, SWIR1 |
| NDBI | Zonas urbanas | SWIR1, NIR |
| NBR  | Áreas quemadas | NIR, SWIR2 |
| NDSI | Nieve/hielo | Verde, SWIR1 |

## Fuentes de datos satelitales

| Satélite | Resolución | Revisita | Fuente |
|----------|-----------|---------|--------|
| Sentinel-2 L2A | 10–60 m | 5 días | Copernicus / Planetary Computer |
| Landsat-8/9 C2 L2 | 30 m | 16 días | USGS / Earth Search |
| MODIS | 250–1000 m | Diaria | NASA |

## TFM - Trabajo Fin de Máster

Este proyecto implementa la arquitectura descrita en el TFM:
- Módulo de backend y API REST con FastAPI
- Agente Planificador (interpretación de consultas geoespaciales)
- Agente de Adquisición de Datos (conexión autónoma a catálogos satelitales)
- Agente Analista Geoespacial (generación y ejecución de código Python)
- Agente de Síntesis e Informes (informe NL + mapas interactivos)
- Interfaz web de usuario

## Licencia

MIT
