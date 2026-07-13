# Manual de Instalación y Puesta en Marcha — GeoMultiAgent

Este manual describe, paso a paso, cómo instalar y ejecutar en local los dos componentes que forman el sistema GeoMultiAgent:

1. **`geo_multiagent_backend`** — API en Python (FastAPI + LangGraph) que orquesta los cuatro agentes (Planner, DataAcquisition, Analyst, Reporter).
2. **`front_geo_multiagent`** — Interfaz web en JavaScript (Vite + CesiumJS) que consume la API y muestra el globo 3D con los resultados.

Ambos proyectos son independientes: el backend expone una API REST + WebSocket, y el frontend es una aplicación estática que se conecta a esa API. Pueden ejecutarse en la misma máquina o en máquinas distintas.

---

## 1. Requisitos previos

| Herramienta | Versión mínima | Uso |
|---|---|---|
| [Python](https://www.python.org/downloads/) | 3.12 | Backend |
| [Node.js](https://nodejs.org/) (incluye npm) | 18 LTS | Frontend |
| [Git](https://git-scm.com/) | Cualquiera | Clonar los repositorios |
| Cuenta [Groq](https://console.groq.com/keys) | — | API key gratuita, obligatoria para el LLM |

Los datos satelitales se obtienen de catálogos STAC de acceso público (Microsoft Planetary Computer y Element84 Earth Search), por lo que no se necesita ninguna credencial adicional a la de Groq.

> **Nota de seguridad:** nunca compartas ni subas a un repositorio público el archivo `.env` con tus claves reales. Usa siempre `.env.example` como plantilla.

---

## 2. Backend (`geo_multiagent_backend`)

### 2.1 Clonar el repositorio

```bash
git clone https://github.com/pablocarvajal12/geo_multiagent_backend.git
cd geo_multiagent_backend
```

### 2.2 Crear y activar un entorno virtual

```bash
python -m venv venv

# Windows (PowerShell / cmd)
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2.3 Instalar las dependencias

```bash
pip install -r requirements.txt
```

Incluye LangGraph/LangChain, librerías geoespaciales y de procesamiento (rasterio, numpy, matplotlib) y FastAPI/Uvicorn. La instalación puede tardar varios minutos por el tamaño de las librerías geoespaciales.

### 2.4 Configurar las variables de entorno

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Edita `.env` y rellena como mínimo:

```dotenv
GROQ_API_KEY=tu_clave_de_groq        # Obligatorio — https://console.groq.com/keys
GROQ_MODEL=llama-3.3-70b-versatile   # Opcional, este es el valor por defecto
```

### 2.5 Arrancar el servidor

```bash
python main.py
```

Esto levanta el backend en `http://localhost:8000` (host y puerto configurables con `APP_HOST` / `APP_PORT` en `.env`).

### 2.6 Verificar que funciona

- Documentación interactiva de la API (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)
- Prueba rápida por línea de comandos, sin necesidad del frontend:

  ```bash
  python cli.py --demo
  ```

---

## 3. Frontend (`front_geo_multiagent`)

### 3.1 Clonar el repositorio

```bash
git clone https://github.com/pablocarvajal12/front_geo_multiagent.git
cd front_geo_multiagent
```

### 3.2 Instalar las dependencias

```bash
npm install
```

Instala Vite, el plugin `vite-plugin-cesium` y la librería `cesium` (motor de globo 3D).

### 3.3 Configuración

El frontend se conecta al backend mediante las constantes `API_BASE` y `WS_BASE` definidas al principio de `main.js`:

```js
var API_BASE = "http://localhost:8000";
var WS_BASE  = "ws://localhost:8000";
```

Si el backend se ejecuta en otra máquina o en otro puerto, actualiza estos valores antes de arrancar o compilar el frontend.

> El proyecto incluye un token de [Cesium Ion](https://ion.cesium.com/) de ejemplo (variable `Ion.defaultAccessToken` en `main.js`) para cargar el terreno/imaginería base. Para un despliegue propio se recomienda generar un token gratuito en tu propia cuenta de Cesium Ion y sustituirlo.

### 3.4 Ejecutar en modo desarrollo

```bash
npm run dev
```

Vite levanta un servidor local (por defecto en `http://localhost:5173`) con recarga en caliente.

### 3.5 Compilar para producción

```bash
npm run build      # genera los archivos estáticos en dist/
npm run preview    # sirve esa build para comprobarla localmente
```

Los archivos de `dist/` pueden desplegarse en cualquier servidor de archivos estáticos (Nginx, Netlify, GitHub Pages, etc.), siempre que `API_BASE`/`WS_BASE` apunten a la URL pública del backend.

---

## 4. Puesta en marcha conjunta

1. Arranca primero el **backend** (`python main.py`, puerto 8000).
2. Arranca el **frontend** (`npm run dev`, puerto 5173).
3. Abre el frontend en el navegador, escribe una consulta en lenguaje natural (o usa uno de los ejemplos predefinidos) y pulsa "Ejecutar análisis".
4. El frontend abre un WebSocket contra el backend, muestra el progreso de cada agente en tiempo real y, al terminar, carga el informe, la tabla de datos, el código Python generado y las capas del análisis sobre el globo 3D.

---

## 5. Solución de problemas comunes

| Problema | Causa probable | Solución |
|---|---|---|
| El frontend muestra "No se pudo conectar con el servidor" | El backend no está arrancado o corre en otro host/puerto | Verifica `python main.py` y que `API_BASE`/`WS_BASE` en `main.js` coincidan |
| Error instalando `rasterio` / `geopandas` en Windows | Falta de wheels precompiladas para tu versión de Python | Usa Python 3.12 (la versión probada) y una versión reciente de `pip` (`python -m pip install --upgrade pip`) antes de instalar `requirements.txt` |
| El Analyst usa "datos sintéticos de demostración" | No se pudo descargar ninguna banda (sin conexión, catálogo STAC caído o ninguna escena en la ventana de fechas) | Comprueba la conexión y revisa el log de la adquisición; los catálogos (Planetary Computer, Earth Search) son públicos y no requieren credenciales |
| `401`/`invalid_api_key` del LLM | `GROQ_API_KEY` ausente o incorrecta en `.env` | Genera una clave en [console.groq.com/keys](https://console.groq.com/keys) |
| Puerto 8000 u 5173 ocupado | Otro proceso usando ese puerto | Cambia `APP_PORT` en `.env` (backend) o lanza Vite con `npm run dev -- --port <otro>` (frontend), actualizando `API_BASE`/`WS_BASE` en consecuencia |
