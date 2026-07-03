"""
main.py - Punto de Entrada de la Aplicación API
Inicia el backend de FastAPI puramente como API para que sea consumido
por el frontend independiente en JavaScript (CesiumJS).

Uso:
    python main.py
    # o
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import os
from dotenv import load_dotenv

# Cargar variables de entorno del archivo .env
load_dotenv()

# Importamos la aplicación configurada desde backend.py
from backend import app

if __name__ == "__main__":
    import uvicorn
    
    # Arrancar el servidor ASGI Uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
    )