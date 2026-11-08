"""Configuración central del asistente: modelos, rutas, fuentes y API key."""

from __future__ import annotations

import os
from pathlib import Path

# --- Rutas -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"            # PDFs / .md / .txt propios (opcional)
CACHE_DIR = DATA_DIR / "cache"          # copia local de las páginas descargadas
CHROMA_DIR = ROOT_DIR / "chroma_db"     # base vectorial persistente (no se sube a git)

# --- Modelos de Gemini (se pueden cambiar con variables de entorno) --------
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")  # respaldo si no se puede listar
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_FALLBACKS = ["gemini-embedding-001", "gemini-embedding-2"]  # se prueban en orden si uno falla


def collection_name() -> str:
    """Una colección por modelo de embeddings: los vectores de modelos distintos no son compatibles."""
    model = os.getenv("GEMINI_EMBEDDING_MODEL", EMBEDDING_MODEL)
    return "ga4_docs_" + "".join(c if c.isalnum() else "_" for c in model)

# --- Parámetros RAG --------------------------------------------------------
CHUNK_SIZE = 1000        # caracteres por fragmento
CHUNK_OVERLAP = 150      # solapamiento para no cortar ideas a la mitad
TOP_K = 5                # fragmentos que se pasan al LLM
FETCH_K = 20             # candidatos que evalúa MMR antes de elegir los TOP_K
MAX_HISTORY_MESSAGES = 10  # turnos recientes que ve el LLM al generar

# --- Base de conocimiento: documentación oficial de GA4 en español ---------
# Centro de Ayuda de Google Analytics + guías para desarrolladores.
# Las páginas se descargan al construir el índice; el repo no redistribuye su contenido.
SOURCES: list[dict[str, str]] = [
    # Fundamentos y configuración
    {"title": "Introducción a Google Analytics 4", "url": "https://support.google.com/analytics/answer/10089681?hl=es"},
    {"title": "ID de medición", "url": "https://support.google.com/analytics/answer/9539598?hl=es"},
    # Eventos
    {"title": "Acerca de los eventos", "url": "https://support.google.com/analytics/answer/9322688?hl=es"},
    {"title": "Eventos recogidos automáticamente", "url": "https://support.google.com/analytics/answer/9234069?hl=es"},
    {"title": "Eventos de medición mejorada", "url": "https://support.google.com/analytics/answer/9216061?hl=es"},
    {"title": "Eventos recomendados", "url": "https://support.google.com/analytics/answer/9267735?hl=es"},
    {"title": "Configurar parámetros de eventos", "url": "https://support.google.com/analytics/answer/11396839?hl=es"},
    {"title": "Dimensiones y métricas personalizadas", "url": "https://support.google.com/analytics/answer/10075209?hl=es"},
    # Depuración e informes
    {"title": "Informe en tiempo real", "url": "https://support.google.com/analytics/answer/9271392?hl=es"},
    {"title": "DebugView", "url": "https://support.google.com/analytics/answer/7201382?hl=es"},
    # Adquisición y atribución
    {"title": "Grupo de canales predeterminado", "url": "https://support.google.com/analytics/answer/9756891?hl=es"},
    {"title": "Informe de rutas de atribución", "url": "https://support.google.com/analytics/answer/10595568?hl=es"},
    {"title": "Informe de rendimiento de todos los canales", "url": "https://support.google.com/analytics/answer/12198930?hl=es"},
    # Guías para desarrolladores (implementación)
    {"title": "Guía para desarrolladores: enviar eventos", "url": "https://developers.google.com/analytics/devguides/collection/ga4/events?hl=es"},
    {"title": "Guía para desarrolladores: medir comercio electrónico", "url": "https://developers.google.com/analytics/devguides/collection/ga4/ecommerce?hl=es"},
    {"title": "Referencia de eventos recomendados", "url": "https://developers.google.com/analytics/devguides/collection/ga4/reference/events?hl=es"},
]


def load_api_key() -> str:
    """Busca la API key de Gemini en Colab Secrets, .env o variables de entorno.

    Funciona igual en Google Colab, Jupyter local y Streamlit Cloud
    (Streamlit expone los secrets de primer nivel como variables de entorno).
    """
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not key:
        try:  # Google Colab -> icono de la llave (Secrets) en la barra lateral
            from google.colab import userdata  # type: ignore

            key = userdata.get("GOOGLE_API_KEY")
        except Exception:
            pass

    if not key:
        try:  # Jupyter local -> archivo .env en la raíz del proyecto
            from dotenv import load_dotenv

            load_dotenv(ROOT_DIR / ".env")
            key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        except ImportError:
            pass

    # Limpia saltos de línea o espacios pegados por error (p. ej. la clave pegada dos veces)
    if key:
        key = key.strip().split()[0]

    if not key:
        raise RuntimeError(
            "No encontré la API key de Gemini. Defínela como GOOGLE_API_KEY en "
            "Colab Secrets, en el archivo .env o en los secrets de Streamlit."
        )

    os.environ["GOOGLE_API_KEY"] = key  # langchain-google-genai la lee de aquí
    return key
