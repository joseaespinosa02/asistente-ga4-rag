"""Construcción de la base de conocimiento: descarga, limpieza, chunking e indexación en ChromaDB."""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ga4-rag-student-project/1.0)"}

# Contenedores donde vive el texto del artículo en cada sitio (de más a menos específico)
CONTENT_SELECTORS = [
    "div.devsite-article-body",   # developers.google.com
    "div.article-content-container",
    "section.article-content",
    "div.cc",                     # cuerpo de artículos del Centro de Ayuda
    "article",
    "main",
]
NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript", "svg", "form", "button"]
NOISE_PATTERNS = re.compile(
    r"^(¿Te ha resultado útil\?|¿Cómo podemos mejorarla\?|Sí|No|Enviar|Buscar|Borrar búsqueda|"
    r"Cerrar búsqueda|Menú principal|Aplicaciones de Google|Was this helpful\?|Send feedback)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 1. Carga de documentos
# ---------------------------------------------------------------------------
def clean_html(html: str) -> str:
    """Extrae el texto útil de una página y elimina menús, scripts y ruido."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    container = next((soup.select_one(s) for s in CONTENT_SELECTORS if soup.select_one(s)), None)
    text = (container or soup.body or soup).get_text("\n")

    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if len(line) > 1 and not NOISE_PATTERNS.match(line):
            lines.append(line)
    return "\n".join(lines)


def _cache_path(url: str) -> Path:
    return config.CACHE_DIR / f"{hashlib.md5(url.encode()).hexdigest()[:12]}.txt"


def fetch_source(source: dict[str, str], use_cache: bool = True) -> Document | None:
    """Descarga y limpia una página web (con caché local para no repetir descargas)."""
    cache = _cache_path(source["url"])
    if use_cache and cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        try:
            resp = requests.get(source["url"], headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  ✗ No se pudo descargar {source['url']}: {exc}")
            return None
        text = clean_html(resp.text)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")

    if len(text) < 300:  # página vacía o bloqueada: mejor no indexarla
        print(f"  ✗ Contenido insuficiente en {source['url']} ({len(text)} caracteres)")
        return None
    return Document(page_content=text, metadata={"title": source["title"], "source": source["url"]})


def load_local_files(folder: Path = config.DATA_DIR) -> list[Document]:
    """Carga PDFs, .md y .txt que el usuario deje en /data (opcional)."""
    docs: list[Document] = []
    if not folder.exists():
        return docs
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            pages = PyPDFLoader(str(path)).load()
            text = "\n".join(p.page_content for p in pages)
        elif path.suffix.lower() in {".md", ".txt"} and path.name.lower() != "readme.md":
            text = path.read_text(encoding="utf-8")
        else:
            continue
        docs.append(Document(page_content=text, metadata={"title": path.stem, "source": path.name}))
    return docs


def load_documents(use_cache: bool = True) -> list[Document]:
    """Carga todas las fuentes web configuradas + los archivos locales de /data."""
    docs = []
    for src in config.SOURCES:
        doc = fetch_source(src, use_cache=use_cache)
        if doc:
            docs.append(doc)
    docs.extend(load_local_files())
    return docs


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------
def split_documents(docs: list[Document]) -> list[Document]:
    """Divide los documentos en fragmentos solapados, respetando párrafos y frases."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


# ---------------------------------------------------------------------------
# 3. Embeddings + ChromaDB
# ---------------------------------------------------------------------------
TRANSIENT_ERRORS = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "INTERNAL", "UNAVAILABLE", "DEADLINE")


def with_retries(fn, *args, attempts: int = 6, **kwargs):
    """Reintenta con espera creciente los errores temporales de la API de Gemini (límites, 500, 503)."""
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts - 1 or not any(code in str(exc) for code in TRANSIENT_ERRORS):
                raise
            if "PerDay" in str(exc) or "per day" in str(exc):
                raise RuntimeError(
                    "Se agotó la cuota DIARIA gratuita de la API de Gemini para este modelo. "
                    "Espera a que se reinicie (indicado en el error) o usa otro modelo de embeddings "
                    "con la variable GEMINI_EMBEDDING_MODEL."
                ) from exc
            wait = min(5 * 2**attempt, 60)
            print(f"  Error temporal de la API ({str(exc)[:60]}…). Reintento en {wait}s")
            time.sleep(wait)


class RetryingEmbeddings(Embeddings):
    """Envuelve los embeddings de Gemini para que los fallos puntuales del servidor no rompan el flujo."""

    def __init__(self, inner: Embeddings):
        self.inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return with_retries(self.inner.embed_documents, texts)

    def embed_query(self, text: str) -> list[float]:
        # Se envía la consulta por la misma vía que los documentos (formato por lotes), pero con
        # task_type RETRIEVAL_QUERY: la vía de texto suelto de embed_query devolvía 500 INTERNAL
        # de forma sistemática con gemini-embedding-001.
        try:
            return with_retries(self.inner.embed_documents, [text], task_type="RETRIEVAL_QUERY")[0]
        except TypeError:  # embeddings que no aceptan task_type (p. ej. en tests)
            return with_retries(self.inner.embed_query, text)


def get_embeddings() -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    config.load_api_key()
    model = os.getenv("GEMINI_EMBEDDING_MODEL", config.EMBEDDING_MODEL)
    print(f"Modelo de embeddings: {model}")
    return RetryingEmbeddings(GoogleGenerativeAIEmbeddings(model=model))


def _add_in_batches(store: Chroma, chunks: list[Document], batch_size: int = 50) -> None:
    """Indexa por lotes para respetar los límites de la capa gratuita de la API."""
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        ids = [f"chunk-{c.metadata['chunk_id']}" for c in batch]
        store.add_documents(batch, ids=ids)
        print(f"  Indexados {min(start + batch_size, len(chunks))}/{len(chunks)} fragmentos")


def build_vectorstore(
    chunks: list[Document],
    embeddings: Embeddings | None = None,
    persist_dir: Path = config.CHROMA_DIR,
    reset: bool = True,
) -> Chroma:
    """Crea (o recrea) la colección en ChromaDB con los fragmentos y sus embeddings."""
    embeddings = embeddings or get_embeddings()
    store = Chroma(
        collection_name=config.collection_name(),
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    existing = store._collection.count()
    if existing == len(chunks):
        # Mismo contenido ya indexado (p. ej. tras reiniciar la sesión): no se gasta cuota de la API
        print(f"  Índice ya existente con {existing} fragmentos: se reutiliza")
        return store
    if reset and existing > 0:
        store.reset_collection()
    _add_in_batches(store, chunks)
    return store


def load_vectorstore(embeddings: Embeddings | None = None, persist_dir: Path = config.CHROMA_DIR) -> Chroma:
    """Abre la colección ya persistida en disco."""
    return Chroma(
        collection_name=config.collection_name(),
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_or_build_vectorstore(embeddings: Embeddings | None = None) -> Chroma:
    """Reutiliza el índice si ya existe; si no, lo construye de cero."""
    embeddings = embeddings or get_embeddings()
    store = load_vectorstore(embeddings)
    if store._collection.count() == 0:
        chunks = split_documents(load_documents())
        store = build_vectorstore(chunks, embeddings, reset=False)
    return store
