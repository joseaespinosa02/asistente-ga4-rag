"""Interfaz web del asistente GA4. Ejecutar con:  streamlit run app/streamlit_app.py"""

import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.agent import ask, build_agent  # noqa: E402
from src.ingest import get_or_build_vectorstore  # noqa: E402

st.set_page_config(page_title="Analytica · Asistente GA4", page_icon="📊")


@st.cache_resource(show_spinner="Preparando la base de conocimiento (solo la primera vez)…")
def load_agent():
    config.load_api_key()
    store = get_or_build_vectorstore()
    return build_agent(store), store._collection.count()


# --- Barra lateral -----------------------------------------------------------
with st.sidebar:
    st.header("📊 Analytica")
    st.write("Asistente experto en **Google Analytics 4** basado en la documentación oficial.")
    st.caption(f"LLM: `{config.LLM_MODEL}` · Embeddings: `{config.EMBEDDING_MODEL}`")
    if st.button("🗑️ Nueva conversación"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    st.markdown("**Prueba a preguntar:**")
    st.markdown(
        "- ¿Qué tipos de eventos hay en GA4?\n"
        "- ¿Qué recoge la medición mejorada?\n"
        "- ¿Y cómo se desactiva?\n"
        "- ¿Puedo editar el grupo de canales predeterminado?"
    )

try:
    agent, n_chunks = load_agent()
except RuntimeError as err:
    st.error(str(err))
    st.stop()

st.sidebar.caption(f"Fragmentos indexados en ChromaDB: {n_chunks}")

# --- Estado de la sesión: cada pestaña del navegador es un hilo de memoria ----
st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("history", [])

st.title("Asistente experto en GA4")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Fragmentos consultados"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s['title']}]({s['source']})")

if question := st.chat_input("Pregunta lo que quieras sobre Google Analytics 4…"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la documentación…"):
            result = ask(agent, question, thread_id=st.session_state.thread_id)
        answer = result["messages"][-1].text
        sources = list({c["source"]: c for c in result["context"]}.values())
        st.markdown(answer)
        with st.expander("Fragmentos consultados"):
            st.caption(f"Búsqueda: _{result['standalone_question']}_")
            for s in sources:
                st.markdown(f"- [{s['title']}]({s['source']})")

    st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})
