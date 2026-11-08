"""Agente RAG en LangGraph: reformula la pregunta, recupera de ChromaDB y responde con Gemini."""

from __future__ import annotations

import os
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from . import config

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Eres «Analytica», un consultor experto en Google Analytics 4 (GA4) que ayuda a \
profesionales de marketing digital a configurar, medir e interpretar sus datos.

## Cómo respondes
- Te basas ÚNICAMENTE en el CONTEXTO recuperado de la documentación oficial de GA4 que aparece abajo.
- Si el contexto no contiene la respuesta, lo dices con claridad («No encuentro esa información en la \
documentación que tengo») y sugieres qué buscar o reformular. Nunca inventas nombres de eventos, \
parámetros, menús ni límites.
- Cuando el usuario se refiera a algo dicho antes («eso», «el segundo», «¿y cómo se activa?»), usa el \
historial de la conversación para entender a qué se refiere.
- Citas las fuentes al final con el formato: «Fuentes: [título 1], [título 2]».

## Estilo
- Español neutro, tono profesional y cercano, como un analista senior explicando a un compañero.
- Empiezas con la respuesta directa en 1-2 frases y luego das el detalle.
- Usas listas o pasos numerados para procedimientos (rutas del menú de Administración, configuraciones).
- Nombres técnicos tal cual aparecen en GA4 (`page_view`, `purchase`, «Eventos clave»).
- Respuestas concisas: máximo ~250 palabras salvo que pidan más detalle.

## Límites
- Si la pregunta no tiene relación con analítica web o GA4, indicas amablemente que solo puedes \
ayudar con Google Analytics 4.
- No das consejos legales sobre privacidad o consentimiento; puedes explicar lo que dice la \
documentación y recomendar consultar a un especialista.

## CONTEXTO
{context}
"""

CONTEXTUALIZE_PROMPT = """Dado el historial de la conversación y la última pregunta del usuario, \
reescribe la pregunta para que se entienda por sí sola, sin el historial (sustituye «eso», «ese \
evento», «el anterior», etc. por lo que corresponda). NO respondas la pregunta: devuelve solo la \
pregunta reescrita. Si ya es independiente, devuélvela tal cual."""


# ---------------------------------------------------------------------------
# Estado del grafo
# ---------------------------------------------------------------------------
class AgentState(MessagesState):
    """Mensajes de la conversación + pregunta reformulada + fragmentos recuperados."""

    standalone_question: str
    context: list[dict]


def resolve_llm_model() -> str:
    """Elige el modelo Flash más reciente disponible para la API key.

    Los nombres de Gemini cambian con frecuencia (y los alias no siempre están habilitados),
    así que se consulta la lista real de modelos. Si se define GEMINI_MODEL, se respeta.
    """
    if os.getenv("GEMINI_MODEL"):
        return os.environ["GEMINI_MODEL"]
    try:
        from google import genai

        available = [
            m.name.removeprefix("models/")
            for m in genai.Client().models.list()
            if m.supported_actions is None or "generateContent" in m.supported_actions
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo listar los modelos ({exc}); uso {config.LLM_MODEL}")
        return config.LLM_MODEL

    excluded = ("lite", "image", "tts", "live", "audio", "exp", "preview", "thinking", "8b")
    flash = [
        m for m in available
        if m.startswith("gemini-") and "flash" in m and not any(x in m for x in excluded)
    ]
    if not flash:  # sin estables: aceptar previews
        flash = [m for m in available if m.startswith("gemini-") and "flash" in m and "lite" not in m]

    def version(name: str) -> tuple:
        nums = re.findall(r"\d+(?:\.\d+)?", name.split("flash")[0])
        return (float(nums[0]) if nums else 0.0, -len(name))

    # Las familias 2.x ya no se ofrecen a cuentas nuevas: se priorizan las 3.x o superiores
    flash = [m for m in flash if version(m)[0] >= 3] or flash
    if not flash:
        return config.LLM_MODEL
    return max(flash, key=version)


def get_llm() -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    config.load_api_key()
    model = resolve_llm_model()
    config.LLM_MODEL = model
    print(f"Modelo LLM: {model}")
    # max_retries: reintenta automáticamente errores temporales del servidor (500/503/429)
    return ChatGoogleGenerativeAI(model=model, max_retries=6)


def format_context(context: list[dict]) -> str:
    if not context:
        return "(No se recuperó ningún fragmento relevante.)"
    return "\n\n---\n\n".join(
        f"[Fuente {i}: {c['title']}]\n{c['content']}" for i, c in enumerate(context, start=1)
    )


# ---------------------------------------------------------------------------
# Construcción del grafo
# ---------------------------------------------------------------------------
def build_agent(vectorstore: VectorStore, llm: BaseChatModel | None = None, checkpointer=None):
    """Devuelve el grafo compilado: contextualize -> retrieve -> generate (con memoria)."""
    llm = llm or get_llm()
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance: relevantes y no redundantes
        search_kwargs={"k": config.TOP_K, "fetch_k": config.FETCH_K},
    )

    def contextualize(state: AgentState) -> dict:
        """Nodo 1: convierte preguntas de seguimiento en preguntas autónomas para el retriever."""
        question = state["messages"][-1].text
        previous = state["messages"][:-1]
        if not previous:
            return {"standalone_question": question}
        history = previous[-config.MAX_HISTORY_MESSAGES :]
        rewritten = llm.invoke(
            [SystemMessage(CONTEXTUALIZE_PROMPT), *history, HumanMessage(question)]
        )
        return {"standalone_question": rewritten.text.strip() or question}

    def retrieve(state: AgentState) -> dict:
        """Nodo 2: busca en ChromaDB los fragmentos más relevantes."""
        docs = retriever.invoke(state["standalone_question"])
        context = [
            {
                "content": d.page_content,
                "title": d.metadata.get("title", "Documento"),
                "source": d.metadata.get("source", ""),
            }
            for d in docs
        ]
        return {"context": context}

    def generate(state: AgentState) -> dict:
        """Nodo 3: Gemini responde usando el system prompt, el contexto y el historial."""
        system = SystemMessage(SYSTEM_PROMPT.format(context=format_context(state["context"])))
        history = state["messages"][-config.MAX_HISTORY_MESSAGES :]
        response = llm.invoke([system, *history])
        return {"messages": [AIMessage(response.text)]}

    graph = StateGraph(AgentState)
    graph.add_node("contextualize", contextualize)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # El checkpointer guarda el estado por thread_id -> memoria entre turnos
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def ask(agent, question: str, thread_id: str = "demo") -> dict:
    """Envía una pregunta al agente dentro de un hilo de conversación y devuelve el estado final."""
    return agent.invoke(
        {"messages": [HumanMessage(question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
