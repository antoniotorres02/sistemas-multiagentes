"""Write node for the didactic demo."""

from __future__ import annotations

from news_system_demo.llm import LlmClientProtocol
from news_system_demo.models import ArticlePayload, State
from news_system_demo.nodes.shared import build_article_prompt, build_no_evidence_article
from news_system_demo.runtime import RunLogger


async def write_node(
    state: State,
    *,
    llm_client: LlmClientProtocol,
    logger: RunLogger,
) -> State:
    """Draft a clean Markdown article from the selected evidence."""

    selected_items = state.get("selected_items", [])
    evidence_note = state.get("evidence_note", "")
    revision_count = state.get("revision_count", 0)
    if not selected_items:
        article_text = build_no_evidence_article(state["topic"])
    else:
        try:
            response = await llm_client.complete_json(
                system_prompt=(
                    "Eres un redactor de noticias. Devuelve JSON con article_text. "
                    "article_text debe ser Markdown limpio con titular, entradilla, cuerpo y fuentes. "
                    "No menciones LangGraph, agentes, estado interno ni trazas."
                ),
                user_prompt=build_article_prompt(
                    state["topic"],
                    selected_items,
                    evidence_note,
                    state.get("review_note"),
                    revision_count,
                ),
                response_model=ArticlePayload,
                temperature=0.1,
            )
        except Exception as exc:
            logger.error("write", f"No se pudo redactar una noticia válida: {exc}")
            raise
        article_text = response.article_text
    logger.step(
        "write",
        "Se redacta la noticia final en Markdown.",
        [f"Longitud: {len(article_text)} caracteres"],
    )
    return {"article_text": article_text, "needs_revision": False}
