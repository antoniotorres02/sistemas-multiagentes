"""Review node for the didactic demo."""

from __future__ import annotations

from news_system_demo.llm import LlmClientProtocol
from news_system_demo.models import ReviewPayload, State
from news_system_demo.nodes.shared import build_review_prompt, load_workspace
from news_system_demo.runtime import RunLogger


async def review_node(
    state: State,
    *,
    llm_client: LlmClientProtocol,
    logger: RunLogger,
) -> State:
    """Review the article and decide whether to rewrite it once."""

    workspace = load_workspace()
    revision_count = state.get("revision_count", 0)
    try:
        response = await llm_client.complete_json(
            system_prompt=(
                "Eres un editor. Devuelve JSON con approved y note. "
                "No apruebes si hay errores claros, falta de fuentes o exageraciones."
            ),
            user_prompt=build_review_prompt(
                state["topic"],
                state.get("article_text", ""),
                state.get("evidence_note", ""),
                revision_count,
            ),
            response_model=ReviewPayload,
            temperature=0.1,
        )
    except Exception as exc:
        logger.error("review", f"No se pudo revisar la noticia: {exc}")
        raise
    force_first_revision = workspace.force_first_revision and revision_count < 1
    revision_limit_reached = revision_count >= workspace.max_revisions
    needs_revision = force_first_revision or (not response.approved and not revision_limit_reached)
    next_revision_count = revision_count + 1 if needs_revision else revision_count
    if needs_revision:
        next_step = "Se reescribirá una vez para que el bucle del grafo sea visible."
    elif response.approved:
        next_step = "La noticia queda aprobada."
    else:
        next_step = "La noticia se cierra por límite de revisiones."
    logger.step(
        "review",
        "Se revisa la noticia antes de guardarla.",
        [
            f"Aprobada por el editor: {response.approved}",
            f"Nota: {response.note}",
            next_step,
        ],
    )
    return {
        "review_note": response.note,
        "needs_revision": needs_revision,
        "revision_count": next_revision_count,
    }


def route_after_review(state: State) -> str:
    """Choose the next node after review."""

    return "write" if state.get("needs_revision", False) else "render"
