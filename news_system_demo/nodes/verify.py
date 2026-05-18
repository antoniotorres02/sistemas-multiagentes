"""Verify node for the didactic demo."""

from __future__ import annotations

from news_system_demo.llm import LlmClientProtocol
from news_system_demo.models import EvidencePayload, State, StateUpdate
from news_system_demo.nodes.shared import build_evidence_prompt
from news_system_demo.runtime import RunLogger


async def verify_node(
    state: State,
    *,
    llm_client: LlmClientProtocol,
    logger: RunLogger,
) -> StateUpdate:
    """Produce one short plain-language evidence note."""

    selected_items = state.get("selected_items", [])
    if not selected_items:
        evidence_note = "No se encontró evidencia suficiente en el corpus local."
        logger.step("verify", "No hay evidencia que verificar.", [evidence_note])
        return {"evidence_note": evidence_note}
    try:
        response = await llm_client.complete_json(
            system_prompt=(
                "Eres un verificador editorial. Devuelve JSON con evidence_note. "
                "Resume en texto breve qué permite afirmar la evidencia y qué no conviene exagerar."
            ),
            user_prompt=build_evidence_prompt(state["topic"], selected_items),
            response_model=EvidencePayload,
            temperature=0.1,
        )
    except Exception as exc:
        logger.error("verify", f"No se pudo obtener una nota de evidencia válida: {exc}")
        raise
    logger.step("verify", "Se resume la evidencia útil para la noticia.", [response.evidence_note])
    return {"evidence_note": response.evidence_note}
