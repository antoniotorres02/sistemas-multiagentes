"""Didactic LangGraph workflow used by the demo CLI."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from news_system_demo.llm import DemoOpenRouterClient
from news_system_demo.models import (
    DemoCorpusItem,
    DemoDraftPayload,
    DemoHandoff,
    DemoReviewPayload,
    DemoRunArtifactPaths,
    DemoState,
    DemoVerificationPayload,
    DemoWorkspace,
)
from news_system_demo.runtime import DemoTracer


class DemoLlmClientProtocol(Protocol):
    """Protocol describing the subset of LLM behavior used by the demo."""

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
        temperature: float = 0.2,
    ) -> Any:
        """Return a structured JSON completion."""


def load_demo_workspace() -> DemoWorkspace:
    """Return the fixed demo workspace configuration."""

    return DemoWorkspace(
        available_topics=[
            "ai regulation",
            "ai in health",
            "energy transition",
            "european policy",
        ]
    )


def load_demo_corpus(corpus_path: Path) -> list[DemoCorpusItem]:
    """Load the local demo corpus from JSON."""

    raw_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    return [DemoCorpusItem.model_validate(item) for item in raw_payload]


def topic_matches_item(topic: str, item: DemoCorpusItem) -> bool:
    """Return whether a user topic matches one corpus item."""

    normalized_topic = topic.lower()
    haystack = " ".join(
        [
            item.title.lower(),
            item.summary.lower(),
            item.body.lower(),
            item.semantic_topic.lower(),
            " ".join(tag.lower() for tag in item.tags),
        ]
    )
    topic_terms = [term for term in normalized_topic.split() if len(term) >= 3]
    return any(term in haystack for term in topic_terms)


def normalize_research_item(item: DemoCorpusItem) -> dict[str, Any]:
    """Normalize one corpus item into the shape used by downstream agents."""

    return {
        "item_id": item.item_id,
        "title": item.title,
        "summary": item.summary,
        "source_name": item.source_name,
        "url": item.url,
        "published_at": item.published_at.isoformat(),
        "semantic_topic": item.semantic_topic,
        "content_excerpt": item.body[:360],
        "tags": item.tags,
    }


def build_story_briefs(raw_items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group research items into didactic story briefs."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in raw_items:
        grouped.setdefault(str(item["semantic_topic"]), []).append(item)

    story_briefs: list[dict[str, Any]] = []
    for semantic_topic, items in grouped.items():
        representative = items[0]
        source_names = sorted({str(item["source_name"]) for item in items})
        story_briefs.append(
            {
                "story_id": semantic_topic.replace(" ", "-"),
                "semantic_topic": semantic_topic,
                "representative_title": representative["title"],
                "source_names": source_names,
                "independent_source_count": len(source_names),
                "items": list(items),
            }
        )
    story_briefs.sort(
        key=lambda story: (len(story["items"]), story["representative_title"]),
        reverse=True,
    )
    return story_briefs


def build_verify_prompt(story_brief: dict[str, Any], topic: str) -> str:
    """Build the verification prompt for one didactic story brief."""

    lines = [
        f"Tema del usuario: {topic}",
        f"Historia agrupada: {story_brief['semantic_topic']}",
        f"Fuentes independientes: {story_brief['independent_source_count']}",
    ]
    for item in story_brief["items"]:
        lines.append(f"- {item['source_name']}: {item['title']}")
        lines.append(f"  Resumen: {item['summary']}")
    return "\n".join(lines)


def build_write_prompt(
    topic: str,
    story_briefs: Sequence[dict[str, Any]],
    verifications: Sequence[dict[str, Any]],
    review_feedback: dict[str, Any] | None,
    revision_count: int,
) -> str:
    """Build the writer prompt from verified story briefs."""

    sections: list[str] = [f"Tema del usuario: {topic}", f"Revision actual: {revision_count}"]
    if review_feedback and review_feedback.get("revision_instructions"):
        sections.append("Instrucciones del revisor:")
        for instruction in cast(list[str], review_feedback["revision_instructions"]):
            sections.append(f"- {instruction}")
    for story_brief, verification in zip(story_briefs, verifications, strict=True):
        sections.append(
            "\n".join(
                [
                    f"Historia: {story_brief['representative_title']}",
                    f"Topic semántico: {story_brief['semantic_topic']}",
                    f"Fuentes: {', '.join(story_brief['source_names'])}",
                    f"Conclusión: {verification['conclusion']}",
                    f"Precauciones: {'; '.join(verification['caution_notes']) or 'ninguna'}",
                ]
            )
        )
    return "\n\n".join(sections)


def build_review_prompt(
    topic: str,
    draft: dict[str, Any],
    verifications: Sequence[dict[str, Any]],
    revision_count: int,
) -> str:
    """Build the reviewer prompt from the current draft and verified stories."""

    sections = [
        f"Tema: {topic}",
        f"Revision completada por el writer: {revision_count}",
        f"Titular: {draft.get('headline', '')}",
        f"Resumen: {draft.get('summary', '')}",
        "Puntos clave:",
    ]
    for bullet_point in cast(list[str], draft.get("bullet_points", [])):
        sections.append(f"- {bullet_point}")
    sections.append(f"Cierre: {draft.get('closing_note', '')}")
    sections.append("Verificaciones disponibles:")
    for verification in verifications:
        sections.append(
            "\n".join(
                [
                    f"- {verification['story_id']}: {verification['conclusion']}",
                    f"  Cautelas: {'; '.join(verification['caution_notes']) or 'ninguna'}",
                ]
            )
        )
    return "\n".join(sections)


def build_no_evidence_draft(topic: str) -> dict[str, Any]:
    """Build a deterministic fallback draft when research finds nothing useful."""

    return {
        "headline": f"No hay evidencia suficiente sobre {topic}",
        "summary": (
            "La demo no encontró historias relevantes en el corpus local para ese tema, "
            "así que el flujo termina mostrando el caso de ausencia de evidencia."
        ),
        "bullet_points": [
            "El agente de research no encontró items suficientes en el corpus.",
            "La organización deja la falta de evidencia explícita en el estado compartido.",
        ],
        "closing_note": "La salida sigue siendo trazable aunque no haya redacción sustantiva.",
    }


def render_markdown_report(state: DemoState) -> str:
    """Render the final demo report as Markdown."""

    topic = state["topic"]
    draft = state["draft"]
    verifications = state.get("verifications", [])
    handoffs = state.get("handoffs", [])
    review_feedback = state.get("review_feedback", {})
    lines = [
        f"# news_system_demo LangGraph: {topic}",
        "",
        f"## {draft['headline']}",
        "",
        draft["summary"],
        "",
        "### Puntos clave",
    ]
    for bullet_point in cast(list[str], draft["bullet_points"]):
        lines.append(f"- {bullet_point}")
    lines.extend(
        [
            "",
            "### Cierre",
            draft["closing_note"],
            "",
            "### Revision",
            f"- revision_count={state.get('revision_count', 0)}",
            f"- needs_revision_final={state.get('needs_revision', False)}",
            f"- feedback={review_feedback.get('feedback_summary', 'sin feedback')}",
            "",
            "### Verificacion didactica",
        ]
    )
    for verification in verifications:
        caution_notes = verification["caution_notes"] or ["Sin notas de cautela."]
        lines.append(
            f"- **{verification['story_id']}** · "
            f"confianza={verification['confidence_score']:.2f} · "
            f"fuentes={verification['independent_source_count']}"
        )
        for note in caution_notes:
            lines.append(f"  - {note}")
    lines.extend(["", "### Handoffs"])
    for handoff in handoffs:
        lines.append(
            f"- {handoff['from_agent']} -> {handoff['to_agent']}: {handoff['summary']}"
        )
    return "\n".join(lines)


def summarize_state(state: DemoState) -> dict[str, Any]:
    """Summarize key counters from the current shared state."""

    return {
        "raw_items": len(state.get("raw_items", [])),
        "story_briefs": len(state.get("story_briefs", [])),
        "verifications": len(state.get("verifications", [])),
        "handoffs": len(state.get("handoffs", [])),
        "revision_count": state.get("revision_count", 0),
        "has_draft": "draft" in state,
    }


def append_handoff(
    state: DemoState,
    tracer: DemoTracer,
    handoff: DemoHandoff,
) -> list[dict[str, Any]]:
    """Append one handoff to shared state and trace it."""

    next_handoffs = list(state.get("handoffs", []))
    handoff_payload = handoff.model_dump(mode="json")
    next_handoffs.append(handoff_payload)
    tracer.handoff(handoff)
    return next_handoffs


def build_demo_graph(
    *,
    corpus_path: Path,
    artifacts: DemoRunArtifactPaths,
    tracer: DemoTracer,
    llm_client: DemoLlmClientProtocol,
    checkpointer: SqliteSaver,
) -> Any:
    """Build and compile the didactic LangGraph workflow."""

    corpus = load_demo_corpus(corpus_path)

    def load_workspace_node(state: DemoState) -> DemoState:
        """Load the tiny demo workspace and explain what enters shared state."""

        workspace = load_demo_workspace()
        tracer.node(
            "load_workspace",
            "start",
            "Inicializando el workspace mínimo de la demo.",
            {"topic": state["topic"]},
        )
        next_state: DemoState = {
            "workspace": workspace.model_dump(mode="json"),
            "artifacts": artifacts.model_dump(mode="json"),
            "handoffs": [],
            "revision_count": 0,
            "needs_revision": False,
        }
        next_handoffs = append_handoff(
            {**state, **next_state},
            tracer,
            DemoHandoff(
                from_agent="load_workspace",
                to_agent="research",
                purpose="Preparar el contexto común antes de leer noticias.",
                inputs_used=["topic"],
                outputs_written=["workspace", "artifacts", "revision_count"],
                summary="Workspace cargado y contexto compartido preparado para research.",
            ),
        )
        next_state["handoffs"] = next_handoffs
        tracer.node(
            "load_workspace",
            "end",
            "Workspace cargado. La organización ya conoce el tema, límites y política de revisión.",
            summarize_state({**state, **next_state}),
        )
        tracer.edge(
            "load_workspace",
            "research",
            (
                "Transición lineal: con el workspace listo, la organización "
                "pasa a recopilar evidencia."
            ),
        )
        return next_state

    def research_node(state: DemoState) -> DemoState:
        """Select and normalize local corpus items that match the user topic."""

        workspace = DemoWorkspace.model_validate(state["workspace"])
        tracer.node(
            "research",
            "start",
            "Seleccionando evidencia del corpus local y dejándola en un formato homogéneo.",
            {"topic": state["topic"], "corpus_items": len(corpus)},
        )
        matched_items = [item for item in corpus if topic_matches_item(state["topic"], item)]
        limited_items = matched_items[: workspace.max_research_items]
        raw_items = [normalize_research_item(item) for item in limited_items]
        next_handoffs = append_handoff(
            state,
            tracer,
            DemoHandoff(
                from_agent="research",
                to_agent="curate",
                purpose="Entregar evidencia comparable para agrupar historias.",
                inputs_used=["topic", "workspace"],
                outputs_written=["raw_items"],
                summary="Research deja artículos homogéneos y listos para curación.",
            ),
        )
        tracer.node(
            "research",
            "end",
            "Research completado. El estado ya contiene evidencia local normalizada.",
            {"raw_titles": [item["title"] for item in raw_items]},
        )
        tracer.edge(
            "research",
            "curate",
            "La evidencia ya es comparable; el siguiente agente puede agruparla por historia.",
        )
        return {"raw_items": raw_items, "handoffs": next_handoffs}

    def curate_node(state: DemoState) -> DemoState:
        """Group research items into a smaller set of didactic story briefs."""

        workspace = DemoWorkspace.model_validate(state["workspace"])
        raw_items = state.get("raw_items", [])
        tracer.node(
            "curate",
            "start",
            "Agrupando artículos en historias observables para reducir complejidad narrativa.",
            {"raw_items": len(raw_items)},
        )
        story_briefs = build_story_briefs(raw_items)[: workspace.max_story_briefs]
        next_handoffs = append_handoff(
            state,
            tracer,
            DemoHandoff(
                from_agent="curate",
                to_agent="verify",
                purpose="Pasar historias agrupadas y priorizadas al verificador.",
                inputs_used=["raw_items"],
                outputs_written=["story_briefs"],
                summary="Curate reduce el ruido y entrega historias agrupadas al verificador.",
            ),
        )
        tracer.node(
            "curate",
            "end",
            "Curación completada. El estado ya expresa historias, no artículos sueltos.",
            {
                "story_ids": [story["story_id"] for story in story_briefs],
                "story_sizes": [len(story["items"]) for story in story_briefs],
            },
        )
        tracer.edge(
            "curate",
            "verify",
            "Las historias agrupadas se transfieren al verificador para juicio semántico.",
        )
        return {"story_briefs": story_briefs, "handoffs": next_handoffs}

    def verify_node(state: DemoState) -> DemoState:
        """Verify the most relevant story briefs with a real LLM call."""

        story_briefs = state.get("story_briefs", [])
        workspace = DemoWorkspace.model_validate(state["workspace"])
        tracer.node(
            "verify",
            "start",
            "Verificando historias con LLM real para añadir conclusiones y cautelas explícitas.",
            {
                "story_briefs": len(story_briefs),
                "max_written_stories": workspace.max_written_stories,
            },
        )
        verifications: list[dict[str, Any]] = []
        for story_brief in story_briefs[: workspace.max_written_stories]:
            response = asyncio.run(
                llm_client.complete_json(
                    system_prompt=(
                        "Eres un agente verificador didáctico. Devuelve JSON con "
                        "conclusion, claims y caution_notes. Resume si varias fuentes "
                        "hablan de la misma historia y qué matices conviene no exagerar."
                    ),
                    user_prompt=build_verify_prompt(story_brief, state["topic"]),
                    response_model=DemoVerificationPayload,
                    temperature=0.1,
                )
            )
            verification = {
                "story_id": story_brief["story_id"],
                "conclusion": response.conclusion,
                "claims": response.model_dump(mode="json")["claims"],
                "caution_notes": response.caution_notes,
                "independent_source_count": story_brief["independent_source_count"],
                "confidence_score": min(
                    0.45 + (0.2 * max(story_brief["independent_source_count"] - 1, 0)),
                    0.95,
                ),
            }
            tracer.node(
                "verify",
                "story",
                "Historia verificada. El juicio del LLM queda externalizado en el estado.",
                {
                    "story_id": verification["story_id"],
                    "confidence_score": verification["confidence_score"],
                },
            )
            verifications.append(verification)
        next_handoffs = append_handoff(
            state,
            tracer,
            DemoHandoff(
                from_agent="verify",
                to_agent="write",
                purpose="Entregar conclusiones verificadas al redactor.",
                inputs_used=["story_briefs"],
                outputs_written=["verifications"],
                summary="Verify añade juicio y cautelas antes de la redacción.",
            ),
        )
        tracer.node(
            "verify",
            "end",
            "Verificación completada. El writer ya puede redactar con soporte explícito.",
            {"verified_stories": len(verifications)},
        )
        tracer.edge(
            "verify",
            "write",
            "Con la evidencia verificada, el writer puede sintetizar un borrador.",
        )
        return {"verifications": verifications, "handoffs": next_handoffs}

    def write_node(state: DemoState) -> DemoState:
        """Draft a concise didactic report from verified stories."""

        story_briefs = state.get("story_briefs", [])
        verifications = state.get("verifications", [])
        review_feedback = state.get("review_feedback")
        revision_count = state.get("revision_count", 0)
        tracer.node(
            "write",
            "start",
            "Redactando un borrador corto y trazable a partir del estado verificado.",
            {"verified_stories": len(verifications), "revision_count": revision_count},
        )
        if not verifications:
            draft = build_no_evidence_draft(state["topic"])
        else:
            response = asyncio.run(
                llm_client.complete_json(
                    system_prompt=(
                        "Eres un agente escritor didáctico. Devuelve JSON con headline, "
                        "summary, bullet_points y closing_note. Resume el flujo con "
                        "tono claro, menciona cautelas y respeta las instrucciones del revisor."
                    ),
                    user_prompt=build_write_prompt(
                        state["topic"],
                        story_briefs[: len(verifications)],
                        verifications,
                        review_feedback,
                        revision_count,
                    ),
                    response_model=DemoDraftPayload,
                    temperature=0.2,
                )
            )
            draft = response.model_dump(mode="json")
        next_handoffs = append_handoff(
            state,
            tracer,
            DemoHandoff(
                from_agent="write",
                to_agent="review",
                purpose="Enviar un borrador legible para control de calidad.",
                inputs_used=["verifications", "review_feedback"],
                outputs_written=["draft"],
                summary="Write entrega un borrador que el revisor puede aprobar o devolver.",
            ),
        )
        tracer.node(
            "write",
            "end",
            "Borrador completado. La siguiente transición depende del revisor.",
            {"headline": draft["headline"], "bullet_points": len(draft["bullet_points"])},
        )
        tracer.edge(
            "write",
            "review",
            "El borrador sale del writer y pasa al control de calidad.",
        )
        return {"draft": draft, "handoffs": next_handoffs, "needs_revision": False}

    def review_node(state: DemoState) -> DemoState:
        """Review the draft and decide whether the graph should loop once."""

        draft = state["draft"]
        verifications = state.get("verifications", [])
        workspace = DemoWorkspace.model_validate(state["workspace"])
        revision_count = state.get("revision_count", 0)
        tracer.node(
            "review",
            "start",
            "Revisando el borrador para decidir si la organización debe volver a escribir.",
            {"revision_count": revision_count, "verifications": len(verifications)},
        )
        response = asyncio.run(
            llm_client.complete_json(
                system_prompt=(
                    "Eres un revisor didáctico. Devuelve JSON con approved, feedback_summary, "
                    "revision_instructions y strengths. Evalúa claridad, trazabilidad y ajuste "
                    "a las cautelas del verificador."
                ),
                user_prompt=build_review_prompt(
                    state["topic"],
                    draft,
                    verifications,
                    revision_count,
                ),
                response_model=DemoReviewPayload,
                temperature=0.1,
            )
        )
        force_first_revision = workspace.force_first_revision and revision_count < 1
        needs_revision = force_first_revision or (
            not response.approved and revision_count < workspace.max_revisions
        )
        next_revision_count = revision_count + 1 if needs_revision else revision_count
        review_feedback = {
            "approved": not needs_revision,
            "needs_revision": needs_revision,
            "feedback_summary": response.feedback_summary,
            "revision_instructions": response.revision_instructions,
            "strengths": response.strengths,
            "forced_for_demo": force_first_revision,
        }
        next_target = "write" if needs_revision else "render"
        next_handoffs = append_handoff(
            state,
            tracer,
            DemoHandoff(
                from_agent="review",
                to_agent=next_target,
                purpose=(
                    "Solicitar una nueva redacción."
                    if needs_revision
                    else "Aprobar el borrador y permitir su salida final."
                ),
                inputs_used=["draft", "verifications"],
                outputs_written=["review_feedback", "needs_revision", "revision_count"],
                summary=(
                    "Review devuelve feedback al writer para una iteración adicional."
                    if needs_revision
                    else "Review aprueba el borrador y habilita la salida final."
                ),
            ),
        )
        tracer.node(
            "review",
            "end",
            "Revisión completada. La siguiente arista depende de la decisión del revisor.",
            {
                "needs_revision": needs_revision,
                "revision_count": next_revision_count,
                "forced_for_demo": force_first_revision,
            },
        )
        return {
            "review_feedback": review_feedback,
            "needs_revision": needs_revision,
            "revision_count": next_revision_count,
            "handoffs": next_handoffs,
        }

    def render_node(state: DemoState) -> DemoState:
        """Render the final Markdown report and point to generated artifacts."""

        tracer.node(
            "render",
            "start",
            "Persistiendo el informe final y cerrando la ejecución observable.",
            {"run_dir": artifacts.run_dir},
        )
        report_md = render_markdown_report(state)
        Path(artifacts.report_md).write_text(report_md, encoding="utf-8")
        tracer.node(
            "render",
            "end",
            "Render completado. Los artefactos y la traza ya están fuera del grafo.",
            {
                "report_md": artifacts.report_md,
                "events_jsonl": artifacts.events_jsonl,
            },
        )
        tracer.edge(
            "render",
            "END",
            "La organización termina tras persistir el informe y la traza completa.",
        )
        return {"final_report": report_md}

    def route_after_review(state: DemoState) -> str:
        """Choose the next node after review and trace the taken edge."""

        if state.get("needs_revision", False):
            tracer.edge(
                "review",
                "write",
                "Arista condicional tomada: el revisor pide una única reescritura visible.",
                {"revision_count": state.get("revision_count", 0)},
            )
            return "write"
        tracer.edge(
            "review",
            "render",
            "Arista condicional tomada: el revisor aprueba el borrador.",
            {"revision_count": state.get("revision_count", 0)},
        )
        return "render"

    graph_builder = StateGraph(DemoState)
    graph_builder.add_node("load_workspace", load_workspace_node)
    graph_builder.add_node("research", research_node)
    graph_builder.add_node("curate", curate_node)
    graph_builder.add_node("verify", verify_node)
    graph_builder.add_node("write", write_node)
    graph_builder.add_node("review", review_node)
    graph_builder.add_node("render", render_node)
    graph_builder.add_edge(START, "load_workspace")
    graph_builder.add_edge("load_workspace", "research")
    graph_builder.add_edge("research", "curate")
    graph_builder.add_edge("curate", "verify")
    graph_builder.add_edge("verify", "write")
    graph_builder.add_edge("write", "review")
    graph_builder.add_conditional_edges(
        "review",
        route_after_review,
        {
            "write": "write",
            "render": "render",
        },
    )
    graph_builder.add_edge("render", END)
    return graph_builder.compile(checkpointer=checkpointer, name="demo_langgraph_workflow")


def build_default_llm_client(env_path: Path) -> DemoOpenRouterClient:
    """Create the demo OpenRouter client from the repository environment."""

    return DemoOpenRouterClient(env_path=str(env_path))
