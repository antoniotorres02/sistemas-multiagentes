"""Shared helpers used by the demo nodes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from news_system_demo.models import DemoCorpusItem, DemoHandoff, DemoState, DemoWorkspace
from news_system_demo.runtime import DemoTracer


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
    """Normalize one corpus item into the shape used by downstream nodes."""

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
    """Build the verification prompt for one story brief."""

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
        for instruction in review_feedback["revision_instructions"]:
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
    for bullet_point in draft.get("bullet_points", []):
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
    for bullet_point in draft["bullet_points"]:
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
    next_handoffs.append(handoff.model_dump(mode="json"))
    tracer.handoff(handoff)
    return next_handoffs
