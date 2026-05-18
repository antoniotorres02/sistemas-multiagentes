"""Shared helpers used by the demo nodes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from news_system_demo.models import CorpusItem, State, Workspace


def load_workspace() -> Workspace:
    """Return the fixed demo workspace configuration."""

    return Workspace(
        available_topics=[
            "ai regulation",
            "ai in health",
            "energy transition",
            "european policy",
        ]
    )


def load_corpus(corpus_path: Path) -> list[CorpusItem]:
    """Load the local demo corpus from JSON."""

    raw_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    return [CorpusItem.model_validate(item) for item in raw_payload]


def topic_matches_item(topic: str, item: CorpusItem) -> bool:
    """Return whether a user topic matches one corpus item."""

    normalized_topic = topic.lower()
    semantic_topic = item.semantic_topic.lower()
    if semantic_topic and semantic_topic in normalized_topic:
        return True
    haystack = " ".join(
        [
            item.title.lower(),
            item.summary.lower(),
            item.body.lower(),
            semantic_topic,
            " ".join(tag.lower() for tag in item.tags),
        ]
    )
    topic_terms = [term for term in normalized_topic.split() if len(term) >= 3]
    if not topic_terms:
        return False
    match_count = sum(1 for term in topic_terms if term in haystack)
    if len(topic_terms) <= 2:
        return match_count == len(topic_terms)
    return match_count >= 2


def normalize_item(item: CorpusItem) -> dict[str, Any]:
    """Normalize one corpus item into the small shape passed between nodes."""

    return {
        "title": item.title,
        "summary": item.summary,
        "source_name": item.source_name,
        "url": item.url,
        "published_at": item.published_at.isoformat(),
        "semantic_topic": item.semantic_topic,
        "content_excerpt": item.body[:420],
    }


def curate_items(items: Sequence[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    """Keep a compact, source-diverse selection for the writer."""

    seen_topics: set[str] = set()
    curated: list[dict[str, Any]] = []
    for item in items:
        topic = str(item.get("semantic_topic", ""))
        if topic in seen_topics and len(curated) >= limit:
            continue
        curated.append(dict(item))
        seen_topics.add(topic)
        if len(curated) >= limit:
            break
    return curated


def build_evidence_prompt(topic: str, selected_items: Sequence[dict[str, Any]]) -> str:
    """Build the verifier prompt from selected evidence."""

    lines = [f"Tema del usuario: {topic}", "Evidencia seleccionada:"]
    for item in selected_items:
        lines.extend(
            [
                f"- {item['source_name']}: {item['title']}",
                f"  Resumen: {item['summary']}",
                f"  Extracto: {item['content_excerpt']}",
            ]
        )
    return "\n".join(lines)


def build_article_prompt(
    topic: str,
    selected_items: Sequence[dict[str, Any]],
    evidence_note: str,
    review_note: str | None,
    revision_count: int,
) -> str:
    """Build the writer prompt for a clean news-style Markdown article."""

    sections = [
        f"Tema del usuario: {topic}",
        f"Revision actual: {revision_count}",
        "Redacta una noticia final en Markdown, sin explicar el grafo ni el estado interno.",
        "Incluye titular, entradilla, cuerpo breve y fuentes al final.",
        f"Nota de evidencia: {evidence_note}",
    ]
    if review_note:
        sections.append(f"Nota del revisor: {review_note}")
    sections.append("Fuentes disponibles:")
    for item in selected_items:
        sections.append(f"- {item['source_name']}: {item['title']} ({item['url']})")
    return "\n".join(sections)


def build_review_prompt(topic: str, article_text: str, evidence_note: str, revision_count: int) -> str:
    """Build the reviewer prompt from the current article."""

    return "\n".join(
        [
            f"Tema: {topic}",
            f"Revision: {revision_count}",
            "Aprueba solo si la noticia es clara, está bien escrita y respeta la evidencia.",
            f"Nota de evidencia: {evidence_note}",
            "Articulo:",
            article_text,
        ]
    )


def build_no_evidence_article(topic: str) -> str:
    """Build a deterministic article when the local corpus has no matching evidence."""

    return "\n".join(
        [
            f"# No hay evidencia suficiente sobre {topic}",
            "",
            "La demo no encontró noticias relevantes en el corpus local para ese tema.",
            "",
            "Sin evidencia suficiente, el sistema evita redactar una noticia sustantiva.",
        ]
    )


def render_markdown_report(state: State) -> str:
    """Render the final clean Markdown report."""

    return state.get("article_text", "").strip() + "\n"
