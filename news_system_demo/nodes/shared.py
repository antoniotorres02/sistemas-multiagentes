"""Shared helpers used by the demo nodes."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from news_system_demo.models import CorpusItem, State, Workspace

MIN_MATCH_SCORE = 15

STOPWORDS = {
    "about",
    "ante",
    "como",
    "con",
    "del",
    "desde",
    "for",
    "las",
    "los",
    "para",
    "por",
    "que",
    "sobre",
    "the",
    "una",
    "unos",
}


def _normalize_search_text(text: str) -> str:
    """Lowercase and strip accents for forgiving demo searches."""

    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def load_workspace() -> Workspace:
    """Return the fixed demo workspace configuration."""

    return Workspace(
        available_topics=[
            "ai regulation",
            "ai in health",
            "energy transition",
            "cybersecurity",
            "digital policy",
            "climate infrastructure",
            "media and platforms",
            "european policy",
        ]
    )


def load_corpus(corpus_path: Path) -> list[CorpusItem]:
    """Load the local demo corpus from JSON."""

    raw_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    return [CorpusItem.model_validate(item) for item in raw_payload]


def _tokenize(text: str) -> list[str]:
    """Return compact lowercase search terms for deterministic ranking."""

    return [
        term
        for term in re.findall(r"[\w]+", _normalize_search_text(text), flags=re.UNICODE)
        if len(term) >= 3 and term not in STOPWORDS
    ]


def score_item_for_topic(topic: str, item: CorpusItem) -> tuple[int, list[str]]:
    """Score one corpus item and explain why it matches a user topic."""

    normalized_topic = _normalize_search_text(topic)
    topic_terms = set(_tokenize(topic))
    semantic_topic = _normalize_search_text(item.semantic_topic)
    semantic_terms = set(_tokenize(item.semantic_topic))
    tag_terms = {term for tag in item.tags for term in _tokenize(tag)}
    entity_terms = {_normalize_search_text(entity) for entity in item.entities}
    title_terms = set(_tokenize(item.title))
    summary_terms = set(_tokenize(item.summary))
    body_terms = set(_tokenize(item.body))
    source_terms = set(_tokenize(item.source_name))
    region_terms = set(_tokenize(item.region))

    score = 0
    reasons: list[str] = []

    if semantic_topic and semantic_topic in normalized_topic:
        score += 60
        reasons.append(f"tema directo: {item.semantic_topic}")

    semantic_matches = sorted(topic_terms & semantic_terms)
    if semantic_matches:
        score += 16 * len(semantic_matches)
        reasons.append(f"tema cercano: {', '.join(semantic_matches)}")

    tag_matches = sorted(topic_terms & tag_terms)
    if tag_matches:
        score += 10 * len(tag_matches)
        reasons.append(f"etiquetas: {', '.join(tag_matches[:4])}")

    entity_matches = sorted(entity for entity in entity_terms if entity in normalized_topic)
    if entity_matches:
        score += 18 * len(entity_matches)
        reasons.append(f"entidades: {', '.join(entity_matches[:3])}")

    title_matches = sorted(topic_terms & title_terms)
    if title_matches:
        score += 9 * len(title_matches)
        reasons.append(f"titular: {', '.join(title_matches[:4])}")

    summary_matches = sorted(topic_terms & summary_terms)
    if summary_matches:
        score += 5 * len(summary_matches)

    body_matches = sorted(topic_terms & body_terms)
    if body_matches:
        score += 2 * len(body_matches)

    source_matches = sorted(topic_terms & source_terms)
    if source_matches:
        score += 4 * len(source_matches)

    region_matches = sorted(topic_terms & region_terms)
    if region_matches:
        score += 6 * len(region_matches)
        reasons.append(f"region: {item.region}")

    if score:
        score += max(1, min(item.importance, 5)) * 2
        if item.story_angle:
            reasons.append(f"angulo: {item.story_angle}")

    return score, reasons[:5]


def topic_matches_item(topic: str, item: CorpusItem) -> bool:
    """Return whether a user topic matches one corpus item."""

    score, _ = score_item_for_topic(topic, item)
    return score >= MIN_MATCH_SCORE


def normalize_item(
    item: CorpusItem,
    *,
    match_score: int = 0,
    match_reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Normalize one corpus item into the small shape passed between nodes."""

    return {
        "title": item.title,
        "summary": item.summary,
        "source_name": item.source_name,
        "url": item.url,
        "published_at": item.published_at.isoformat(),
        "semantic_topic": item.semantic_topic,
        "source_kind": item.source_kind,
        "region": item.region,
        "importance": item.importance,
        "entities": list(item.entities),
        "story_angle": item.story_angle,
        "match_score": match_score,
        "match_reasons": list(match_reasons or []),
        "content_excerpt": item.body[:420],
    }


def curate_items(items: Sequence[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    """Keep a compact, source-diverse selection for the writer."""

    if not items:
        return []

    scored_items = [item for item in items if item.get("match_score")]
    if scored_items:
        top_score = int(scored_items[0].get("match_score", 0))
        editorial_floor = max(MIN_MATCH_SCORE, int(top_score * 0.5))
        items = [item for item in items if int(item.get("match_score", 0)) >= editorial_floor]

    seen_topics: set[str] = set()
    seen_sources: set[str] = set()
    curated: list[dict[str, Any]] = []
    for item in items:
        topic = str(item.get("semantic_topic", ""))
        source = str(item.get("source_name", ""))
        if topic in seen_topics and source in seen_sources and len(curated) < len(items):
            continue
        curated.append(dict(item))
        seen_topics.add(topic)
        seen_sources.add(source)
        if len(curated) >= limit:
            break
    if len(curated) < limit:
        selected_ids = {item.get("title") for item in curated}
        for item in items:
            if item.get("title") in selected_ids:
                continue
            curated.append(dict(item))
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
                f"  Rol editorial: {item.get('editorial_role', 'evidencia')}",
                f"  Resumen: {item['summary']}",
                f"  Angulo: {item.get('story_angle', '')}",
                "  Relevancia editorial: "
                f"{item.get('match_score', 0)} ({'; '.join(item.get('match_reasons', []))})",
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
        sections.append(
            f"- {item['source_name']}: {item['title']} ({item['url']}). "
            f"Rol: {item.get('editorial_role', 'evidencia')}. "
            f"Angulo: {item.get('story_angle', 'contexto')}"
        )
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
