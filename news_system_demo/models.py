"""Typed models used by the didactic LangGraph demo."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel, Field, field_validator


def coerce_text(value: Any) -> str:
    """Coerce nested model output into a readable plain string."""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("text", "article_text", "evidence_note", "note", "summary", "content", "value"):
            if key in value:
                return coerce_text(value[key])
        return " | ".join(part for part in (coerce_text(item) for item in value.values()) if part)
    if isinstance(value, list):
        return " | ".join(part for part in (coerce_text(item) for item in value) if part)
    return ""


class CorpusItem(BaseModel):
    """One local news-like item used by the demo as deterministic input."""

    item_id: str
    title: str
    summary: str
    body: str
    source_name: str
    url: str
    published_at: datetime
    tags: list[str] = Field(default_factory=list)
    semantic_topic: str


class Workspace(BaseModel):
    """Small configuration object loaded by the demo workflow."""

    max_research_items: int = 5
    max_story_briefs: int = 3
    max_written_stories: int = 2
    max_revisions: int = 1
    force_first_revision: bool = False
    available_topics: list[str] = Field(default_factory=list)


class EvidencePayload(BaseModel):
    """Short evidence note produced by the verification step."""

    evidence_note: str

    @field_validator("evidence_note", mode="before")
    @classmethod
    def normalize_evidence_note(cls, value: Any) -> str:
        """Normalize the note returned by the verifier."""

        return coerce_text(value)


class ArticlePayload(BaseModel):
    """Final article draft produced by the writer step."""

    article_text: str

    @field_validator("article_text", mode="before")
    @classmethod
    def normalize_article_text(cls, value: Any) -> str:
        """Normalize the article body returned by the writer."""

        return coerce_text(value)


class ReviewPayload(BaseModel):
    """Small review response produced by the reviewer step."""

    approved: bool
    note: str

    @field_validator("note", mode="before")
    @classmethod
    def normalize_note(cls, value: Any) -> str:
        """Normalize the review note into plain text."""

        return coerce_text(value)


class RunArtifactPaths(BaseModel):
    """Artifact paths generated for one demo run."""

    run_dir: str
    report_md: str
    run_log: str


class State(TypedDict, total=False):
    """Shared state passed between LangGraph nodes in the demo."""

    topic: str
    thread_id: str
    selected_items: list[dict[str, Any]]
    evidence_note: str
    article_text: str
    review_note: str
    revision_count: int
    needs_revision: bool
    report_path: str


class RuntimePaths(BaseModel):
    """Filesystem layout used by the demo runtime."""

    root_dir: Path
    corpus_path: Path
    runs_dir: Path
    env_path: Path
