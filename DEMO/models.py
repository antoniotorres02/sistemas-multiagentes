"""Typed models used by the didactic LangGraph demo."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


def coerce_text(value: Any) -> str:
    """Coerce nested model output into a readable plain string."""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        preferred_keys = (
            "text",
            "claim",
            "conclusion",
            "summary",
            "headline",
            "note",
            "message",
            "content",
            "value",
            "label",
            "feedback_summary",
        )
        for key in preferred_keys:
            if key in value:
                return coerce_text(value[key])
        flattened = [coerce_text(item) for item in value.values()]
        return " | ".join(part for part in flattened if part)
    if isinstance(value, list):
        flattened = [coerce_text(item) for item in value]
        return " | ".join(part for part in flattened if part)
    return ""


def coerce_text_list(value: Any) -> list[str]:
    """Coerce arbitrary nested values into a flat list of readable strings."""

    if value is None:
        return []
    if isinstance(value, list):
        collected: list[str] = []
        for item in value:
            if isinstance(item, list):
                collected.extend(coerce_text_list(item))
                continue
            text = coerce_text(item)
            if text:
                collected.append(text)
        return collected
    text = coerce_text(value)
    return [text] if text else []


class DemoCorpusItem(BaseModel):
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


class DemoWorkspace(BaseModel):
    """Small configuration object loaded by the demo workflow."""

    language: str = "es"
    tone: str = "didactico"
    max_research_items: int = 5
    max_story_briefs: int = 3
    max_written_stories: int = 2
    max_revisions: int = 1
    force_first_revision: bool = True
    available_topics: list[str] = Field(default_factory=list)


class DemoClaim(BaseModel):
    """Claim extracted or synthesized during the verify step."""

    text: str
    supporting_points: list[str] = Field(default_factory=list)

    @field_validator("text", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        """Normalize claim text coming from slightly irregular LLM payloads."""

        if isinstance(value, dict):
            for key in ("text", "claim", "afirmacion", "afirmación", "statement"):
                if key in value:
                    return coerce_text(value[key])
        return coerce_text(value)

    @field_validator("supporting_points", mode="before")
    @classmethod
    def normalize_supporting_points(cls, value: Any) -> list[str]:
        """Normalize supporting points into a plain string list."""

        return coerce_text_list(value)


class DemoVerificationPayload(BaseModel):
    """Structured verification response produced by the LLM."""

    conclusion: str
    claims: list[DemoClaim] = Field(default_factory=list)
    caution_notes: list[str] = Field(default_factory=list)

    @field_validator("conclusion", mode="before")
    @classmethod
    def normalize_conclusion(cls, value: Any) -> str:
        """Normalize a possibly nested conclusion field."""

        return coerce_text(value)

    @field_validator("claims", mode="before")
    @classmethod
    def normalize_claims(cls, value: Any) -> list[Any]:
        """Normalize claims so each entry exposes a `text` field."""

        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        normalized: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                text = ""
                for key in ("text", "claim", "afirmacion", "afirmación", "statement"):
                    if key in item:
                        text = coerce_text(item[key])
                        break
                if not text:
                    text = coerce_text(item)
                supporting_points = item.get("supporting_points") or item.get("points")
                normalized.append(
                    {
                        "text": text,
                        "supporting_points": coerce_text_list(supporting_points),
                    }
                )
                continue
            normalized.append({"text": coerce_text(item), "supporting_points": []})
        return normalized

    @field_validator("caution_notes", mode="before")
    @classmethod
    def normalize_caution_notes(cls, value: Any) -> list[str]:
        """Normalize caution notes into readable flat strings."""

        return coerce_text_list(value)


class DemoDraftPayload(BaseModel):
    """Structured writing response produced by the writer agent."""

    headline: str
    summary: str
    bullet_points: list[str] = Field(default_factory=list)
    closing_note: str

    @field_validator("headline", "summary", "closing_note", mode="before")
    @classmethod
    def normalize_text_field(cls, value: Any) -> str:
        """Normalize top-level textual fields returned by the writer agent."""

        return coerce_text(value)

    @field_validator("bullet_points", mode="before")
    @classmethod
    def normalize_bullet_points(cls, value: Any) -> list[str]:
        """Normalize bullets into a flat readable string list."""

        return coerce_text_list(value)


class DemoReviewPayload(BaseModel):
    """Structured review response produced by the reviewer agent."""

    approved: bool
    feedback_summary: str
    revision_instructions: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)

    @field_validator("feedback_summary", mode="before")
    @classmethod
    def normalize_feedback_summary(cls, value: Any) -> str:
        """Normalize the review summary into plain text."""

        return coerce_text(value)

    @field_validator("revision_instructions", "strengths", mode="before")
    @classmethod
    def normalize_string_list(cls, value: Any) -> list[str]:
        """Normalize review list fields into readable flat strings."""

        return coerce_text_list(value)


class DemoHandoff(BaseModel):
    """Explicit message exchanged between two agents through shared state."""

    from_agent: str
    to_agent: str
    purpose: str
    inputs_used: list[str] = Field(default_factory=list)
    outputs_written: list[str] = Field(default_factory=list)
    summary: str


class DemoRunArtifactPaths(BaseModel):
    """Artifact paths generated for one demo run."""

    run_dir: str
    events_jsonl: str
    graph_mermaid: str
    report_md: str
    state_history_json: str


class DemoTraceEvent(BaseModel):
    """One persisted trace event emitted by the demo runtime."""

    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_kind: Literal["node", "edge", "handoff"]
    agent: str
    phase: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class DemoState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes in the demo."""

    topic: str
    thread_id: str
    workspace: dict[str, Any]
    raw_items: list[dict[str, Any]]
    story_briefs: list[dict[str, Any]]
    verifications: list[dict[str, Any]]
    draft: dict[str, Any]
    review_feedback: dict[str, Any]
    revision_count: int
    needs_revision: bool
    handoffs: list[dict[str, Any]]
    final_report: str
    artifacts: dict[str, str]


class DemoRuntimePaths(BaseModel):
    """Filesystem layout used by the demo runtime."""

    root_dir: Path
    corpus_path: Path
    data_dir: Path
    runs_dir: Path
    checkpoint_db_path: Path
    env_path: Path
