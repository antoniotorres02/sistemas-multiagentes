"""Graph builder for the didactic LangGraph demo."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from news_system_demo.llm import LlmClientProtocol, OpenRouterClient
from news_system_demo.models import RunArtifactPaths, State
from news_system_demo.nodes import (
    curate_node,
    load_workspace_node,
    render_node,
    research_node,
    review_node,
    route_after_review,
    verify_node,
    write_node,
)
from news_system_demo.nodes.shared import load_corpus
from news_system_demo.runtime import RunLogger


def build_demo_graph(
    *,
    corpus_path: Path,
    artifacts: RunArtifactPaths,
    logger: RunLogger,
    llm_client: LlmClientProtocol,
) -> Any:
    """Build and compile the didactic LangGraph workflow."""

    corpus = load_corpus(corpus_path)
    graph_builder = StateGraph(State)
    graph_builder.add_node("load_workspace", partial(load_workspace_node, logger=logger))
    graph_builder.add_node("research", partial(research_node, corpus=corpus, logger=logger))
    graph_builder.add_node("curate", partial(curate_node, logger=logger))
    graph_builder.add_node("verify", partial(verify_node, llm_client=llm_client, logger=logger))
    graph_builder.add_node("write", partial(write_node, llm_client=llm_client, logger=logger))
    graph_builder.add_node("review", partial(review_node, llm_client=llm_client, logger=logger))
    graph_builder.add_node("render", partial(render_node, artifacts=artifacts, logger=logger))

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
    return graph_builder.compile(name="news_system_workflow")


def build_default_llm_client(env_path: Path) -> OpenRouterClient:
    """Create the OpenRouter client from the repository environment."""

    return OpenRouterClient(env_path=str(env_path))
