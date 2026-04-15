"""Graph builder for the didactic LangGraph demo."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from news_system_demo.llm import DemoLlmClientProtocol, DemoOpenRouterClient
from news_system_demo.models import DemoRunArtifactPaths, DemoState
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
from news_system_demo.nodes.shared import load_demo_corpus
from news_system_demo.runtime import DemoTracer


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

    graph_builder = StateGraph(DemoState)
    graph_builder.add_node(
        "load_workspace",
        partial(load_workspace_node, artifacts=artifacts, tracer=tracer),
    )
    graph_builder.add_node("research", partial(research_node, corpus=corpus, tracer=tracer))
    graph_builder.add_node("curate", partial(curate_node, tracer=tracer))
    graph_builder.add_node("verify", partial(verify_node, llm_client=llm_client, tracer=tracer))
    graph_builder.add_node("write", partial(write_node, llm_client=llm_client, tracer=tracer))
    graph_builder.add_node("review", partial(review_node, llm_client=llm_client, tracer=tracer))
    graph_builder.add_node("render", partial(render_node, artifacts=artifacts, tracer=tracer))
    graph_builder.add_edge(START, "load_workspace")
    graph_builder.add_edge("load_workspace", "research")
    graph_builder.add_edge("research", "curate")
    graph_builder.add_edge("curate", "verify")
    graph_builder.add_edge("verify", "write")
    graph_builder.add_edge("write", "review")
    graph_builder.add_conditional_edges(
        "review",
        partial(route_after_review, tracer=tracer),
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
