"""LangGraph workflow for agentic RAG."""

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
import operator

from ..models import Citation, QueryResponse


class AgentState(TypedDict):
    """State passed between nodes in the graph."""
    # Input
    query: str
    max_citations: int
    
    # Parsed query info
    parsed_query: str
    component_ids: list[str]
    needs_cross_reference: bool
    
    # Retrieved information
    citations: Annotated[Sequence[Citation], operator.add]
    cross_references: dict[str, list[Citation]]
    
    # Generated response
    draft_response: str
    final_response: str
    grounded: bool
    confidence: float
    
    # Control flow
    retry_count: int
    error: str | None


def create_rag_graph():
    """
    Create the LangGraph workflow for agentic RAG.
    
    The workflow:
    1. Parse Query -> Extract component IDs, determine intent
    2. Retrieve -> Semantic + keyword search
    3. Cross-Reference -> If needed, find component across doc types
    4. Reason -> Generate response with Claude
    5. Verify -> Check groundedness
    6. Return or Retry
    """
    from .nodes import (
        parse_query,
        retrieve_documents,
        cross_reference,
        generate_response,
        verify_response,
        should_retry,
    )
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("parse_query", parse_query)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("cross_reference", cross_reference)
    workflow.add_node("generate", generate_response)
    workflow.add_node("verify", verify_response)
    
    # Define edges
    workflow.set_entry_point("parse_query")
    
    workflow.add_edge("parse_query", "retrieve")
    
    # Conditional edge: cross-reference if needed
    workflow.add_conditional_edges(
        "retrieve",
        lambda state: "cross_reference" if state.get("needs_cross_reference") else "generate",
        {
            "cross_reference": "cross_reference",
            "generate": "generate",
        }
    )
    
    workflow.add_edge("cross_reference", "generate")
    workflow.add_edge("generate", "verify")
    
    # Conditional edge: retry or end
    workflow.add_conditional_edges(
        "verify",
        should_retry,
        {
            "retry": "parse_query",
            "end": END,
        }
    )
    
    return workflow.compile()


def run_query(query: str, max_citations: int = 5) -> QueryResponse:
    """
    Run a query through the agentic RAG pipeline.
    
    Args:
        query: User's natural language query.
        max_citations: Maximum number of citations to include.
        
    Returns:
        QueryResponse with answer and citations.
    """
    graph = create_rag_graph()
    
    initial_state: AgentState = {
        "query": query,
        "max_citations": max_citations,
        "parsed_query": "",
        "component_ids": [],
        "needs_cross_reference": False,
        "citations": [],
        "cross_references": {},
        "draft_response": "",
        "final_response": "",
        "grounded": False,
        "confidence": 0.0,
        "retry_count": 0,
        "error": None,
    }
    
    result = graph.invoke(initial_state)
    
    return QueryResponse(
        answer=result.get("final_response", "Unable to process query."),
        citations=list(result.get("citations", []))[:max_citations],
        confidence=result.get("confidence", 0.0),
        cross_references=list(result.get("cross_references", {}).keys()),
        grounded=result.get("grounded", False),
    )
