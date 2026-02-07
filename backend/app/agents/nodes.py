"""Agent nodes for the RAG workflow."""

import re
import anthropic

from ..config import get_settings
from ..retrieval import HybridRetriever
from ..ingestion.embedder import COMPONENT_PATTERNS


def parse_query(state: dict) -> dict:
    """
    Parse the user query to extract intent and component IDs.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with parsed query info.
    """
    query = state["query"]
    
    # Extract component IDs from query
    component_ids = []
    for pattern in COMPONENT_PATTERNS:
        matches = re.findall(pattern, query, re.IGNORECASE)
        component_ids.extend([m.upper() for m in matches])
    
    component_ids = list(set(component_ids))
    
    # Determine if cross-reference is needed
    cross_ref_keywords = [
        "wired", "connected", "wiring", "electrical",
        "located", "location", "where",
        "how", "trace", "connection",
    ]
    needs_cross_ref = any(kw in query.lower() for kw in cross_ref_keywords)
    needs_cross_ref = needs_cross_ref and len(component_ids) > 0
    
    return {
        **state,
        "parsed_query": query,
        "component_ids": component_ids,
        "needs_cross_reference": needs_cross_ref,
    }


def retrieve_documents(state: dict) -> dict:
    """
    Retrieve relevant documents using hybrid search.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with retrieved citations.
    """
    retriever = HybridRetriever()
    
    citations = retriever.search(
        query=state["parsed_query"],
        limit=state["max_citations"] * 2,  # Get extra for filtering
        component_ids=state["component_ids"] if state["component_ids"] else None,
    )
    
    return {
        **state,
        "citations": citations,
    }


def cross_reference(state: dict) -> dict:
    """
    Find components across different document types.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with cross-reference information.
    """
    retriever = HybridRetriever()
    
    cross_refs = {}
    for component_id in state["component_ids"]:
        refs = retriever.cross_document_search(component_id)
        if refs:
            cross_refs[component_id] = refs
    
    # Add cross-reference citations to the main list
    additional_citations = []
    for component_id, doc_type_refs in cross_refs.items():
        for doc_type, citations in doc_type_refs.items():
            additional_citations.extend(citations[:2])  # Top 2 per doc type
    
    return {
        **state,
        "cross_references": cross_refs,
        "citations": state["citations"] + additional_citations,
    }


def generate_response(state: dict) -> dict:
    """
    Generate a response using Claude with strict grounding.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with draft response.
    """
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    
    # Build context from citations
    context_parts = []
    for i, citation in enumerate(state["citations"][:10]):
        context_parts.append(
            f"[Source {i+1}] {citation.filename}, Page {citation.page_number} "
            f"({citation.document_type}):\n{citation.snippet}\n"
        )
    
    context = "\n".join(context_parts)
    
    # Build cross-reference summary
    cross_ref_summary = ""
    if state["cross_references"]:
        cross_ref_parts = []
        for comp_id, refs in state["cross_references"].items():
            doc_types = list(refs.keys())
            cross_ref_parts.append(f"- {comp_id} found in: {', '.join(doc_types)}")
        cross_ref_summary = "\nCross-document references:\n" + "\n".join(cross_ref_parts)
    
    prompt = f"""You are a technical expert assistant analyzing engineering documentation. 
Answer the user's question based ONLY on the provided source material.

CRITICAL RULES:
1. ONLY use information explicitly stated in the sources
2. Cite sources using [Source N] format for every factual claim
3. If information is not available, say "Information not available in uploaded documents"
4. Never infer, speculate, or use outside knowledge
5. For component locations/connections, specify the exact document and page

User Question: {state["parsed_query"]}

Available Sources:
{context}
{cross_ref_summary}

Provide a clear, grounded answer:"""

    response = client.messages.create(
        model=settings.vlm_model,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    
    draft = response.content[0].text
    
    return {
        **state,
        "draft_response": draft,
    }


def verify_response(state: dict) -> dict:
    """
    Verify the response is grounded in sources.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with verification results.
    """
    draft = state["draft_response"]
    citations = state["citations"]
    
    # Simple verification: check for source citations and "not available" patterns
    has_citations = "[Source" in draft
    admits_limitations = "not available" in draft.lower() or "not found" in draft.lower()
    
    # Check for potential hallucination indicators
    hallucination_indicators = [
        "I believe", "I think", "probably", "might be", "could be",
        "typically", "usually", "generally", "in my experience",
    ]
    
    has_hedging = any(ind in draft.lower() for ind in hallucination_indicators)
    
    # Calculate confidence
    confidence = 0.8  # Base confidence
    if has_citations:
        confidence += 0.1
    if has_hedging:
        confidence -= 0.3
    if not citations:
        confidence = 0.2
    
    confidence = max(0.0, min(1.0, confidence))
    
    grounded = has_citations or admits_limitations
    grounded = grounded and not has_hedging
    
    return {
        **state,
        "final_response": draft,
        "grounded": grounded,
        "confidence": confidence,
    }


def should_retry(state: dict) -> str:
    """
    Determine if the query should be retried.
    
    Args:
        state: Current agent state.
        
    Returns:
        "retry" or "end"
    """
    # Retry conditions
    if state["retry_count"] >= 2:
        return "end"
    
    if not state["grounded"] and state["citations"]:
        # We have sources but response isn't grounded - try again
        state["retry_count"] = state.get("retry_count", 0) + 1
        return "retry"
    
    return "end"
