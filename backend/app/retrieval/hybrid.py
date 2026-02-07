"""Hybrid retrieval combining semantic and keyword search."""

from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from ..config import get_settings
from ..models import DocumentType, Citation
from ..ingestion.embedder import DocumentEmbedder


class HybridRetriever:
    """Combines semantic vector search with keyword matching."""
    
    def __init__(self):
        """Initialize the hybrid retriever."""
        self.settings = get_settings()
        self.embedder = DocumentEmbedder()
        self.qdrant = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
        )
    
    def search(
        self,
        query: str,
        limit: int = 10,
        document_types: Optional[list[DocumentType]] = None,
        component_ids: Optional[list[str]] = None,
    ) -> list[Citation]:
        """
        Perform hybrid search across the document corpus.
        
        Args:
            query: Natural language query.
            limit: Maximum results to return.
            document_types: Optional filter by document types.
            component_ids: Optional filter by specific component IDs.
            
        Returns:
            List of Citation objects ranked by relevance.
        """
        # Build filter conditions
        filter_conditions = []
        
        if document_types:
            filter_conditions.append(
                FieldCondition(
                    key="document_type",
                    match=MatchAny(any=[dt.value for dt in document_types]),
                )
            )
        
        if component_ids:
            # Search for any of the specified component IDs
            filter_conditions.append(
                FieldCondition(
                    key="component_ids",
                    match=MatchAny(any=[cid.upper() for cid in component_ids]),
                )
            )
        
        # Construct filter
        search_filter = None
        if filter_conditions:
            search_filter = Filter(must=filter_conditions)
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        if not query_embedding:
            return []
        
        # Perform vector search
        results = self.qdrant.search(
            collection_name=self.settings.qdrant_collection_name,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )
        
        # Convert to Citation objects
        citations = []
        for result in results:
            payload = result.payload
            
            citation = Citation(
                document_id=payload.get("document_id", ""),
                filename=payload.get("filename", ""),
                page_number=payload.get("page_number", 0),
                document_type=DocumentType(payload.get("document_type", "unknown")),
                relevance_score=result.score,
                snippet=payload.get("content", "")[:500],
            )
            citations.append(citation)
        
        return citations
    
    def search_by_component(
        self,
        component_id: str,
        limit: int = 10,
    ) -> list[Citation]:
        """
        Search specifically for a component ID across all documents.
        
        This is useful for cross-document triangulation.
        
        Args:
            component_id: The component ID to search for.
            limit: Maximum results to return.
            
        Returns:
            List of citations where this component appears.
        """
        # Use scroll to find all matches with this component ID
        results, _ = self.qdrant.scroll(
            collection_name=self.settings.qdrant_collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="component_ids",
                        match=MatchValue(value=component_id.upper()),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        )
        
        citations = []
        for result in results:
            payload = result.payload
            
            citation = Citation(
                document_id=payload.get("document_id", ""),
                filename=payload.get("filename", ""),
                page_number=payload.get("page_number", 0),
                document_type=DocumentType(payload.get("document_type", "unknown")),
                relevance_score=1.0,  # Exact match
                snippet=payload.get("content", "")[:500],
            )
            citations.append(citation)
        
        return citations
    
    def cross_document_search(
        self,
        component_id: str,
    ) -> dict[str, list[Citation]]:
        """
        Find a component across different document types.
        
        Args:
            component_id: The component ID to search for.
            
        Returns:
            Dict mapping document type to list of citations.
        """
        all_citations = self.search_by_component(component_id, limit=50)
        
        # Group by document type
        grouped = {}
        for citation in all_citations:
            doc_type = citation.document_type.value
            if doc_type not in grouped:
                grouped[doc_type] = []
            grouped[doc_type].append(citation)
        
        return grouped
