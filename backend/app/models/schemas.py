"""Pydantic schemas for Project Mandolin."""

from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Type of technical document."""
    MANUAL = "manual"
    ELECTRICAL_SCHEMATIC = "electrical_schematic"
    MEDIA_LAYOUT = "media_layout"  # P&ID, pneumatic, water lines
    UNKNOWN = "unknown"


class Document(BaseModel):
    """Represents an uploaded document."""
    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    document_type: DocumentType = Field(..., description="Classified document type")
    page_count: int = Field(..., description="Total number of pages")
    upload_timestamp: datetime = Field(default_factory=datetime.now)
    processed: bool = Field(default=False)
    
    class Config:
        use_enum_values = True


class DocumentChunk(BaseModel):
    """A chunk of document content with metadata."""
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    filename: str = Field(..., description="Source filename")
    page_number: int = Field(..., description="Page number (1-indexed)")
    document_type: DocumentType = Field(..., description="Document type")
    content: Optional[str] = Field(None, description="Text content if extracted")
    component_ids: list[str] = Field(default_factory=list, description="Extracted component IDs")
    image_path: Optional[str] = Field(None, description="Path to page image")
    embedding: Optional[list[float]] = Field(None, description="Vector embedding")
    
    class Config:
        use_enum_values = True


class Citation(BaseModel):
    """A citation reference to a source document."""
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Source filename")
    page_number: int = Field(..., description="Page number")
    document_type: DocumentType = Field(..., description="Document type")
    relevance_score: float = Field(..., description="Relevance score 0-1")
    snippet: Optional[str] = Field(None, description="Relevant text snippet")
    
    class Config:
        use_enum_values = True


class QueryRequest(BaseModel):
    """User query request."""
    query: str = Field(..., description="Natural language query", min_length=1)
    document_types: Optional[list[DocumentType]] = Field(
        None, 
        description="Filter by document types"
    )
    max_citations: int = Field(default=5, description="Maximum citations to return")


class QueryResponse(BaseModel):
    """Response to a user query."""
    answer: str = Field(..., description="Synthesized answer")
    citations: list[Citation] = Field(default_factory=list, description="Source citations")
    confidence: float = Field(..., description="Confidence score 0-1")
    cross_references: list[str] = Field(
        default_factory=list, 
        description="Component IDs referenced across documents"
    )
    grounded: bool = Field(..., description="Whether answer is fully grounded in sources")


class IngestionStatus(BaseModel):
    """Status of document ingestion."""
    document_id: str
    filename: str
    status: str = Field(..., description="processing, completed, failed")
    pages_processed: int = 0
    total_pages: int = 0
    document_type: Optional[DocumentType] = None
    error: Optional[str] = None
