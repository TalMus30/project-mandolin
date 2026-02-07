"""Document embedder using ColPali and Gemini embeddings."""

import re
from typing import Optional
from PIL import Image
import google.generativeai as genai

from ..config import get_settings
from ..models import DocumentChunk


# Component ID patterns for extraction
COMPONENT_PATTERNS = [
    r'\b[A-Z]{1,3}\d{1,4}\b',           # K102, Y20, X1
    r'\b[A-Z]\d+:\d+\b',                 # X2:15, A1:3
    r'\b[A-Z]{2,3}-\d{2,4}\b',           # PV-001, FIC-102
    r'\bTag[:\s]*[A-Z0-9-]+\b',          # Tag: ABC-123
]


class DocumentEmbedder:
    """Handles embedding generation for document chunks."""
    
    def __init__(self):
        """Initialize embedder with Gemini."""
        settings = get_settings()
        genai.configure(api_key=settings.google_api_key)
        self.embedding_model = settings.embedding_model
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in COMPONENT_PATTERNS]
    
    def extract_component_ids(self, text: str) -> list[str]:
        """
        Extract component IDs from text using regex patterns.
        
        Args:
            text: Text content to search.
            
        Returns:
            List of unique component IDs found.
        """
        component_ids = set()
        
        for pattern in self._compiled_patterns:
            matches = pattern.findall(text)
            component_ids.update(m.upper() for m in matches)
        
        return list(component_ids)
    
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for text content.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector.
        """
        if not text.strip():
            return []
        
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document",
        )
        
        return result["embedding"]
    
    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Query text.
            
        Returns:
            Embedding vector optimized for retrieval.
        """
        result = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="retrieval_query",
        )
        
        return result["embedding"]
    
    def create_chunk(
        self,
        document_id: str,
        filename: str,
        page_number: int,
        document_type: str,
        text_content: str,
        image_path: Optional[str] = None,
    ) -> DocumentChunk:
        """
        Create a fully processed document chunk with embeddings.
        
        Args:
            document_id: Parent document ID.
            filename: Source filename.
            page_number: Page number (1-indexed).
            document_type: Type of document.
            text_content: Extracted text content.
            image_path: Path to page image.
            
        Returns:
            DocumentChunk with embeddings and extracted component IDs.
        """
        # Extract component IDs
        component_ids = self.extract_component_ids(text_content)
        
        # Generate text embedding if content exists
        embedding = None
        if text_content.strip():
            embedding = self.embed_text(text_content)
        
        chunk_id = f"{document_id}_page_{page_number}"
        
        return DocumentChunk(
            id=chunk_id,
            document_id=document_id,
            filename=filename,
            page_number=page_number,
            document_type=document_type,
            content=text_content,
            component_ids=component_ids,
            image_path=image_path,
            embedding=embedding,
        )
