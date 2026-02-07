"""Document ingestion pipeline orchestration."""

from pathlib import Path
from typing import Callable, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    PayloadSchemaType,
)

from ..config import get_settings
from ..models import Document, DocumentType, DocumentChunk, IngestionStatus
from .pdf_processor import PDFProcessor
from .classifier import DocumentClassifier
from .embedder import DocumentEmbedder


class IngestionPipeline:
    """Orchestrates the complete document ingestion process."""
    
    EMBEDDING_DIM = 768  # Gemini embedding dimension
    
    def __init__(
        self,
        on_progress: Optional[Callable[[IngestionStatus], None]] = None,
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            on_progress: Optional callback for progress updates.
        """
        self.settings = get_settings()
        self.pdf_processor = PDFProcessor(dpi=150)
        self.classifier = DocumentClassifier()
        self.embedder = DocumentEmbedder()
        self.on_progress = on_progress
        
        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
        )
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        collections = self.qdrant.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.settings.qdrant_collection_name not in collection_names:
            self.qdrant.create_collection(
                collection_name=self.settings.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=self.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            
            # Create payload indices for efficient filtering
            self.qdrant.create_payload_index(
                collection_name=self.settings.qdrant_collection_name,
                field_name="document_type",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.qdrant.create_payload_index(
                collection_name=self.settings.qdrant_collection_name,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
    
    def _update_progress(self, status: IngestionStatus):
        """Update progress via callback if provided."""
        if self.on_progress:
            self.on_progress(status)
    
    def process_document(
        self,
        document_id: str,
        pdf_path: Path,
        filename: str,
    ) -> Document:
        """
        Process a single document through the full pipeline.
        
        Args:
            document_id: Unique document identifier.
            pdf_path: Path to the PDF file.
            filename: Original filename.
            
        Returns:
            Processed Document object.
        """
        status = IngestionStatus(
            document_id=document_id,
            filename=filename,
            status="processing",
        )
        self._update_progress(status)
        
        try:
            # Get page count
            page_count = self.pdf_processor.get_page_count(pdf_path)
            status.total_pages = page_count
            self._update_progress(status)
            
            # Extract pages
            pages = list(self.pdf_processor.extract_pages(pdf_path))
            
            # Classify document type from sample pages
            page_images = [p.image for p in pages]
            classification = self.classifier.classify_document(page_images)
            document_type = classification["document_type"]
            status.document_type = document_type
            self._update_progress(status)
            
            # Save page images
            self.pdf_processor.save_page_images(
                pdf_path,
                self.settings.processed_dir,
                document_id,
            )
            
            # Process each page into chunks and store
            points = []
            for page in pages:
                status.pages_processed = page.page_number
                self._update_progress(status)
                
                image_path = str(
                    self.settings.processed_dir / document_id / f"page_{page.page_number}.png"
                )
                
                chunk = self.embedder.create_chunk(
                    document_id=document_id,
                    filename=filename,
                    page_number=page.page_number,
                    document_type=document_type.value,
                    text_content=page.text_content,
                    image_path=image_path,
                )
                
                # Only store if we have an embedding
                if chunk.embedding:
                    point = PointStruct(
                        id=hash(chunk.id) & 0x7FFFFFFFFFFFFFFF,  # Positive int64
                        vector=chunk.embedding,
                        payload={
                            "chunk_id": chunk.id,
                            "document_id": chunk.document_id,
                            "filename": chunk.filename,
                            "page_number": chunk.page_number,
                            "document_type": chunk.document_type,
                            "content": chunk.content[:2000] if chunk.content else "",
                            "component_ids": chunk.component_ids,
                            "image_path": chunk.image_path,
                        },
                    )
                    points.append(point)
            
            # Batch upsert to Qdrant
            if points:
                self.qdrant.upsert(
                    collection_name=self.settings.qdrant_collection_name,
                    points=points,
                )
            
            status.status = "completed"
            self._update_progress(status)
            
            return Document(
                id=document_id,
                filename=filename,
                document_type=document_type,
                page_count=page_count,
                processed=True,
            )
            
        except Exception as e:
            status.status = "failed"
            status.error = str(e)
            self._update_progress(status)
            raise
