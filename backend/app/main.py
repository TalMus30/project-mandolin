"""FastAPI main entry point for Project Mandolin."""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
import shutil

from .config import get_settings
from .models import (
    QueryRequest,
    QueryResponse,
    IngestionStatus,
    Document,
    DocumentType,
)

# Initialize FastAPI app
app = FastAPI(
    title="Project Mandolin",
    description="Multi-Source Engineering Intelligence System",
    version="0.1.0",
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for ingestion status (replace with DB in production)
ingestion_status: dict[str, IngestionStatus] = {}


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Project Mandolin"}


@app.get("/health")
async def health_check():
    """Detailed health check."""
    settings = get_settings()
    return {
        "status": "healthy",
        "qdrant_configured": bool(settings.qdrant_host),
        "anthropic_configured": bool(settings.anthropic_api_key),
        "google_configured": bool(settings.google_api_key),
    }


@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a PDF document for processing."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    settings = get_settings()
    
    # Create directories if they don't exist
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique document ID
    doc_id = str(uuid.uuid4())
    
    # Save the uploaded file
    file_path = settings.documents_dir / f"{doc_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Initialize ingestion status
    status = IngestionStatus(
        document_id=doc_id,
        filename=file.filename,
        status="queued",
        total_pages=0,
    )
    ingestion_status[doc_id] = status
    
    # Queue background processing
    # background_tasks.add_task(process_document, doc_id, file_path)
    
    return {
        "document_id": doc_id,
        "filename": file.filename,
        "status": "queued",
        "message": "Document uploaded successfully. Processing will begin shortly.",
    }


@app.get("/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """Get the processing status of a document."""
    if document_id not in ingestion_status:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return ingestion_status[document_id]


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    return {"documents": list(ingestion_status.values())}


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document corpus with natural language."""
    # Placeholder response - will be implemented with LangGraph agent
    return QueryResponse(
        answer="This endpoint will be implemented with the LangGraph agentic RAG pipeline.",
        citations=[],
        confidence=0.0,
        cross_references=[],
        grounded=False,
    )


@app.get("/documents/{document_id}/page/{page_number}")
async def get_document_page(document_id: str, page_number: int):
    """Get a specific page image from a processed document."""
    settings = get_settings()
    
    # Find the page image
    page_path = settings.processed_dir / document_id / f"page_{page_number}.png"
    
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")
    
    return FileResponse(page_path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
