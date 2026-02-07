# Project Mandolin

Multi-Source Engineering Intelligence System - A multimodal RAG system for technical document understanding.

## Overview

Project Mandolin is a proof-of-concept AI system that can ingest and understand technical documents (operating manuals, electrical schematics, P&ID diagrams) and provide grounded, citation-backed answers to natural language queries.

## Features (Planned)

- **Visual Document Understanding**: Uses ColPali for OCR-free document embedding
- **Cross-Document Triangulation**: Find related information across different document types
- **Hallucination Prevention**: Strict grounding with source citations
- **Dual-Pane UI**: Answers with clickable PDF page citations

## Tech Stack

- **Backend**: Python, FastAPI, LangGraph
- **Frontend**: React + Vite
- **VLM**: Claude 4.5 Sonnet
- **Embeddings**: ColPali (visual), Gemini embedding-001 (text)
- **Vector DB**: Qdrant (self-hosted)

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- Node.js 18+ (for frontend)

### Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and add your API keys
3. Start Qdrant: `docker-compose up -d qdrant`
4. Install Python dependencies: `pip install -r backend/requirements.txt`
5. Start the backend: `uvicorn app.main:app --reload`
6. Install frontend dependencies: `cd frontend && npm install`
7. Start the frontend: `npm run dev`

### Adding Documents

Place your PDF files in `data/documents/` to be processed.

## License

MIT
