"""Configuration management for Project Mandolin."""

from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    anthropic_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""  # Optional fallback
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "mandolin_documents"
    
    # Model Configuration
    vlm_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "models/embedding-001"  # Gemini embedding
    
    # Path Configuration
    documents_dir: Path = Path("data/documents")
    processed_dir: Path = Path("data/processed")
    
    # Processing Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 100
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
