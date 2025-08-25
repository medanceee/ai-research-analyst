"""
Configuration settings for AI Research Analyst

This module provides comprehensive configuration management using Pydantic
for type validation and environment variable support.
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
from pathlib import Path
import os


class IngestionConfig(BaseSettings):
    """
    Configuration for document ingestion system.
    """
    
    # File processing settings
    supported_formats: List[str] = Field(
        default=['.pdf', '.docx', '.txt', '.md', '.html'],
        description="List of supported file extensions"
    )
    max_file_size: int = Field(
        default=100_000_000,  # 100MB
        description="Maximum file size in bytes"
    )
    encoding: str = Field(
        default='utf-8',
        description="Default text encoding"
    )
    
    # Content validation
    min_content_length: int = Field(
        default=10,
        description="Minimum content length in characters"
    )
    max_content_length: int = Field(
        default=10_000_000,
        description="Maximum content length in characters"
    )
    
    # Chunking settings
    chunking_strategy: str = Field(
        default='fixed_size',
        description="Default chunking strategy (fixed_size, semantic, paragraph, sentence)"
    )
    chunk_size: int = Field(
        default=1000,
        description="Default chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks in characters"
    )
    min_chunk_length: int = Field(
        default=50,
        description="Minimum chunk length in characters"
    )
    
    # Metadata settings
    extract_metadata: bool = Field(
        default=True,
        description="Whether to extract document metadata"
    )
    enrich_metadata: bool = Field(
        default=True,
        description="Whether to enrich metadata with computed fields"
    )
    
    # Processing settings
    batch_size: int = Field(
        default=10,
        description="Number of files to process in batch"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing"
    )
    max_workers: int = Field(
        default=4,
        description="Maximum number of worker processes"
    )
    
    # Output settings
    output_format: str = Field(
        default='json',
        description="Output format (json, jsonl)"
    )
    save_intermediate: bool = Field(
        default=False,
        description="Save intermediate processing results"
    )

    class Config:
        env_prefix = "INGESTION_"
        case_sensitive = False


class VectorStoreConfig(BaseSettings):
    """
    Configuration for vector database (ChromaDB).
    """
    
    # Database settings
    persist_directory: str = Field(
        default="./chroma_db",
        description="Directory for persistent ChromaDB storage"
    )
    collection_name: str = Field(
        default="research_documents", 
        description="Default collection name"
    )
    
    # Embedding settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    embedding_dimensions: int = Field(
        default=384,
        description="Embedding vector dimensions"
    )
    
    # Search settings
    default_k: int = Field(
        default=5,
        description="Default number of results to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.1,
        description="Minimum similarity threshold for results"
    )
    
    class Config:
        env_prefix = "VECTOR_"
        case_sensitive = False


class LLMConfig(BaseSettings):
    """
    Configuration for Large Language Model integration.
    """
    
    # Model settings
    default_provider: str = Field(
        default="ollama",
        description="Default LLM provider (ollama, openai, anthropic)"
    )
    
    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    ollama_model: str = Field(
        default="llama3.2:3b",
        description="Ollama model name"
    )
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    openai_model: str = Field(
        default="gpt-3.5-turbo",
        description="OpenAI model name"
    )
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    anthropic_model: str = Field(
        default="claude-3-sonnet-20240229",
        description="Anthropic model name"
    )
    
    # Generation settings
    temperature: float = Field(
        default=0.1,
        description="Temperature for text generation"
    )
    max_tokens: int = Field(
        default=2000,
        description="Maximum tokens for generation"
    )
    timeout: int = Field(
        default=60,
        description="Request timeout in seconds"
    )
    
    class Config:
        env_prefix = "LLM_"
        case_sensitive = False


class APIConfig(BaseSettings):
    """
    Configuration for FastAPI web service.
    """
    
    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    port: int = Field(
        default=8000,
        description="Server port"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode"
    )
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-this",
        description="Secret key for JWT tokens"
    )
    
    # CORS
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    class Config:
        env_prefix = "API_"
        case_sensitive = False


class LoggingConfig(BaseSettings):
    """
    Configuration for logging system.
    """
    
    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Log file path (None for stdout only)"
    )
    max_file_size: int = Field(
        default=10_000_000,  # 10MB
        description="Maximum log file size in bytes"
    )
    backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )
    
    class Config:
        env_prefix = "LOG_"
        case_sensitive = False


class Settings(BaseSettings):
    """
    Main settings class that combines all configuration sections.
    """
    
    # Project info
    project_name: str = "AI Research Analyst"
    version: str = "0.1.0"
    description: str = "Enterprise AI research assistant with RAG capabilities"
    
    # Component configurations
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Data directories
    data_dir: Path = Field(
        default=Path("./data"),
        description="Base data directory"
    )
    raw_data_dir: Path = Field(
        default=Path("./data/raw"),
        description="Raw data directory"
    )
    processed_data_dir: Path = Field(
        default=Path("./data/processed"),
        description="Processed data directory"
    )
    
    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Allow extra fields from environment variables

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create data directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance (lazy initialization)
_settings: Settings = None


def get_settings() -> Settings:
    """
    Get the global settings instance (lazy initialization).
    
    Returns:
        Settings: Global settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def load_settings_from_file(file_path: str) -> Settings:
    """
    Load settings from a specific file.
    
    Args:
        file_path: Path to the settings file
        
    Returns:
        Settings: Settings instance loaded from file
    """
    return Settings(_env_file=file_path)


if __name__ == "__main__":
    # Test configuration loading
    print("AI Research Analyst - Configuration Test")
    print("=" * 50)
    
    config = get_settings()
    
    print(f"Project: {config.project_name} v{config.version}")
    print(f"Data directory: {config.data_dir}")
    print(f"Supported formats: {config.ingestion.supported_formats}")
    print(f"LLM Provider: {config.llm.default_provider}")
    print(f"Vector Store: {config.vector_store.collection_name}")
    
    print("\n✅ Configuration loaded successfully!")