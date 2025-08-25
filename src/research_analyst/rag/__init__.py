"""
RAG (Retrieval-Augmented Generation) System

This package provides a complete RAG implementation with:
- Vector storage using ChromaDB
- Multiple chunking strategies
- Embedding generation and caching
- Semantic search and retrieval
- Integration with document processors

Main Components:
- RAGPipeline: Complete end-to-end RAG workflow
- RAGRetriever: Core retrieval system
- VectorStore: ChromaDB vector database interface
- EmbeddingService: Embedding generation with caching
- Various chunking strategies for optimal document processing
"""

# Main pipeline interface
from .rag_pipeline import RAGPipeline, RAGQueryEngine, DocumentIngestionManager

# Core RAG components
from .retrieval import RAGRetriever, RetrievalContext, SearchResult
from .vector_store import VectorStore, VectorStoreManager, DocumentEmbedding
from .embeddings import EmbeddingService, EmbeddingResult
from .chunking import (
    DocumentChunk, BaseChunker, FixedSizeChunker, 
    ParagraphChunker, SentenceChunker, AdaptiveChunker, ChunkingPipeline
)

# Version info
__version__ = "1.0.0"

# Default configurations
DEFAULT_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "collection_name": "research_documents", 
    "persist_directory": "./rag_data",
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "similarity_threshold": 0.5,
    "enable_cache": True
}

# Quick setup function
def create_rag_pipeline(
    collection_name: str = None,
    persist_directory: str = None,
    embedding_model: str = None,
    config: dict = None
) -> RAGPipeline:
    """
    Quick setup function for creating a RAG pipeline with defaults.
    
    Args:
        collection_name: Name for document collection
        persist_directory: Storage directory
        embedding_model: Embedding model name
        config: Additional configuration
        
    Returns:
        Configured RAGPipeline instance
    """
    final_config = DEFAULT_CONFIG.copy()
    if config:
        final_config.update(config)
    
    return RAGPipeline(
        collection_name=collection_name or final_config["collection_name"],
        persist_directory=persist_directory or final_config["persist_directory"],
        embedding_model=embedding_model or final_config["embedding_model"],
        config=final_config
    )

__all__ = [
    # Main interfaces
    'RAGPipeline', 'RAGQueryEngine', 'create_rag_pipeline',
    
    # Core components
    'RAGRetriever', 'VectorStore', 'EmbeddingService',
    
    # Data structures
    'DocumentEmbedding', 'DocumentChunk', 'SearchResult', 'RetrievalContext',
    
    # Chunking strategies
    'BaseChunker', 'FixedSizeChunker', 'ParagraphChunker', 
    'SentenceChunker', 'AdaptiveChunker', 'ChunkingPipeline',
    
    # Utilities
    'VectorStoreManager', 'DocumentIngestionManager',
    
    # Configuration
    'DEFAULT_CONFIG'
]