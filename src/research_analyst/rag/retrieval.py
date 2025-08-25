"""
Semantic Search and Retrieval System for RAG Pipeline

This module provides a comprehensive retrieval system that combines document chunking,
embedding generation, vector storage, and advanced search capabilities.

Key Components:
- RAGRetriever: Main interface for document storage and retrieval
- SearchResult: Structured search result with relevance scoring
- QueryProcessor: Enhanced query processing and expansion
- RetrievalPipeline: End-to-end document processing and search
"""

import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import time

from .vector_store import VectorStore, DocumentEmbedding
from .embeddings import EmbeddingService
from .chunking import BaseChunker, AdaptiveChunker, ChunkingPipeline

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with relevance information."""
    chunk_id: str
    text: str
    similarity: float
    distance: float
    metadata: Dict[str, Any]
    source_document: Optional[str] = None
    chunk_index: Optional[int] = None
    
    def __post_init__(self):
        # Extract source info from metadata
        if self.metadata:
            self.source_document = self.metadata.get('source_file', self.metadata.get('doc_id'))
            self.chunk_index = self.metadata.get('chunk_index')


@dataclass 
class RetrievalContext:
    """Context information from retrieval with multiple results."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    avg_similarity: float
    
    def get_context_text(self, max_length: Optional[int] = None) -> str:
        """Get combined context text from all results."""
        context_parts = []
        for i, result in enumerate(self.results):
            source_info = f"[Source: {result.source_document}]" if result.source_document else ""
            context_parts.append(f"{source_info}\n{result.text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        if max_length and len(context) > max_length:
            context = context[:max_length] + "... [truncated]"
        
        return context


class QueryProcessor:
    """
    Processes and enhances queries for better retrieval performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.expand_queries = self.config.get('expand_queries', True)
        self.min_query_length = self.config.get('min_query_length', 3)
    
    def process_query(self, query: str) -> List[str]:
        """
        Process and potentially expand a query.
        
        Args:
            query: Original search query
            
        Returns:
            List of processed queries (original + expansions)
        """
        # Clean the query
        cleaned_query = self._clean_query(query)
        
        if len(cleaned_query) < self.min_query_length:
            raise ValueError(f"Query too short: {len(cleaned_query)} chars (min: {self.min_query_length})")
        
        queries = [cleaned_query]
        
        # Add query expansions if enabled
        if self.expand_queries:
            expanded = self._expand_query(cleaned_query)
            queries.extend(expanded)
        
        return queries
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query text."""
        # Remove excessive whitespace
        query = ' '.join(query.split())
        
        # Remove special characters that might interfere with search
        query = query.strip()
        
        return query
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Generate query expansions for better retrieval.
        Simple expansion strategies (can be enhanced with NLP models).
        """
        expansions = []
        
        # Add variations with common synonyms
        synonym_map = {
            'AI': ['artificial intelligence', 'machine learning'],
            'ML': ['machine learning', 'artificial intelligence'],
            'vector': ['embedding', 'vector space'],
            'database': ['storage', 'repository', 'store'],
            'search': ['retrieval', 'query', 'find'],
            'document': ['text', 'file', 'content']
        }
        
        query_lower = query.lower()
        for term, synonyms in synonym_map.items():
            if term.lower() in query_lower:
                for synonym in synonyms:
                    expansion = query.replace(term, synonym, 1)
                    if expansion != query:
                        expansions.append(expansion)
        
        return expansions[:2]  # Limit to 2 expansions


class RAGRetriever:
    """
    Main RAG retrieval system that combines all components.
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./rag_data",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunker: Optional[BaseChunker] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RAG retriever.
        
        Args:
            collection_name: Name for the document collection
            persist_directory: Directory for persistent storage
            embedding_model: Embedding model to use
            chunker: Document chunking strategy
            config: Additional configuration
        """
        self.config = config or {}
        
        # Initialize components
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model
        )
        
        self.embedding_service = EmbeddingService(
            model_name=embedding_model,
            enable_cache=self.config.get('enable_cache', True)
        )
        
        self.chunker = chunker or AdaptiveChunker()
        self.chunking_pipeline = ChunkingPipeline(self.chunker)
        
        self.query_processor = QueryProcessor(self.config.get('query_processing', {}))
        
        logger.info(f"RAGRetriever initialized with collection: {collection_name}")
    
    def add_document(
        self, 
        text: str, 
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a document to the retrieval system.
        
        Args:
            text: Document text content
            doc_id: Optional document ID (generated if not provided)
            metadata: Document metadata
            
        Returns:
            Document ID that was used
        """
        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}
        metadata['doc_id'] = doc_id
        
        # Chunk the document
        chunks = self.chunking_pipeline.process_document(text, metadata)
        
        # Convert chunks to document embeddings
        document_embeddings = []
        for chunk in chunks:
            doc_embedding = DocumentEmbedding(
                id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata
            )
            document_embeddings.append(doc_embedding)
        
        # Add to vector store
        self.vector_store.add_documents(document_embeddings)
        
        logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
        return doc_id
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        similarity_threshold: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> RetrievalContext:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            n_results: Maximum number of results
            similarity_threshold: Minimum similarity score
            metadata_filter: Filter by metadata fields
            
        Returns:
            RetrievalContext with search results and metadata
        """
        start_time = time.time()
        
        # Process query
        queries = self.query_processor.process_query(query)
        
        all_results = []
        
        # Search with each query variant
        for q in queries:
            search_results = self.vector_store.search(
                query=q,
                n_results=n_results * 2,  # Get more to filter later
                where=metadata_filter
            )
            
            # Convert to SearchResult objects
            for result in search_results:
                if result['similarity'] >= similarity_threshold:
                    search_result = SearchResult(
                        chunk_id=result['id'],
                        text=result['document'],
                        similarity=result['similarity'],
                        distance=result['distance'],
                        metadata=result['metadata']
                    )
                    all_results.append(search_result)
        
        # Remove duplicates and sort by similarity
        unique_results = self._deduplicate_results(all_results)
        unique_results.sort(key=lambda x: x.similarity, reverse=True)
        
        # Limit to requested number
        final_results = unique_results[:n_results]
        
        search_time = time.time() - start_time
        avg_similarity = sum(r.similarity for r in final_results) / len(final_results) if final_results else 0.0
        
        context = RetrievalContext(
            query=query,
            results=final_results,
            total_results=len(final_results),
            search_time=search_time,
            avg_similarity=avg_similarity
        )
        
        logger.info(f"Search completed: {len(final_results)} results in {search_time:.3f}s")
        return context
    
    def get_document_chunks(self, doc_id: str) -> List[SearchResult]:
        """
        Get all chunks for a specific document.
        
        Args:
            doc_id: Document ID to retrieve chunks for
            
        Returns:
            List of chunks for the document
        """
        # Search with metadata filter for doc_id
        search_results = self.vector_store.search(
            query="",  # Empty query to get all
            n_results=1000,  # Large number to get all chunks
            where={"doc_id": doc_id}
        )
        
        chunks = []
        for result in search_results:
            search_result = SearchResult(
                chunk_id=result['id'],
                text=result['document'],
                similarity=result['similarity'],
                distance=result['distance'],
                metadata=result['metadata']
            )
            chunks.append(search_result)
        
        # Sort by chunk index
        chunks.sort(key=lambda x: x.metadata.get('chunk_index', 0))
        return chunks
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete all chunks of a document.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            chunks = self.get_document_chunks(doc_id)
            for chunk in chunks:
                self.vector_store.delete_document(chunk.chunk_id)
            
            logger.info(f"Deleted document {doc_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the retrieval system."""
        vector_stats = self.vector_store.get_collection_stats()
        embedding_stats = self.embedding_service.get_model_info()
        
        return {
            'vector_store': vector_stats,
            'embedding_service': embedding_stats,
            'chunking_strategy': type(self.chunker).__name__
        }
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on chunk_id."""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result.chunk_id not in seen_ids:
                seen_ids.add(result.chunk_id)
                unique_results.append(result)
        
        return unique_results


class RetrievalPipeline:
    """
    End-to-end pipeline for document processing and retrieval.
    Integrates with document processors for complete RAG workflow.
    """
    
    def __init__(
        self, 
        retriever: RAGRetriever,
        document_processors: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize retrieval pipeline.
        
        Args:
            retriever: RAG retriever instance
            document_processors: Optional document processor instances
        """
        self.retriever = retriever
        self.document_processors = document_processors or {}
    
    def ingest_file(
        self, 
        file_path: str, 
        doc_id: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ingest a file into the RAG system.
        
        Args:
            file_path: Path to the file to ingest
            doc_id: Optional document ID
            additional_metadata: Additional metadata for the document
            
        Returns:
            Document ID that was assigned
        """
        # This would integrate with document processors from file_processors.py
        # For now, we'll handle basic text processing
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content (simplified - would use actual processors)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Prepare metadata
        metadata = {
            'source_file': str(file_path),
            'file_name': file_path_obj.name,
            'file_size': file_path_obj.stat().st_size,
            'file_type': file_path_obj.suffix
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Add to retriever
        doc_id = self.retriever.add_document(text, doc_id, metadata)
        
        logger.info(f"Ingested file {file_path} as document {doc_id}")
        return doc_id
    
    def search_documents(
        self,
        query: str,
        n_results: int = 5,
        similarity_threshold: float = 0.5,
        source_filter: Optional[List[str]] = None,
        file_type_filter: Optional[List[str]] = None
    ) -> RetrievalContext:
        """
        Search across all ingested documents with advanced filtering.
        
        Args:
            query: Search query
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score
            source_filter: Filter by source file names
            file_type_filter: Filter by file types
            
        Returns:
            RetrievalContext with search results
        """
        # Build metadata filter
        metadata_filter = {}
        
        if source_filter:
            # This would need to be adapted based on ChromaDB's query syntax
            # For now, we'll do post-filtering
            pass
        
        if file_type_filter:
            # This would need to be adapted based on ChromaDB's query syntax
            pass
        
        # Perform search
        context = self.retriever.search(
            query=query,
            n_results=n_results,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter
        )
        
        # Apply additional filters if needed
        if source_filter or file_type_filter:
            filtered_results = []
            for result in context.results:
                # Filter by source
                if source_filter:
                    source_file = result.metadata.get('source_file', '')
                    if not any(source in source_file for source in source_filter):
                        continue
                
                # Filter by file type
                if file_type_filter:
                    file_type = result.metadata.get('file_type', '')
                    if file_type not in file_type_filter:
                        continue
                
                filtered_results.append(result)
            
            # Update context with filtered results
            context.results = filtered_results
            context.total_results = len(filtered_results)
            if filtered_results:
                context.avg_similarity = sum(r.similarity for r in filtered_results) / len(filtered_results)
        
        return context
    
    def get_similar_documents(self, doc_id: str, n_results: int = 5) -> List[SearchResult]:
        """
        Find documents similar to a given document.
        
        Args:
            doc_id: Document ID to find similarities for
            n_results: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
        # Get chunks for the document
        chunks = self.retriever.get_document_chunks(doc_id)
        
        if not chunks:
            return []
        
        # Use the first chunk as the query (could be enhanced)
        query_chunk = chunks[0]
        
        # Search for similar documents (excluding the original)
        context = self.retriever.search(
            query=query_chunk.text,
            n_results=n_results + 10  # Get extra to filter out original
        )
        
        # Filter out chunks from the original document
        similar_results = []
        for result in context.results:
            if result.metadata.get('doc_id') != doc_id:
                similar_results.append(result)
                if len(similar_results) >= n_results:
                    break
        
        return similar_results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize retrieval system
    retriever = RAGRetriever(
        collection_name="test_rag",
        persist_directory="./test_rag_data"
    )
    
    # Add sample documents
    documents = [
        ("Vector databases are specialized systems for storing and querying high-dimensional embeddings.", 
         {"source": "tech_docs", "type": "technical"}),
        ("ChromaDB provides persistent storage for embeddings with built-in similarity search capabilities.",
         {"source": "chromadb_docs", "type": "documentation"}),
        ("RAG systems combine retrieval-augmented generation to provide contextually relevant responses.",
         {"source": "ai_research", "type": "academic"})
    ]
    
    for text, metadata in documents:
        doc_id = retriever.add_document(text, metadata=metadata)
        print(f"Added document: {doc_id}")
    
    # Test search
    print("\n=== Search Results ===")
    context = retriever.search("vector database storage", n_results=3)
    
    print(f"Query: {context.query}")
    print(f"Found {context.total_results} results in {context.search_time:.3f}s")
    print(f"Average similarity: {context.avg_similarity:.3f}")
    
    for i, result in enumerate(context.results):
        print(f"\nResult {i+1} (similarity: {result.similarity:.3f}):")
        print(f"Text: {result.text}")
        print(f"Source: {result.source_document}")
        print(f"Metadata: {result.metadata}")
    
    # Test context generation
    print("\n=== Context Text ===")
    context_text = context.get_context_text(max_length=500)
    print(context_text)
    
    # Get system stats
    stats = retriever.get_collection_stats()
    print(f"\n=== System Stats ===")
    print(f"Vector store: {stats['vector_store']}")
    print(f"Embedding service: {stats['embedding_service']}")
    print(f"Chunking strategy: {stats['chunking_strategy']}")