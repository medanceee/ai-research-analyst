"""
Complete RAG Pipeline Integration

This module integrates the document processors with the RAG system to provide
a seamless end-to-end pipeline from document ingestion to intelligent retrieval.

Key Components:
- RAGPipeline: Main orchestrator for the complete RAG workflow
- DocumentIngestionManager: Manages document processing and storage
- RAGQueryEngine: Handles complex queries with context building
- AnalyticsCollector: Tracks usage and performance metrics
"""

import uuid
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import time
import json

# Import document processors
from ...core.file_processors import DocumentProcessorFactory
from ...core.document_ingestion import Document

# Import RAG components
from .retrieval import RAGRetriever, RetrievalContext, SearchResult
from .vector_store import VectorStore
from .embeddings import EmbeddingService
from .chunking import AdaptiveChunker, BaseChunker

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of document ingestion process."""
    doc_id: str
    source_path: str
    chunks_created: int
    processing_time: float
    file_size: int
    processor_used: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class QueryResult:
    """Enhanced query result with context and analysis."""
    query: str
    retrieval_context: RetrievalContext
    answer: Optional[str] = None
    confidence: Optional[float] = None
    sources_used: Optional[List[str]] = None
    processing_time: Optional[float] = None


class DocumentIngestionManager:
    """
    Manages the complete document ingestion workflow.
    """
    
    def __init__(self, rag_retriever: RAGRetriever):
        """
        Initialize document ingestion manager.
        
        Args:
            rag_retriever: RAG retriever instance for storage
        """
        self.rag_retriever = rag_retriever
        self.processor_factory = DocumentProcessorFactory()
        
        # Track ingestion history
        self.ingestion_history: List[IngestionResult] = []
    
    def ingest_document(
        self, 
        file_path: str, 
        doc_id: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionResult:
        """
        Ingest a document through the complete pipeline.
        
        Args:
            file_path: Path to document or URL
            doc_id: Optional document ID
            additional_metadata: Additional metadata to attach
            
        Returns:
            IngestionResult with processing details
        """
        start_time = time.time()
        doc_id = doc_id or str(uuid.uuid4())
        additional_metadata = additional_metadata or {}
        
        try:
            # Check if format is supported
            if not self.processor_factory.is_supported(file_path):
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Create appropriate processor
            processor = self.processor_factory.create_processor(file_path)
            
            # Extract text and metadata
            text_content = processor.extract_text(file_path)
            doc_metadata = processor.extract_metadata(file_path)
            
            # Merge metadata
            doc_metadata.update(additional_metadata)
            doc_metadata.update({
                'doc_id': doc_id,
                'source_file': file_path,
                'ingestion_timestamp': time.time()
            })
            
            # Get file size
            file_size = 0
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
            elif file_path.startswith(('http://', 'https://')):
                file_size = len(text_content.encode('utf-8'))
            
            # Add to RAG system
            final_doc_id = self.rag_retriever.add_document(
                text=text_content,
                doc_id=doc_id,
                metadata=doc_metadata
            )
            
            # Get chunk count (approximate)
            chunks = self.rag_retriever.get_document_chunks(final_doc_id)
            chunks_created = len(chunks)
            
            processing_time = time.time() - start_time
            
            result = IngestionResult(
                doc_id=final_doc_id,
                source_path=file_path,
                chunks_created=chunks_created,
                processing_time=processing_time,
                file_size=file_size,
                processor_used=type(processor).__name__,
                success=True
            )
            
            self.ingestion_history.append(result)
            
            logger.info(f"Successfully ingested {file_path} as {final_doc_id} "
                       f"({chunks_created} chunks, {processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            result = IngestionResult(
                doc_id=doc_id,
                source_path=file_path,
                chunks_created=0,
                processing_time=processing_time,
                file_size=0,
                processor_used="unknown",
                success=False,
                error_message=error_msg
            )
            
            self.ingestion_history.append(result)
            logger.error(f"Failed to ingest {file_path}: {error_msg}")
            
            return result
    
    def ingest_multiple_documents(
        self, 
        file_paths: List[str],
        metadata_per_file: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[IngestionResult]:
        """
        Ingest multiple documents in batch.
        
        Args:
            file_paths: List of file paths to ingest
            metadata_per_file: Optional metadata for each file
            
        Returns:
            List of ingestion results
        """
        results = []
        metadata_per_file = metadata_per_file or {}
        
        for file_path in file_paths:
            file_metadata = metadata_per_file.get(file_path, {})
            result = self.ingest_document(file_path, additional_metadata=file_metadata)
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch ingestion completed: {successful}/{len(file_paths)} successful")
        
        return results
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get statistics about document ingestion."""
        if not self.ingestion_history:
            return {"total_documents": 0}
        
        successful = [r for r in self.ingestion_history if r.success]
        failed = [r for r in self.ingestion_history if not r.success]
        
        stats = {
            "total_documents": len(self.ingestion_history),
            "successful": len(successful),
            "failed": len(failed),
            "total_chunks": sum(r.chunks_created for r in successful),
            "avg_processing_time": sum(r.processing_time for r in successful) / len(successful) if successful else 0,
            "total_file_size": sum(r.file_size for r in successful),
            "processors_used": list(set(r.processor_used for r in successful))
        }
        
        return stats


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates the complete workflow.
    """
    
    def __init__(
        self,
        collection_name: str = "research_documents",
        persist_directory: str = "./rag_data",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunker: Optional[BaseChunker] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the complete RAG pipeline.
        
        Args:
            collection_name: Name for the document collection
            persist_directory: Directory for persistent storage
            embedding_model: Embedding model to use
            chunker: Document chunking strategy
            config: Pipeline configuration
        """
        self.config = config or {}
        
        # Initialize RAG retriever
        self.retriever = RAGRetriever(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            chunker=chunker or AdaptiveChunker(),
            config=self.config
        )
        
        # Initialize ingestion manager
        self.ingestion_manager = DocumentIngestionManager(self.retriever)
        
        # Track query history
        self.query_history: List[QueryResult] = []
        
        logger.info(f"RAG Pipeline initialized with collection: {collection_name}")
    
    def add_document(
        self, 
        file_path: str, 
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionResult:
        """
        Add a document to the RAG system.
        
        Args:
            file_path: Path to document file or URL
            doc_id: Optional document ID
            metadata: Optional metadata for the document
            
        Returns:
            IngestionResult with processing details
        """
        return self.ingestion_manager.ingest_document(
            file_path=file_path,
            doc_id=doc_id,
            additional_metadata=metadata
        )
    
    def add_multiple_documents(
        self, 
        file_paths: List[str],
        metadata_per_file: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[IngestionResult]:
        """Add multiple documents to the RAG system."""
        return self.ingestion_manager.ingest_multiple_documents(
            file_paths=file_paths,
            metadata_per_file=metadata_per_file
        )
    
    def query(
        self,
        question: str,
        n_results: int = 5,
        similarity_threshold: float = 0.5,
        include_context: bool = True
    ) -> QueryResult:
        """
        Query the RAG system for information.
        
        Args:
            question: Question or query to search for
            n_results: Number of relevant chunks to retrieve
            similarity_threshold: Minimum similarity for results
            include_context: Whether to include full context in result
            
        Returns:
            QueryResult with retrieved information
        """
        start_time = time.time()
        
        # Retrieve relevant context
        retrieval_context = self.retriever.search(
            query=question,
            n_results=n_results,
            similarity_threshold=similarity_threshold
        )
        
        processing_time = time.time() - start_time
        
        # Extract source information
        sources_used = list(set(
            result.source_document for result in retrieval_context.results
            if result.source_document
        ))
        
        # Create query result
        query_result = QueryResult(
            query=question,
            retrieval_context=retrieval_context,
            sources_used=sources_used,
            processing_time=processing_time,
            confidence=retrieval_context.avg_similarity
        )
        
        self.query_history.append(query_result)
        
        logger.info(f"Query processed: {len(retrieval_context.results)} results, "
                   f"{len(sources_used)} sources, {processing_time:.3f}s")
        
        return query_result
    
    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a specific document.
        
        Args:
            doc_id: Document ID to summarize
            
        Returns:
            Document summary with chunks and metadata
        """
        chunks = self.retriever.get_document_chunks(doc_id)
        
        if not chunks:
            return None
        
        # Aggregate metadata from chunks
        first_chunk = chunks[0]
        doc_metadata = {k: v for k, v in first_chunk.metadata.items() 
                       if not k.startswith('chunk')}
        
        total_text_length = sum(len(chunk.text) for chunk in chunks)
        
        return {
            'doc_id': doc_id,
            'metadata': doc_metadata,
            'chunk_count': len(chunks),
            'total_text_length': total_text_length,
            'avg_chunk_length': total_text_length / len(chunks),
            'chunks': [
                {
                    'chunk_id': chunk.chunk_id,
                    'text_preview': chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                    'chunk_index': chunk.chunk_index
                }
                for chunk in chunks
            ]
        }
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks."""
        return self.retriever.delete_document(doc_id)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the system.
        
        Returns:
            List of document summaries
        """
        # This is a simplified implementation
        # In a production system, you'd want to maintain a document registry
        
        # Get all unique doc_ids from the vector store
        # This is a workaround since ChromaDB doesn't directly support this
        try:
            # Get a large number of results to find all documents
            sample_results = self.retriever.vector_store.search("", n_results=10000)
            
            doc_ids = set()
            for result in sample_results:
                doc_id = result.get('metadata', {}).get('doc_id')
                if doc_id:
                    doc_ids.add(doc_id)
            
            # Get summary for each document
            documents = []
            for doc_id in doc_ids:
                summary = self.get_document_summary(doc_id)
                if summary:
                    documents.append(summary)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        rag_stats = self.retriever.get_collection_stats()
        ingestion_stats = self.ingestion_manager.get_ingestion_stats()
        
        query_stats = {
            "total_queries": len(self.query_history),
            "avg_results_per_query": sum(len(q.retrieval_context.results) for q in self.query_history) / len(self.query_history) if self.query_history else 0,
            "avg_query_time": sum(q.processing_time for q in self.query_history if q.processing_time) / len(self.query_history) if self.query_history else 0,
            "avg_confidence": sum(q.confidence for q in self.query_history if q.confidence) / len(self.query_history) if self.query_history else 0
        }
        
        return {
            "rag_system": rag_stats,
            "ingestion": ingestion_stats,
            "queries": query_stats,
            "pipeline_config": self.config
        }
    
    def export_knowledge_base(self, output_file: str) -> bool:
        """
        Export the current knowledge base to a JSON file.
        
        Args:
            output_file: Path to output JSON file
            
        Returns:
            True if export was successful
        """
        try:
            documents = self.list_documents()
            pipeline_stats = self.get_pipeline_stats()
            
            export_data = {
                "metadata": {
                    "export_timestamp": time.time(),
                    "total_documents": len(documents),
                    "pipeline_stats": pipeline_stats
                },
                "documents": documents
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported knowledge base to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export knowledge base: {e}")
            return False


class RAGQueryEngine:
    """
    Advanced query engine for the RAG system with context building and analysis.
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize query engine.
        
        Args:
            rag_pipeline: RAG pipeline instance
        """
        self.rag_pipeline = rag_pipeline
        self.query_cache: Dict[str, QueryResult] = {}
    
    def ask(
        self,
        question: str,
        context_length: int = 5,
        similarity_threshold: float = 0.5,
        use_cache: bool = True
    ) -> QueryResult:
        """
        Ask a question and get a comprehensive answer with context.
        
        Args:
            question: Question to ask
            context_length: Number of relevant chunks to include
            similarity_threshold: Minimum similarity for inclusion
            use_cache: Whether to use cached results
            
        Returns:
            QueryResult with answer and context
        """
        # Check cache
        cache_key = f"{question}_{context_length}_{similarity_threshold}"
        if use_cache and cache_key in self.query_cache:
            logger.info("Using cached query result")
            return self.query_cache[cache_key]
        
        # Query the RAG system
        query_result = self.rag_pipeline.query(
            question=question,
            n_results=context_length,
            similarity_threshold=similarity_threshold
        )
        
        # Cache result
        if use_cache:
            self.query_cache[cache_key] = query_result
        
        return query_result
    
    def multi_query_analysis(
        self,
        questions: List[str],
        n_results: int = 3
    ) -> Dict[str, QueryResult]:
        """
        Analyze multiple related questions and provide comprehensive context.
        
        Args:
            questions: List of questions to analyze
            n_results: Number of results per question
            
        Returns:
            Dictionary mapping questions to results
        """
        results = {}
        
        for question in questions:
            result = self.ask(question, context_length=n_results)
            results[question] = result
        
        logger.info(f"Completed multi-query analysis for {len(questions)} questions")
        return results
    
    def find_contradictions(self, topic: str) -> List[Dict[str, Any]]:
        """
        Find potential contradictions in the knowledge base about a topic.
        
        Args:
            topic: Topic to analyze for contradictions
            
        Returns:
            List of potential contradictions with evidence
        """
        # Search for information about the topic
        context = self.rag_pipeline.retriever.search(topic, n_results=10)
        
        contradictions = []
        
        # Simple contradiction detection (can be enhanced with NLP)
        contradiction_keywords = [
            ('increases', 'decreases'), ('improves', 'worsens'),
            ('positive', 'negative'), ('effective', 'ineffective'),
            ('supports', 'opposes'), ('confirms', 'contradicts')
        ]
        
        for i, result1 in enumerate(context.results):
            for j, result2 in enumerate(context.results[i+1:], i+1):
                # Check for contradictory keywords
                text1_lower = result1.text.lower()
                text2_lower = result2.text.lower()
                
                for pos_word, neg_word in contradiction_keywords:
                    if (pos_word in text1_lower and neg_word in text2_lower) or \
                       (neg_word in text1_lower and pos_word in text2_lower):
                        contradictions.append({
                            'topic': topic,
                            'evidence_1': {
                                'text': result1.text,
                                'source': result1.source_document,
                                'similarity': result1.similarity
                            },
                            'evidence_2': {
                                'text': result2.text,
                                'source': result2.source_document,
                                'similarity': result2.similarity
                            },
                            'contradiction_type': f"{pos_word} vs {neg_word}"
                        })
        
        return contradictions


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize RAG pipeline
    pipeline = RAGPipeline(
        collection_name="demo_pipeline",
        persist_directory="./demo_rag_data"
    )
    
    # Example: Add some sample content (simulating document ingestion)
    sample_docs = [
        ("Vector databases provide efficient storage and retrieval of high-dimensional embeddings for AI applications.", 
         {"source": "tech_overview", "type": "technical"}),
        ("ChromaDB offers persistent storage with built-in similarity search, making it ideal for RAG systems.",
         {"source": "chromadb_guide", "type": "documentation"}),
        ("RAG architectures combine retrieval and generation to provide more accurate and contextually relevant AI responses.",
         {"source": "rag_research", "type": "academic"})
    ]
    
    for text, metadata in sample_docs:
        doc_id = pipeline.retriever.add_document(text, metadata=metadata)
        print(f"Added document: {doc_id}")
    
    # Initialize query engine
    query_engine = RAGQueryEngine(pipeline)
    
    # Test queries
    questions = [
        "What is ChromaDB used for?",
        "How do vector databases work?", 
        "What are the benefits of RAG systems?"
    ]
    
    for question in questions:
        print(f"\n=== Question: {question} ===")
        result = query_engine.ask(question, context_length=2)
        
        print(f"Found {len(result.retrieval_context.results)} relevant chunks")
        print(f"Average similarity: {result.confidence:.3f}")
        print(f"Sources: {result.sources_used}")
        
        for i, search_result in enumerate(result.retrieval_context.results):
            print(f"\nResult {i+1} (similarity: {search_result.similarity:.3f}):")
            print(f"Text: {search_result.text}")
            print(f"Source: {search_result.source_document}")
    
    # Get pipeline statistics
    stats = pipeline.get_pipeline_stats()
    print(f"\n=== Pipeline Stats ===")
    print(f"Vector store: {stats['rag_system']['vector_store']}")
    print(f"Queries processed: {stats['queries']['total_queries']}")
    print(f"Average query time: {stats['queries']['avg_query_time']:.3f}s")