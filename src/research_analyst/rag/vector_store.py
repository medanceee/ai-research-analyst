"""
ChromaDB Vector Store for RAG Pipeline

This module provides a comprehensive interface for managing document embeddings
in ChromaDB, including storage, retrieval, and collection management.

Key Components:
- VectorStore: Main interface for ChromaDB operations
- DocumentEmbedding: Data structure for stored embeddings
- Query operations with metadata filtering
"""

import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class DocumentEmbedding:
    """Represents a document embedding with metadata."""
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


class VectorStore:
    """
    ChromaDB-based vector store for document embeddings.
    
    Features:
    - Persistent storage with ChromaDB
    - Automatic embedding generation
    - Metadata filtering and search
    - Collection management
    - Batch operations for performance
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Sentence transformer model name
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embedding_model_name = embedding_model
        
        # Initialize ChromaDB client
        self._init_chromadb()
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"VectorStore initialized with collection: {collection_name}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB client with persistent storage."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    def _init_embedding_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.embedding_model_name}: {e}")
            raise
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def add_documents(self, documents: List[DocumentEmbedding]) -> None:
        """
        Add multiple documents to the vector store.
        
        Args:
            documents: List of DocumentEmbedding objects
        """
        if not documents:
            return
        
        # Prepare data for batch insertion
        ids = []
        texts = []
        embeddings = []
        metadatas = []
        
        for doc in documents:
            ids.append(doc.id)
            texts.append(doc.text)
            
            # Generate embedding if not provided
            if doc.embedding is None:
                doc.embedding = self._generate_embedding(doc.text)
            
            embeddings.append(doc.embedding)
            
            # Clean metadata to remove None values that ChromaDB doesn't accept
            clean_metadata = {}
            if doc.metadata:
                for k, v in doc.metadata.items():
                    if v is not None:
                        # Convert to string if not basic types
                        if isinstance(v, (str, int, float, bool)):
                            clean_metadata[k] = v
                        else:
                            clean_metadata[k] = str(v)
            
            metadatas.append(clean_metadata)
        
        # Batch insert
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Added {len(documents)} documents to collection")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def add_document(self, document: DocumentEmbedding) -> None:
        """Add a single document to the vector store."""
        self.add_documents([document])
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Metadata filter conditions
            
        Returns:
            List of search results with documents and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                })
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.
        
        Args:
            document_id: Document ID to retrieve
            
        Returns:
            Document data or None if not found
        """
        try:
            result = self.collection.get(
                ids=[document_id],
                include=["documents", "metadatas"]
            )
            
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'document': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.collection.delete(ids=[document_id])
            logger.info(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def update_document(self, document: DocumentEmbedding) -> bool:
        """
        Update an existing document.
        
        Args:
            document: Updated document data
            
        Returns:
            True if updated successfully
        """
        try:
            # Delete existing document
            self.delete_document(document.id)
            
            # Add updated document
            self.add_document(document)
            
            logger.info(f"Updated document: {document.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {document.id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'embedding_model': self.embedding_model_name,
                'persist_directory': str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Get all document IDs
            result = self.collection.get()
            if result['ids']:
                self.collection.delete(ids=result['ids'])
                logger.info(f"Cleared {len(result['ids'])} documents from collection")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise


class VectorStoreManager:
    """
    Manager for multiple vector store collections.
    Useful for organizing documents by type, source, or project.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.stores: Dict[str, VectorStore] = {}
    
    def get_store(self, collection_name: str, embedding_model: str = "all-MiniLM-L6-v2") -> VectorStore:
        """Get or create a vector store for the specified collection."""
        if collection_name not in self.stores:
            self.stores[collection_name] = VectorStore(
                collection_name=collection_name,
                persist_directory=self.persist_directory,
                embedding_model=embedding_model
            )
        return self.stores[collection_name]
    
    def list_collections(self) -> List[str]:
        """List all available collections."""
        return list(self.stores.keys())
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections."""
        stats = {}
        for name, store in self.stores.items():
            stats[name] = store.get_collection_stats()
        return stats


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize vector store
    vector_store = VectorStore(collection_name="test_documents")
    
    # Create sample documents
    documents = [
        DocumentEmbedding(
            text="ChromaDB is a vector database for AI applications",
            metadata={"source": "documentation", "type": "technical"}
        ),
        DocumentEmbedding(
            text="RAG systems combine retrieval and generation for better AI responses",
            metadata={"source": "research", "type": "academic"}
        )
    ]
    
    # Add documents
    vector_store.add_documents(documents)
    
    # Search
    results = vector_store.search("vector database", n_results=2)
    for result in results:
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Text: {result['document']}")
        print(f"Metadata: {result['metadata']}")
        print("-" * 50)
    
    # Get stats
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")