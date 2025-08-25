"""
Embedding Generation Service for RAG Pipeline

This module provides a comprehensive embedding service that supports multiple
embedding models, caching, and batch processing for optimal performance.

Key Components:
- EmbeddingService: Main interface for generating embeddings
- ModelManager: Handles loading and switching between embedding models
- EmbeddingCache: Caches embeddings to avoid recomputation
- BatchProcessor: Processes multiple texts efficiently
"""

import hashlib
import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import logging
import time

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Represents an embedding result with metadata."""
    text: str
    embedding: List[float]
    model_name: str
    embedding_dim: int
    processing_time: float
    text_hash: str
    
    @classmethod
    def from_text_and_embedding(
        cls, 
        text: str, 
        embedding: Union[List[float], np.ndarray], 
        model_name: str, 
        processing_time: float = 0.0
    ) -> 'EmbeddingResult':
        """Create EmbeddingResult from text and embedding array."""
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        return cls(
            text=text,
            embedding=embedding,
            model_name=model_name,
            embedding_dim=len(embedding),
            processing_time=processing_time,
            text_hash=text_hash
        )


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Encode text(s) into embeddings."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name/identifier."""
        pass


class SentenceTransformerModel(BaseEmbeddingModel):
    """Enhanced Sentence Transformer embedding model wrapper with multi-model support."""
    
    # Recommended models by accuracy (ordered by performance)
    ACCURACY_MODELS = {
        'high_accuracy': 'all-mpnet-base-v2',      # 768D, best quality
        'medium_accuracy': 'all-MiniLM-L12-v2',   # 384D, good balance  
        'fast': 'all-MiniLM-L6-v2',               # 384D, fastest
        'multilingual': 'paraphrase-multilingual-mpnet-base-v2',  # 768D, multilingual
        'scientific': 'allenai-specter',          # 768D, scientific papers
    }
    
    def __init__(self, model_name: str = "all-mpnet-base-v2", device: str = "auto"):
        """
        Initialize Sentence Transformer model.
        
        Args:
            model_name: Name of the sentence transformer model (or preset key)
            device: Device to run on ('cpu', 'cuda', or 'auto')
        """
        # Handle preset model names
        if model_name in self.ACCURACY_MODELS:
            self.model_name = self.ACCURACY_MODELS[model_name]
            self.preset = model_name
        else:
            self.model_name = model_name
            self.preset = None
            
        # Handle 'auto' device selection
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._model = None
        self._embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self._model = SentenceTransformer(
                self.model_name, 
                device=self.device
            )
            # Get embedding dimension by encoding a sample text
            sample_embedding = self._model.encode("test", convert_to_tensor=False)
            self._embedding_dim = len(sample_embedding)
            logger.info(f"Loaded model {self.model_name} with {self._embedding_dim}D embeddings")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Encode text(s) into embeddings."""
        if isinstance(texts, str):
            embedding = self._model.encode(texts, convert_to_tensor=False)
            return embedding.tolist()
        else:
            embeddings = self._model.encode(texts, convert_to_tensor=False)
            return [emb.tolist() for emb in embeddings]
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        return self._embedding_dim
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name


class EmbeddingCache:
    """
    SQLite-based cache for embeddings to avoid recomputation.
    """
    
    def __init__(self, cache_file: str = "embeddings_cache.db", max_size: int = 100000):
        """
        Initialize embedding cache.
        
        Args:
            cache_file: Path to SQLite cache file
            max_size: Maximum number of cached embeddings
        """
        self.cache_file = Path(cache_file)
        self.max_size = max_size
        self._init_cache()
    
    def _init_cache(self):
        """Initialize the SQLite cache database."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.cache_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    model_name TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_hash 
                ON embeddings (model_name, text_hash)
            """)
            
            conn.commit()
        
        logger.info(f"Initialized embedding cache: {self.cache_file}")
    
    def get(self, text: str, model_name: str) -> Optional[EmbeddingResult]:
        """Retrieve embedding from cache."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT text, embedding, model_name, embedding_dim
                    FROM embeddings 
                    WHERE text_hash = ? AND model_name = ?
                """, (text_hash, model_name))
                
                row = cursor.fetchone()
                if row:
                    _, embedding_blob, _, embedding_dim = row
                    embedding = json.loads(embedding_blob.decode('utf-8'))
                    
                    return EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        model_name=model_name,
                        embedding_dim=embedding_dim,
                        processing_time=0.0,  # Cached, no processing time
                        text_hash=text_hash
                    )
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def put(self, result: EmbeddingResult):
        """Store embedding in cache."""
        try:
            embedding_blob = json.dumps(result.embedding).encode('utf-8')
            
            with sqlite3.connect(self.cache_file) as conn:
                # Check cache size and clean if necessary
                self._cleanup_cache(conn)
                
                conn.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (text_hash, text, embedding, model_name, embedding_dim)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    result.text_hash,
                    result.text,
                    embedding_blob,
                    result.model_name,
                    result.embedding_dim
                ))
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _cleanup_cache(self, conn: sqlite3.Connection):
        """Remove old entries if cache is too large."""
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        
        if count >= self.max_size:
            # Remove oldest 10% of entries
            remove_count = max(1, count // 10)
            cursor.execute("""
                DELETE FROM embeddings 
                WHERE rowid IN (
                    SELECT rowid FROM embeddings 
                    ORDER BY created_at ASC 
                    LIMIT ?
                )
            """, (remove_count,))
            logger.info(f"Cleaned {remove_count} old cache entries")
    
    def clear(self):
        """Clear all cached embeddings."""
        with sqlite3.connect(self.cache_file) as conn:
            conn.execute("DELETE FROM embeddings")
            conn.commit()
        logger.info("Cleared embedding cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                # Total count
                cursor.execute("SELECT COUNT(*) FROM embeddings")
                total_count = cursor.fetchone()[0]
                
                # Count by model
                cursor.execute("""
                    SELECT model_name, COUNT(*) 
                    FROM embeddings 
                    GROUP BY model_name
                """)
                model_counts = dict(cursor.fetchall())
                
                return {
                    'total_embeddings': total_count,
                    'model_counts': model_counts,
                    'cache_file': str(self.cache_file),
                    'max_size': self.max_size
                }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}


class EmbeddingService:
    """
    Main embedding service that coordinates model loading, caching, and batch processing.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_file: Optional[str] = None,
        enable_cache: bool = True,
        device: str = "auto"
    ):
        """
        Initialize embedding service.
        
        Args:
            model_name: Name of the embedding model to use
            cache_file: Path to cache file (None for default)
            enable_cache: Whether to enable caching
            device: Device to run model on
        """
        self.model_name = model_name
        self.enable_cache = enable_cache
        
        # Initialize model
        self.model = SentenceTransformerModel(model_name, device)
        
        # Initialize cache
        if enable_cache:
            cache_file = cache_file or f"embeddings_cache_{model_name.replace('/', '_')}.db"
            self.cache = EmbeddingCache(cache_file)
        else:
            self.cache = None
        
        logger.info(f"EmbeddingService initialized with model: {model_name}")
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            EmbeddingResult with embedding and metadata
        """
        # Check cache first
        if self.cache:
            cached_result = self.cache.get(text, self.model_name)
            if cached_result:
                return cached_result
        
        # Generate embedding
        start_time = time.time()
        embedding = self.model.encode(text)
        processing_time = time.time() - start_time
        
        # Create result
        result = EmbeddingResult.from_text_and_embedding(
            text=text,
            embedding=embedding,
            model_name=self.model_name,
            processing_time=processing_time
        )
        
        # Cache result
        if self.cache:
            self.cache.put(result)
        
        return result
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of EmbeddingResult objects
        """
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        if self.cache:
            for i, text in enumerate(texts):
                cached_result = self.cache.get(text, self.model_name)
                if cached_result:
                    results.append((i, cached_result))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Process uncached texts in batches
        if uncached_texts:
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i + batch_size]
                batch_indices = uncached_indices[i:i + batch_size]
                
                start_time = time.time()
                batch_embeddings = self.model.encode(batch_texts)
                processing_time = (time.time() - start_time) / len(batch_texts)
                
                # Create results for batch
                for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                    result = EmbeddingResult.from_text_and_embedding(
                        text=text,
                        embedding=embedding,
                        model_name=self.model_name,
                        processing_time=processing_time
                    )
                    
                    # Cache result
                    if self.cache:
                        self.cache.put(result)
                    
                    results.append((batch_indices[j], result))
        
        # Sort results by original order
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.model.get_embedding_dim()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.get_embedding_dimension(),
            'cache_enabled': self.enable_cache,
            'cache_stats': self.cache.get_stats() if self.cache else None
        }
    
    def switch_model(self, model_name: str):
        """
        Switch to a different embedding model.
        
        Args:
            model_name: Name of the new model to use
        """
        if model_name != self.model_name:
            self.model_name = model_name
            self.model = SentenceTransformerModel(model_name)
            
            # Update cache if enabled
            if self.enable_cache:
                cache_file = f"embeddings_cache_{model_name.replace('/', '_')}.db"
                self.cache = EmbeddingCache(cache_file)
            
            logger.info(f"Switched to model: {model_name}")
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize embedding service
    embedding_service = EmbeddingService(
        model_name="all-MiniLM-L6-v2",
        enable_cache=True
    )
    
    # Test single embedding
    text = "ChromaDB is a vector database for AI applications"
    result = embedding_service.embed_text(text)
    print(f"Text: {result.text}")
    print(f"Embedding dimension: {result.embedding_dim}")
    print(f"Processing time: {result.processing_time:.4f}s")
    
    # Test batch embeddings
    texts = [
        "Vector databases store high-dimensional embeddings",
        "RAG systems combine retrieval and generation",
        "Semantic search uses embeddings for similarity matching",
        "ChromaDB provides persistent storage for vectors"
    ]
    
    results = embedding_service.embed_texts(texts, batch_size=2)
    print(f"\nProcessed {len(results)} texts")
    for i, result in enumerate(results):
        print(f"Text {i+1}: {result.text[:50]}... (dim: {result.embedding_dim})")
    
    # Test cache performance
    print("\n=== Cache Performance Test ===")
    start_time = time.time()
    result_cached = embedding_service.embed_text(text)  # Should be cached
    cached_time = time.time() - start_time
    print(f"Cached lookup time: {cached_time:.6f}s")
    
    # Get service info
    info = embedding_service.get_model_info()
    print(f"\n=== Service Info ===")
    print(f"Model: {info['model_name']}")
    print(f"Embedding dimension: {info['embedding_dim']}")
    print(f"Cache enabled: {info['cache_enabled']}")
    if info['cache_stats']:
        print(f"Cache stats: {info['cache_stats']}")