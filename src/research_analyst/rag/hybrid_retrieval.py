"""
Hybrid Retrieval System for Enhanced Accuracy

This module implements advanced retrieval strategies that combine multiple
approaches to improve accuracy in research analysis:

1. Semantic + Lexical Hybrid Search
2. Query Expansion and Reranking  
3. Context-Aware Retrieval
4. Multi-Model Ensemble
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np
from collections import defaultdict, Counter

from .retrieval import RAGRetriever, SearchResult, RetrievalContext
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class RetrievalScore:
    """Combined scores from different retrieval methods."""
    semantic_score: float
    lexical_score: float
    query_expansion_score: float
    context_score: float
    combined_score: float
    method_weights: Dict[str, float]


class QueryExpander:
    """Expands queries with synonyms and related terms for better retrieval."""
    
    def __init__(self):
        # Domain-specific synonym mappings for research/business content
        self.synonyms = {
            # AI/ML terms
            'artificial intelligence': ['AI', 'machine learning', 'ML', 'deep learning', 'neural networks'],
            'machine learning': ['ML', 'AI', 'artificial intelligence', 'predictive modeling'],
            'vector database': ['embedding store', 'vector store', 'semantic database'],
            'embedding': ['vector', 'representation', 'encoding'],
            
            # Business terms  
            'analysis': ['assessment', 'evaluation', 'examination', 'study'],
            'strategy': ['approach', 'plan', 'methodology', 'framework'],
            'performance': ['efficiency', 'effectiveness', 'results', 'outcomes'],
            'implementation': ['deployment', 'execution', 'rollout'],
            
            # Research terms
            'methodology': ['approach', 'method', 'technique', 'framework'],
            'findings': ['results', 'conclusions', 'discoveries', 'insights'],
            'limitations': ['constraints', 'challenges', 'restrictions'],
        }
        
        # Question pattern expansion
        self.question_patterns = {
            'what is': ['definition of', 'meaning of', 'explanation of'],
            'how to': ['method to', 'way to', 'approach to'],
            'benefits of': ['advantages of', 'pros of', 'value of'],
            'limitations of': ['drawbacks of', 'cons of', 'challenges of'],
        }
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original query
            max_expansions: Maximum number of expanded queries
            
        Returns:
            List of expanded queries including original
        """
        expansions = [query]  # Always include original
        query_lower = query.lower()
        
        # Expand with synonyms
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                for synonym in synonyms[:2]:  # Limit to 2 synonyms per term
                    expanded = query.replace(term, synonym, 1)
                    if expanded != query and expanded not in expansions:
                        expansions.append(expanded)
                        if len(expansions) >= max_expansions + 1:
                            break
        
        # Expand question patterns
        for pattern, alternatives in self.question_patterns.items():
            if pattern in query_lower:
                for alt in alternatives[:1]:  # One alternative per pattern
                    expanded = query_lower.replace(pattern, alt, 1)
                    if expanded not in expansions:
                        expansions.append(expanded)
                        if len(expansions) >= max_expansions + 1:
                            break
        
        return expansions[:max_expansions + 1]


class LexicalSearcher:
    """BM25-like lexical search for keyword matching."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = defaultdict(int)  # Document frequencies
        self.doc_lengths = {}  # Document lengths
        self.avg_doc_length = 0
        self.total_docs = 0
    
    def index_documents(self, documents: List[SearchResult]):
        """Index documents for lexical search."""
        all_lengths = []
        term_doc_counts = defaultdict(set)
        
        for doc in documents:
            doc_id = doc.chunk_id
            tokens = self._tokenize(doc.text)
            self.doc_lengths[doc_id] = len(tokens)
            all_lengths.append(len(tokens))
            
            # Track which documents contain each term
            unique_tokens = set(tokens)
            for token in unique_tokens:
                term_doc_counts[token].add(doc_id)
        
        # Calculate document frequencies
        self.doc_freqs = {term: len(docs) for term, docs in term_doc_counts.items()}
        self.avg_doc_length = np.mean(all_lengths) if all_lengths else 0
        self.total_docs = len(documents)
    
    def search(self, query: str, documents: List[SearchResult], n_results: int = 10) -> List[Tuple[SearchResult, float]]:
        """
        Perform BM25-like lexical search.
        
        Args:
            query: Search query
            documents: Documents to search
            n_results: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not documents:
            return []
        
        query_tokens = self._tokenize(query)
        scores = []
        
        for doc in documents:
            doc_tokens = self._tokenize(doc.text)
            score = self._calculate_bm25_score(query_tokens, doc_tokens, doc.chunk_id)
            scores.append((doc, score))
        
        # Sort by score and return top results
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_results]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return tokens
    
    def _calculate_bm25_score(self, query_tokens: List[str], doc_tokens: List[str], doc_id: str) -> float:
        """Calculate BM25 score."""
        doc_length = len(doc_tokens)
        if doc_length == 0:
            return 0.0
        
        score = 0.0
        doc_token_counts = Counter(doc_tokens)
        
        for token in query_tokens:
            if token in doc_token_counts:
                tf = doc_token_counts[token]  # Term frequency in document
                df = self.doc_freqs.get(token, 1)  # Document frequency
                idf = np.log((self.total_docs - df + 0.5) / (df + 0.5))
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                score += idf * (numerator / denominator)
        
        return score


class HybridRetriever:
    """
    Enhanced retriever that combines semantic and lexical search for better accuracy.
    """
    
    def __init__(
        self,
        semantic_retriever: RAGRetriever,
        embedding_service: Optional[EmbeddingService] = None,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            semantic_retriever: Base semantic RAG retriever
            embedding_service: Optional embedding service for reranking
            semantic_weight: Weight for semantic scores (0-1)
            lexical_weight: Weight for lexical scores (0-1)
        """
        self.semantic_retriever = semantic_retriever
        self.embedding_service = embedding_service
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        
        self.query_expander = QueryExpander()
        self.lexical_searcher = LexicalSearcher()
        
        # Normalize weights
        total_weight = semantic_weight + lexical_weight
        self.semantic_weight /= total_weight
        self.lexical_weight /= total_weight
        
        logger.info(f"HybridRetriever initialized (semantic: {self.semantic_weight:.2f}, lexical: {self.lexical_weight:.2f})")
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        similarity_threshold: float = 0.0,
        use_query_expansion: bool = True,
        rerank: bool = True
    ) -> RetrievalContext:
        """
        Perform hybrid search combining semantic and lexical approaches.
        
        Args:
            query: Search query
            n_results: Number of results to return
            similarity_threshold: Minimum similarity threshold
            use_query_expansion: Whether to expand query
            rerank: Whether to rerank results
            
        Returns:
            Enhanced RetrievalContext with combined results
        """
        # Step 1: Query expansion
        queries = [query]
        if use_query_expansion:
            queries = self.query_expander.expand_query(query, max_expansions=2)
        
        # Step 2: Semantic search for each query
        all_semantic_results = []
        for q in queries:
            semantic_context = self.semantic_retriever.search(
                query=q,
                n_results=n_results * 2,  # Get more for combining
                similarity_threshold=similarity_threshold
            )
            all_semantic_results.extend(semantic_context.results)
        
        # Remove duplicates from semantic results
        seen_ids = set()
        unique_semantic_results = []
        for result in all_semantic_results:
            if result.chunk_id not in seen_ids:
                seen_ids.add(result.chunk_id)
                unique_semantic_results.append(result)
        
        # Step 3: Lexical search
        self.lexical_searcher.index_documents(unique_semantic_results)
        lexical_results = self.lexical_searcher.search(
            query, unique_semantic_results, n_results * 2
        )
        
        # Step 4: Combine and rerank results
        combined_results = self._combine_results(
            query,
            unique_semantic_results,
            lexical_results,
            n_results
        )
        
        # Step 5: Apply reranking if enabled
        if rerank and self.embedding_service and len(combined_results) > 1:
            combined_results = self._rerank_results(query, combined_results)
        
        # Step 6: Create enhanced context
        final_results = combined_results[:n_results]
        
        if final_results:
            avg_similarity = np.mean([r.similarity for r in final_results])
        else:
            avg_similarity = 0.0
        
        return RetrievalContext(
            query=query,
            results=final_results,
            total_results=len(final_results),
            search_time=0.0,  # Would track in real implementation
            avg_similarity=avg_similarity
        )
    
    def _combine_results(
        self,
        query: str,
        semantic_results: List[SearchResult],
        lexical_results: List[Tuple[SearchResult, float]],
        n_results: int
    ) -> List[SearchResult]:
        """Combine semantic and lexical search results."""
        # Create lookup for lexical scores
        lexical_scores = {doc.chunk_id: score for doc, score in lexical_results}
        
        # Normalize lexical scores to 0-1 range
        if lexical_scores:
            max_lexical = max(lexical_scores.values())
            min_lexical = min(lexical_scores.values())
            if max_lexical > min_lexical:
                for doc_id in lexical_scores:
                    lexical_scores[doc_id] = (lexical_scores[doc_id] - min_lexical) / (max_lexical - min_lexical)
        
        # Calculate combined scores
        combined_results = []
        for result in semantic_results:
            semantic_score = result.similarity
            lexical_score = lexical_scores.get(result.chunk_id, 0.0)
            
            # Combined score
            combined_score = (
                self.semantic_weight * semantic_score +
                self.lexical_weight * lexical_score
            )
            
            # Create new result with combined score
            enhanced_result = SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                similarity=combined_score,  # Use combined score as similarity
                distance=1 - combined_score,  # Inverse of similarity
                metadata={
                    **result.metadata,
                    'hybrid_scores': {
                        'semantic': semantic_score,
                        'lexical': lexical_score,
                        'combined': combined_score,
                        'weights': {
                            'semantic': self.semantic_weight,
                            'lexical': self.lexical_weight
                        }
                    }
                },
                source_document=result.source_document,
                chunk_index=result.chunk_index
            )
            combined_results.append(enhanced_result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.similarity, reverse=True)
        return combined_results
    
    def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank results using cross-encoder or advanced similarity."""
        if not self.embedding_service or len(results) <= 1:
            return results
        
        try:
            # Get query embedding
            query_embedding = self.embedding_service.embed_text(query)
            
            # Calculate more precise similarities
            reranked_results = []
            for result in results:
                # Get document embedding
                doc_embedding = self.embedding_service.embed_text(result.text)
                
                # Calculate cosine similarity
                q_vec = np.array(query_embedding.embedding)
                d_vec = np.array(doc_embedding.embedding)
                
                cosine_sim = np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
                
                # Combine with existing score
                original_score = result.similarity
                reranked_score = 0.6 * original_score + 0.4 * cosine_sim
                
                # Update result
                result.similarity = reranked_score
                result.distance = 1 - reranked_score
                result.metadata['reranking_applied'] = True
                result.metadata['original_score'] = original_score
                result.metadata['reranked_score'] = reranked_score
                
                reranked_results.append(result)
            
            # Sort by new scores
            reranked_results.sort(key=lambda x: x.similarity, reverse=True)
            return reranked_results
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return results


class AccuracyEnhancedRAG:
    """
    Complete accuracy-enhanced RAG system that integrates all improvements.
    """
    
    def __init__(
        self,
        base_retriever: RAGRetriever,
        embedding_model: str = "high_accuracy",  # Use preset for best accuracy
        enable_hybrid_search: bool = True,
        enable_query_expansion: bool = True,
        enable_reranking: bool = True
    ):
        """
        Initialize accuracy-enhanced RAG system.
        
        Args:
            base_retriever: Base RAG retriever
            embedding_model: Embedding model preset or name
            enable_hybrid_search: Enable semantic + lexical hybrid search
            enable_query_expansion: Enable query expansion
            enable_reranking: Enable result reranking
        """
        self.base_retriever = base_retriever
        
        # Initialize enhanced embedding service
        self.embedding_service = EmbeddingService(
            model_name=embedding_model,
            enable_cache=True
        )
        
        # Initialize hybrid retriever if enabled
        if enable_hybrid_search:
            self.hybrid_retriever = HybridRetriever(
                semantic_retriever=base_retriever,
                embedding_service=self.embedding_service
            )
        else:
            self.hybrid_retriever = None
        
        self.enable_query_expansion = enable_query_expansion
        self.enable_reranking = enable_reranking
        
        logger.info("AccuracyEnhancedRAG initialized with advanced retrieval capabilities")
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        similarity_threshold: float = 0.1,  # Lower threshold for more recall
        context_length: int = 2000
    ) -> RetrievalContext:
        """
        Perform enhanced search with all accuracy improvements.
        
        Args:
            query: Search query
            n_results: Number of results
            similarity_threshold: Minimum similarity
            context_length: Maximum context length
            
        Returns:
            Enhanced retrieval context
        """
        if self.hybrid_retriever:
            # Use hybrid search for best accuracy
            context = self.hybrid_retriever.search(
                query=query,
                n_results=n_results,
                similarity_threshold=similarity_threshold,
                use_query_expansion=self.enable_query_expansion,
                rerank=self.enable_reranking
            )
        else:
            # Fall back to semantic search
            context = self.base_retriever.search(
                query=query,
                n_results=n_results,
                similarity_threshold=similarity_threshold
            )
        
        # Truncate context if needed
        context_text = context.get_context_text(max_length=context_length)
        
        return context
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        base_stats = self.base_retriever.get_collection_stats()
        
        stats = {
            'base_retriever_stats': base_stats,
            'embedding_service_stats': self.embedding_service.get_model_info(),
            'hybrid_search_enabled': self.hybrid_retriever is not None,
            'query_expansion_enabled': self.enable_query_expansion,
            'reranking_enabled': self.enable_reranking
        }
        
        return stats


if __name__ == "__main__":
    # Example usage would go here
    logging.basicConfig(level=logging.INFO)
    print("HybridRetrieval module loaded successfully")