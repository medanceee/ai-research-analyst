"""
Enhanced Research Analyst with Accuracy Improvements

This module integrates all accuracy enhancements into a comprehensive
research analysis system with improved reliability and confidence scoring.

Key Improvements:
1. Upgraded embedding models for better semantic understanding
2. Hybrid retrieval combining semantic and lexical search
3. Advanced contradiction detection using NLP
4. Cross-source fact verification
5. Enhanced LLM prompts with structured analysis
6. Confidence scoring and uncertainty quantification
7. Document quality assessment and filtering
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Import enhanced components
from .rag.rag_pipeline import RAGPipeline
from .rag.hybrid_retrieval import AccuracyEnhancedRAG
from .rag.embeddings import EmbeddingService
from .analysis.contradiction_detector import AdvancedContradictionDetector, Contradiction
from .analysis.fact_verifier import CrossSourceFactVerifier, FactVerification, Claim
from .workflow.workflow import ResearchWorkflow, ResearchResult

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy and confidence metrics for research results."""
    overall_confidence: float = 0.0
    source_credibility_score: float = 0.0
    contradiction_risk: float = 0.0
    fact_verification_score: float = 0.0
    evidence_strength: str = "insufficient"
    uncertainty_level: str = "high"
    
    # Detailed breakdowns
    retrieval_quality: float = 0.0
    source_diversity: float = 0.0
    temporal_consistency: float = 0.0
    cross_validation_score: float = 0.0
    
    def calculate_overall_confidence(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall confidence score from component metrics."""
        if weights is None:
            weights = {
                'source_credibility': 0.25,
                'fact_verification': 0.25,
                'retrieval_quality': 0.15,
                'contradiction_risk': -0.2,  # Negative weight (contradictions reduce confidence)
                'source_diversity': 0.1,
                'cross_validation': 0.15
            }
        
        confidence = (
            weights['source_credibility'] * self.source_credibility_score +
            weights['fact_verification'] * self.fact_verification_score +
            weights['retrieval_quality'] * self.retrieval_quality +
            weights['contradiction_risk'] * (1 - self.contradiction_risk) +
            weights['source_diversity'] * self.source_diversity +
            weights['cross_validation'] * self.cross_validation_score
        )
        
        # Ensure confidence is between 0 and 1
        self.overall_confidence = max(0.0, min(1.0, confidence))
        
        # Set uncertainty level based on confidence
        if self.overall_confidence >= 0.8:
            self.uncertainty_level = "low"
        elif self.overall_confidence >= 0.6:
            self.uncertainty_level = "medium"
        else:
            self.uncertainty_level = "high"
        
        return self.overall_confidence


@dataclass
class EnhancedResearchResult(ResearchResult):
    """Enhanced research result with accuracy metrics and uncertainty quantification."""
    accuracy_metrics: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    contradictions_detected: List[Contradiction] = field(default_factory=list)
    fact_verifications: List[FactVerification] = field(default_factory=list)
    quality_warnings: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    def to_enhanced_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all enhancement data."""
        base_dict = self.to_dict()
        
        base_dict.update({
            'accuracy_metrics': {
                'overall_confidence': self.accuracy_metrics.overall_confidence,
                'source_credibility': self.accuracy_metrics.source_credibility_score,
                'contradiction_risk': self.accuracy_metrics.contradiction_risk,
                'fact_verification': self.accuracy_metrics.fact_verification_score,
                'evidence_strength': self.accuracy_metrics.evidence_strength,
                'uncertainty_level': self.accuracy_metrics.uncertainty_level,
                'retrieval_quality': self.accuracy_metrics.retrieval_quality,
                'source_diversity': self.accuracy_metrics.source_diversity
            },
            'contradictions': [c.to_dict() for c in self.contradictions_detected],
            'fact_verifications': [fv.to_dict() for fv in self.fact_verifications],
            'quality_warnings': self.quality_warnings,
            'improvement_suggestions': self.improvement_suggestions
        })
        
        return base_dict


class EnhancedResearchAnalyst:
    """
    Enhanced research analyst with comprehensive accuracy improvements.
    """
    
    def __init__(
        self,
        collection_name: str = "enhanced_research",
        persist_directory: str = "./enhanced_research_data",
        embedding_model: str = "high_accuracy",
        enable_fact_verification: bool = True,
        enable_contradiction_detection: bool = True,
        enable_hybrid_search: bool = True,
        min_confidence_threshold: float = 0.6
    ):
        """
        Initialize enhanced research analyst.
        
        Args:
            collection_name: Name for document collection
            persist_directory: Directory for persistent storage
            embedding_model: Embedding model preset (high_accuracy, medium_accuracy, fast)
            enable_fact_verification: Enable cross-source fact verification
            enable_contradiction_detection: Enable advanced contradiction detection
            enable_hybrid_search: Enable hybrid semantic+lexical search
            min_confidence_threshold: Minimum confidence for including results
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.min_confidence_threshold = min_confidence_threshold
        
        # Initialize enhanced embedding service
        self.embedding_service = EmbeddingService(
            model_name=embedding_model,
            enable_cache=True
        )
        
        # Initialize base RAG pipeline
        self.base_rag = RAGPipeline(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model
        )
        
        # Initialize accuracy-enhanced RAG if enabled
        if enable_hybrid_search:
            self.enhanced_rag = AccuracyEnhancedRAG(
                base_retriever=self.base_rag.retriever,
                embedding_model=embedding_model,
                enable_hybrid_search=True,
                enable_query_expansion=True,
                enable_reranking=True
            )
        else:
            self.enhanced_rag = None
        
        # Initialize analysis components
        if enable_contradiction_detection:
            self.contradiction_detector = AdvancedContradictionDetector(
                embedding_service=self.embedding_service
            )
        else:
            self.contradiction_detector = None
        
        if enable_fact_verification:
            self.fact_verifier = CrossSourceFactVerifier(
                embedding_service=self.embedding_service
            )
        else:
            self.fact_verifier = None
        
        # Initialize enhanced workflow
        rag_for_workflow = self.enhanced_rag if self.enhanced_rag else self.base_rag
        self.workflow = ResearchWorkflow(rag_for_workflow)
        
        logger.info(f"EnhancedResearchAnalyst initialized with {embedding_model} embeddings")
    
    def analyze_research_query(
        self,
        research_query: str,
        documents: List[str],
        save_report: Optional[str] = None,
        save_markdown: Optional[str] = None,
        markdown_title: Optional[str] = None,
        include_detailed_analysis: bool = True
    ) -> EnhancedResearchResult:
        """
        Perform comprehensive research analysis with accuracy enhancements.
        
        Args:
            research_query: Research question to analyze
            documents: List of document paths/URLs to analyze
            save_report: Optional path to save JSON report
            save_markdown: Optional path to save markdown report
            markdown_title: Optional custom title for markdown report
            include_detailed_analysis: Include detailed accuracy analysis
            
        Returns:
            Enhanced research result with accuracy metrics
        """
        start_time = time.time()
        
        # Step 1: Run base research workflow
        logger.info(f"Starting enhanced research analysis for: {research_query}")
        base_result = self.workflow.run_research(
            research_query=research_query,
            documents=documents
        )
        
        # Step 2: Perform accuracy enhancements if requested
        if include_detailed_analysis:
            accuracy_metrics, contradictions, fact_verifications, warnings, suggestions = \
                self._perform_accuracy_analysis(research_query, documents)
        else:
            accuracy_metrics = AccuracyMetrics()
            contradictions = []
            fact_verifications = []
            warnings = []
            suggestions = []
        
        # Step 3: Create enhanced result
        enhanced_result = EnhancedResearchResult(
            query=base_result.query,
            executive_summary=base_result.executive_summary,
            detailed_analysis=base_result.detailed_analysis,
            key_insights=base_result.key_insights,
            recommendations=base_result.recommendations,
            sources_used=base_result.sources_used,
            contradictions=base_result.contradictions,
            confidence_score=accuracy_metrics.overall_confidence,
            processing_time=time.time() - start_time,
            document_count=base_result.document_count,
            chunk_count=base_result.chunk_count,
            # Enhanced fields
            accuracy_metrics=accuracy_metrics,
            contradictions_detected=contradictions,
            fact_verifications=fact_verifications,
            quality_warnings=warnings,
            improvement_suggestions=suggestions
        )
        
        # Step 4: Apply confidence filtering if enabled
        if enhanced_result.confidence_score < self.min_confidence_threshold:
            enhanced_result.quality_warnings.append(
                f"Analysis confidence ({enhanced_result.confidence_score:.2f}) "
                f"below threshold ({self.min_confidence_threshold}). "
                f"Consider adding more sources or refining query."
            )
        
        # Step 5: Save reports if requested
        if save_report:
            self._save_enhanced_report(enhanced_result, save_report)
        
        if save_markdown:
            self._save_markdown_report(enhanced_result, save_markdown, markdown_title)
        
        logger.info(f"Enhanced analysis completed in {enhanced_result.processing_time:.2f}s "
                   f"(confidence: {enhanced_result.confidence_score:.2f})")
        
        return enhanced_result
    
    def _perform_accuracy_analysis(
        self,
        research_query: str,
        documents: List[str]
    ) -> Tuple[AccuracyMetrics, List[Contradiction], List[FactVerification], List[str], List[str]]:
        """Perform comprehensive accuracy analysis."""
        
        # Get all document chunks for analysis
        all_chunks = self._get_document_chunks_for_analysis()
        
        # Initialize metrics
        metrics = AccuracyMetrics()
        contradictions = []
        fact_verifications = []
        warnings = []
        suggestions = []
        
        # Contradiction detection
        if self.contradiction_detector and len(all_chunks) >= 2:
            contradictions = self.contradiction_detector.detect_contradictions(
                text_chunks=all_chunks,
                min_confidence=0.5
            )
            
            if contradictions:
                metrics.contradiction_risk = min(1.0, len(contradictions) * 0.2)
                warnings.append(f"Detected {len(contradictions)} potential contradictions")
                suggestions.append("Review contradictions and resolve conflicting information")
        
        # Fact verification
        if self.fact_verifier and len(all_chunks) >= 3:
            # Extract claims for verification
            claims = []
            for chunk in all_chunks[:5]:  # Limit to first 5 chunks for performance
                chunk_claims = self.fact_verifier.claim_extractor.extract_claims(
                    text=chunk.get('text', ''),
                    source=chunk.get('source', ''),
                    chunk_id=chunk.get('chunk_id', '')
                )
                claims.extend(chunk_claims[:3])  # Top 3 claims per chunk
            
            if claims:
                fact_verifications = self.fact_verifier.verify_claims(
                    claims=claims,
                    source_documents=all_chunks,
                    min_confidence=0.4
                )
                
                # Calculate fact verification score
                if fact_verifications:
                    verified_count = sum(1 for fv in fact_verifications 
                                       if fv.verification_status.value in ['verified', 'partially_verified'])
                    metrics.fact_verification_score = verified_count / len(fact_verifications)
                    
                    disputed_count = sum(1 for fv in fact_verifications 
                                       if fv.verification_status.value == 'disputed')
                    if disputed_count > 0:
                        warnings.append(f"{disputed_count} claims disputed across sources")
        
        # Calculate other metrics
        metrics.retrieval_quality = self._assess_retrieval_quality(all_chunks)
        metrics.source_credibility_score = self._assess_source_credibility(all_chunks)
        metrics.source_diversity = self._calculate_source_diversity(all_chunks)
        metrics.cross_validation_score = self._calculate_cross_validation_score(all_chunks)
        
        # Determine evidence strength
        if metrics.fact_verification_score >= 0.8 and metrics.contradiction_risk <= 0.2:
            metrics.evidence_strength = "strong"
        elif metrics.fact_verification_score >= 0.6 and metrics.contradiction_risk <= 0.4:
            metrics.evidence_strength = "moderate"
        else:
            metrics.evidence_strength = "weak"
        
        # Calculate overall confidence
        metrics.calculate_overall_confidence()
        
        # Generate improvement suggestions
        if metrics.source_diversity < 0.5:
            suggestions.append("Consider adding sources from different types/domains")
        if metrics.contradiction_risk > 0.3:
            suggestions.append("Investigate and resolve contradictory information")
        if len(all_chunks) < 5:
            suggestions.append("Add more source documents for better validation")
        
        return metrics, contradictions, fact_verifications, warnings, suggestions
    
    def _get_document_chunks_for_analysis(self) -> List[Dict[str, Any]]:
        """Get all document chunks for analysis."""
        # This would extract all chunks from the vector store
        # For now, we'll use a simplified implementation
        documents = self.base_rag.list_documents()
        
        all_chunks = []
        for doc in documents:
            chunks = self.base_rag.retriever.get_document_chunks(doc.get('doc_id', ''))
            for chunk in chunks:
                all_chunks.append({
                    'text': chunk.text,
                    'source': chunk.metadata.get('source_file', ''),
                    'chunk_id': chunk.chunk_id,
                    'metadata': chunk.metadata
                })
        
        return all_chunks
    
    def _assess_retrieval_quality(self, chunks: List[Dict[str, Any]]) -> float:
        """Assess quality of retrieved information."""
        if not chunks:
            return 0.0
        
        # Simple heuristics for retrieval quality
        avg_chunk_length = sum(len(chunk.get('text', '')) for chunk in chunks) / len(chunks)
        length_score = min(1.0, avg_chunk_length / 500)  # Normalize to reasonable length
        
        # Check for diverse sources
        sources = set(chunk.get('source', '') for chunk in chunks)
        diversity_score = min(1.0, len(sources) / 3)  # Normalize to 3+ sources
        
        return (length_score + diversity_score) / 2
    
    def _assess_source_credibility(self, chunks: List[Dict[str, Any]]) -> float:
        """Assess overall source credibility."""
        if not chunks:
            return 0.0
        
        # Simple credibility assessment based on source characteristics
        credibility_scores = []
        
        for chunk in chunks:
            source = chunk.get('source', '').lower()
            score = 0.5  # Base score
            
            # Academic sources
            if any(indicator in source for indicator in ['pdf', 'journal', 'arxiv', 'doi']):
                score += 0.3
            
            # Government sources
            if '.gov' in source or 'official' in source:
                score += 0.25
            
            # News sources
            if any(news in source for news in ['news', 'times', 'post', 'reuters']):
                score += 0.1
            
            # Blog/social sources (lower credibility)
            if any(low in source for low in ['blog', 'twitter', 'facebook']):
                score -= 0.2
            
            credibility_scores.append(max(0.0, min(1.0, score)))
        
        return sum(credibility_scores) / len(credibility_scores)
    
    def _calculate_source_diversity(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate diversity of information sources."""
        if not chunks:
            return 0.0
        
        sources = set(chunk.get('source', '') for chunk in chunks)
        # Normalize by expected maximum diversity (e.g., 5 different source types)
        return min(1.0, len(sources) / 5)
    
    def _calculate_cross_validation_score(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate how well information is cross-validated across sources."""
        if len(chunks) < 2:
            return 0.0
        
        # Simple cross-validation: more sources = better validation
        sources = set(chunk.get('source', '') for chunk in chunks)
        
        if len(sources) >= 4:
            return 0.9
        elif len(sources) >= 3:
            return 0.7
        elif len(sources) >= 2:
            return 0.5
        else:
            return 0.2
    
    def _save_enhanced_report(self, result: EnhancedResearchResult, output_path: str):
        """Save enhanced research report with all accuracy data."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_enhanced_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Enhanced research report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save enhanced report: {e}")
    
    def _save_markdown_report(self, result: EnhancedResearchResult, output_path: str, title: Optional[str] = None):
        """Save research result as formatted markdown report."""
        try:
            from .output.markdown_formatter import MarkdownFormatter
            
            formatter = MarkdownFormatter()
            success = formatter.save_markdown_report(result, output_path, title)
            
            if success:
                logger.info(f"Markdown report saved to {output_path}")
            else:
                logger.error(f"Failed to save markdown report to {output_path}")
                
        except ImportError as e:
            logger.error(f"Markdown formatter not available: {e}")
        except Exception as e:
            logger.error(f"Failed to save markdown report: {e}")
    
    def get_markdown_report(self, result: EnhancedResearchResult, title: Optional[str] = None) -> str:
        """Get research result as formatted markdown string."""
        try:
            from .output.markdown_formatter import MarkdownFormatter
            
            formatter = MarkdownFormatter()
            return formatter.format_research_result(result, title)
            
        except ImportError as e:
            logger.error(f"Markdown formatter not available: {e}")
            return f"# Research Analysis\n\nMarkdown formatting unavailable: {e}"
        except Exception as e:
            logger.error(f"Failed to generate markdown: {e}")
            return f"# Research Analysis\n\nError generating markdown: {e}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and capabilities."""
        base_stats = self.base_rag.get_pipeline_stats()
        
        status = {
            'base_pipeline_stats': base_stats,
            'embedding_service': self.embedding_service.get_model_info(),
            'enhanced_features': {
                'hybrid_search': self.enhanced_rag is not None,
                'contradiction_detection': self.contradiction_detector is not None,
                'fact_verification': self.fact_verifier is not None
            },
            'confidence_threshold': self.min_confidence_threshold,
            'collection_name': self.collection_name
        }
        
        if self.enhanced_rag:
            status['enhanced_rag_stats'] = self.enhanced_rag.get_system_stats()
        
        return status


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize enhanced research analyst
    analyst = EnhancedResearchAnalyst(
        collection_name="demo_enhanced",
        embedding_model="high_accuracy",
        enable_fact_verification=True,
        enable_contradiction_detection=True,
        min_confidence_threshold=0.6
    )
    
    # Example research query
    research_query = "What are the key benefits and limitations of vector databases for AI applications?"
    
    # Example documents (would be real file paths in practice)
    documents = [
        "sample_vector_db_paper.pdf",
        "industry_analysis_report.pdf",
        "technical_benchmark_study.pdf"
    ]
    
    print(f"Enhanced Research Analyst initialized!")
    print(f"System Status: {analyst.get_system_status()}")
    print(f"Ready to analyze: {research_query}")
    
    # Note: Actual analysis would require real documents
    # result = analyst.analyze_research_query(research_query, documents)
    # print(f"Analysis confidence: {result.confidence_score:.2f}")
    # print(f"Quality warnings: {result.quality_warnings}")