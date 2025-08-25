"""
Cross-Source Fact Verification System

This module implements a comprehensive fact verification system that validates
claims by cross-referencing multiple sources, assessing source credibility,
and providing confidence scores for factual assertions.

Key Features:
- Multi-source claim verification
- Source credibility assessment
- Evidence aggregation and weighting
- Temporal consistency checking
- Authority and expertise evaluation
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import logging
import numpy as np
from datetime import datetime, timedelta

from .contradiction_detector import Claim, ClaimExtractor

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of fact verification."""
    VERIFIED = "verified"           # Strong consensus across sources
    PARTIALLY_VERIFIED = "partial"  # Some supporting evidence
    UNVERIFIED = "unverified"      # No supporting evidence
    DISPUTED = "disputed"          # Contradictory evidence
    INSUFFICIENT = "insufficient"   # Not enough sources to verify


class SourceType(Enum):
    """Types of information sources."""
    ACADEMIC_PAPER = "academic"
    NEWS_ARTICLE = "news" 
    GOVERNMENT_REPORT = "government"
    CORPORATE_REPORT = "corporate"
    BLOG_POST = "blog"
    SOCIAL_MEDIA = "social"
    ENCYCLOPEDIA = "encyclopedia"
    UNKNOWN = "unknown"


@dataclass
class SourceCredibility:
    """Assessment of source credibility."""
    source_id: str
    source_type: SourceType
    authority_score: float = 0.5    # 0-1 scale
    recency_score: float = 0.5      # 0-1 based on publication date
    citation_score: float = 0.5     # Based on citations/references
    consistency_score: float = 0.5  # Internal consistency
    overall_credibility: float = 0.5
    
    def calculate_overall_credibility(self, weights: Optional[Dict[str, float]] = None):
        """Calculate overall credibility score."""
        if weights is None:
            weights = {
                'authority': 0.3,
                'recency': 0.2,
                'citation': 0.3,
                'consistency': 0.2
            }
        
        self.overall_credibility = (
            weights['authority'] * self.authority_score +
            weights['recency'] * self.recency_score +
            weights['citation'] * self.citation_score +
            weights['consistency'] * self.consistency_score
        )
        
        return self.overall_credibility


@dataclass
class Evidence:
    """Supporting or contradicting evidence for a claim."""
    text: str
    source_id: str
    chunk_id: str
    credibility: SourceCredibility
    support_strength: float = 0.0  # -1 (contradicts) to 1 (supports)
    confidence: float = 0.0
    semantic_similarity: float = 0.0
    
    def is_supporting(self) -> bool:
        """Check if evidence supports the claim."""
        return self.support_strength > 0.3
    
    def is_contradicting(self) -> bool:
        """Check if evidence contradicts the claim."""
        return self.support_strength < -0.3


@dataclass
class FactVerification:
    """Result of fact verification for a claim."""
    original_claim: Claim
    verification_status: VerificationStatus
    confidence_score: float
    supporting_evidence: List[Evidence] = field(default_factory=list)
    contradicting_evidence: List[Evidence] = field(default_factory=list)
    neutral_evidence: List[Evidence] = field(default_factory=list)
    consensus_score: float = 0.0    # Agreement level across sources
    source_diversity: float = 0.0   # Diversity of source types
    explanation: str = ""
    
    def get_evidence_summary(self) -> Dict[str, int]:
        """Get summary of evidence counts."""
        return {
            'supporting': len(self.supporting_evidence),
            'contradicting': len(self.contradicting_evidence),
            'neutral': len(self.neutral_evidence),
            'total': len(self.supporting_evidence) + len(self.contradicting_evidence) + len(self.neutral_evidence)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'claim': self.original_claim.text,
            'status': self.verification_status.value,
            'confidence': self.confidence_score,
            'consensus_score': self.consensus_score,
            'source_diversity': self.source_diversity,
            'evidence_summary': self.get_evidence_summary(),
            'explanation': self.explanation,
            'supporting_sources': [e.source_id for e in self.supporting_evidence],
            'contradicting_sources': [e.source_id for e in self.contradicting_evidence]
        }


class SourceCredibilityAssessor:
    """Assesses credibility of information sources."""
    
    def __init__(self):
        # Authority scores by source type (can be enhanced with specific source databases)
        self.source_type_authority = {
            SourceType.ACADEMIC_PAPER: 0.9,
            SourceType.GOVERNMENT_REPORT: 0.85,
            SourceType.ENCYCLOPEDIA: 0.8,
            SourceType.NEWS_ARTICLE: 0.6,
            SourceType.CORPORATE_REPORT: 0.55,
            SourceType.BLOG_POST: 0.3,
            SourceType.SOCIAL_MEDIA: 0.2,
            SourceType.UNKNOWN: 0.4
        }
        
        # Keywords that indicate high-authority sources
        self.authority_keywords = {
            'academic': ['journal', 'peer-reviewed', 'university', 'research', 'study', 'doi:', 'arxiv'],
            'government': ['gov', 'official', 'department', 'ministry', 'agency', 'bureau'],
            'news': ['reuters', 'associated press', 'bbc', 'cnn', 'wsj', 'nyt'],
            'high_quality': ['nature', 'science', 'cell', 'lancet', 'nejm', 'pnas']
        }
    
    def assess_source_credibility(
        self,
        source_id: str,
        source_metadata: Dict[str, Any],
        content_sample: str = ""
    ) -> SourceCredibility:
        """
        Assess credibility of a source.
        
        Args:
            source_id: Source identifier
            source_metadata: Metadata about the source
            content_sample: Sample content for analysis
            
        Returns:
            SourceCredibility assessment
        """
        # Determine source type
        source_type = self._determine_source_type(source_id, source_metadata, content_sample)
        
        # Calculate authority score
        authority_score = self._calculate_authority_score(source_id, source_type, source_metadata)
        
        # Calculate recency score
        recency_score = self._calculate_recency_score(source_metadata)
        
        # Calculate citation score
        citation_score = self._calculate_citation_score(content_sample, source_metadata)
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(content_sample)
        
        # Create credibility assessment
        credibility = SourceCredibility(
            source_id=source_id,
            source_type=source_type,
            authority_score=authority_score,
            recency_score=recency_score,
            citation_score=citation_score,
            consistency_score=consistency_score
        )
        
        credibility.calculate_overall_credibility()
        return credibility
    
    def _determine_source_type(self, source_id: str, metadata: Dict[str, Any], content: str) -> SourceType:
        """Determine the type of source."""
        source_lower = source_id.lower()
        
        # Check file extension and URL patterns
        if any(ext in source_lower for ext in ['.pdf', 'arxiv', 'doi:', 'journal']):
            return SourceType.ACADEMIC_PAPER
        elif any(domain in source_lower for domain in ['.gov', 'government', 'official']):
            return SourceType.GOVERNMENT_REPORT
        elif any(domain in source_lower for domain in ['news', 'times', 'post', 'bbc', 'cnn']):
            return SourceType.NEWS_ARTICLE
        elif 'blog' in source_lower or 'medium.com' in source_lower:
            return SourceType.BLOG_POST
        elif any(social in source_lower for social in ['twitter', 'facebook', 'linkedin', 'reddit']):
            return SourceType.SOCIAL_MEDIA
        elif 'wikipedia' in source_lower or 'encyclopedia' in source_lower:
            return SourceType.ENCYCLOPEDIA
        
        # Check content for academic indicators
        if content:
            academic_indicators = ['abstract', 'methodology', 'references', 'doi:', 'et al.', 'p<']
            if sum(indicator in content.lower() for indicator in academic_indicators) >= 2:
                return SourceType.ACADEMIC_PAPER
        
        return SourceType.UNKNOWN
    
    def _calculate_authority_score(self, source_id: str, source_type: SourceType, metadata: Dict[str, Any]) -> float:
        """Calculate authority score for the source."""
        base_score = self.source_type_authority[source_type]
        
        # Boost for high-quality sources
        source_lower = source_id.lower()
        for quality_type, keywords in self.authority_keywords.items():
            if any(keyword in source_lower for keyword in keywords):
                if quality_type == 'high_quality':
                    base_score = min(0.95, base_score + 0.15)
                elif quality_type == 'academic':
                    base_score = min(0.9, base_score + 0.1)
        
        return base_score
    
    def _calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate recency score based on publication date."""
        pub_date = metadata.get('publication_date')
        if not pub_date:
            return 0.5  # Neutral if no date
        
        try:
            if isinstance(pub_date, str):
                # Try to parse common date formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        pub_date = datetime.strptime(pub_date, fmt)
                        break
                    except ValueError:
                        continue
            
            if isinstance(pub_date, datetime):
                days_old = (datetime.now() - pub_date).days
                
                # Recent sources (< 1 year) get higher scores
                if days_old < 365:
                    return 1.0
                elif days_old < 365 * 3:  # 3 years
                    return 0.8
                elif days_old < 365 * 5:  # 5 years
                    return 0.6
                else:
                    return 0.4
                    
        except Exception:
            pass
        
        return 0.5
    
    def _calculate_citation_score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate citation/reference score."""
        if not content:
            return 0.5
        
        # Count references and citations
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2020)
            r'\[\d+\]',               # [1]
            r'et al\.',               # et al.
            r'doi:\s*\S+',           # DOI references
            r'https?://\S+',         # URLs
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, content))
        
        # Normalize to 0-1 scale
        if citation_count == 0:
            return 0.2
        elif citation_count < 5:
            return 0.4
        elif citation_count < 15:
            return 0.7
        else:
            return 0.9
    
    def _calculate_consistency_score(self, content: str) -> float:
        """Calculate internal consistency score."""
        if not content:
            return 0.5
        
        # Simple heuristics for consistency
        score = 0.5
        
        # Longer content might be more developed
        if len(content) > 1000:
            score += 0.1
        
        # Presence of structure indicators
        structure_indicators = ['introduction', 'conclusion', 'abstract', 'summary', 'methodology']
        if any(indicator in content.lower() for indicator in structure_indicators):
            score += 0.1
        
        # Check for contradictory language patterns (simple check)
        contradictory_patterns = [
            (r'\bhowever\b', r'\bbut\b'),
            (r'\balthough\b', r'\bnevertheless\b'),
        ]
        contradiction_count = 0
        for pattern1, pattern2 in contradictory_patterns:
            if re.search(pattern1, content, re.IGNORECASE) and re.search(pattern2, content, re.IGNORECASE):
                contradiction_count += 1
        
        # Too many contradictions reduce consistency
        if contradiction_count > 3:
            score -= 0.2
        
        return max(0.0, min(1.0, score))


class CrossSourceFactVerifier:
    """
    Main fact verification system that validates claims across multiple sources.
    """
    
    def __init__(self, embedding_service=None):
        """
        Initialize fact verifier.
        
        Args:
            embedding_service: Service for calculating semantic similarity
        """
        self.embedding_service = embedding_service
        self.claim_extractor = ClaimExtractor()
        self.credibility_assessor = SourceCredibilityAssessor()
        
        # Verification thresholds
        self.min_sources_for_verification = 2
        self.min_consensus_threshold = 0.6
        self.semantic_similarity_threshold = 0.7
        
        logger.info("CrossSourceFactVerifier initialized")
    
    def verify_claims(
        self,
        claims: List[Claim],
        source_documents: List[Dict[str, Any]],
        min_confidence: float = 0.5
    ) -> List[FactVerification]:
        """
        Verify a list of claims against source documents.
        
        Args:
            claims: Claims to verify
            source_documents: Available source documents
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of fact verification results
        """
        verifications = []
        
        # Assess source credibility for all documents
        source_credibilities = {}
        for doc in source_documents:
            source_id = doc.get('source', doc.get('chunk_id', ''))
            credibility = self.credibility_assessor.assess_source_credibility(
                source_id=source_id,
                source_metadata=doc.get('metadata', {}),
                content_sample=doc.get('text', '')[:1000]  # Sample first 1000 chars
            )
            source_credibilities[source_id] = credibility
        
        # Verify each claim
        for claim in claims:
            verification = self._verify_single_claim(
                claim, source_documents, source_credibilities
            )
            
            if verification.confidence_score >= min_confidence:
                verifications.append(verification)
        
        logger.info(f"Verified {len(verifications)} claims above confidence threshold")
        return verifications
    
    def _verify_single_claim(
        self,
        claim: Claim,
        source_documents: List[Dict[str, Any]],
        source_credibilities: Dict[str, SourceCredibility]
    ) -> FactVerification:
        """Verify a single claim against source documents."""
        
        # Find evidence for and against the claim
        supporting_evidence = []
        contradicting_evidence = []
        neutral_evidence = []
        
        for doc in source_documents:
            # Skip the source document of the original claim
            source_id = doc.get('source', doc.get('chunk_id', ''))
            if source_id == claim.source or source_id == claim.chunk_id:
                continue
            
            evidence = self._extract_evidence_from_document(
                claim, doc, source_credibilities.get(source_id)
            )
            
            if evidence:
                if evidence.is_supporting():
                    supporting_evidence.append(evidence)
                elif evidence.is_contradicting():
                    contradicting_evidence.append(evidence)
                else:
                    neutral_evidence.append(evidence)
        
        # Calculate verification metrics
        consensus_score = self._calculate_consensus_score(
            supporting_evidence, contradicting_evidence
        )
        
        source_diversity = self._calculate_source_diversity(
            supporting_evidence + contradicting_evidence + neutral_evidence
        )
        
        # Determine verification status
        status = self._determine_verification_status(
            supporting_evidence, contradicting_evidence, consensus_score
        )
        
        # Calculate overall confidence
        confidence = self._calculate_verification_confidence(
            status, consensus_score, source_diversity, supporting_evidence, contradicting_evidence
        )
        
        # Generate explanation
        explanation = self._generate_verification_explanation(
            status, len(supporting_evidence), len(contradicting_evidence), consensus_score
        )
        
        return FactVerification(
            original_claim=claim,
            verification_status=status,
            confidence_score=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            neutral_evidence=neutral_evidence,
            consensus_score=consensus_score,
            source_diversity=source_diversity,
            explanation=explanation
        )
    
    def _extract_evidence_from_document(
        self,
        claim: Claim,
        document: Dict[str, Any],
        source_credibility: Optional[SourceCredibility]
    ) -> Optional[Evidence]:
        """Extract evidence from a document for a claim."""
        
        doc_text = document.get('text', '')
        source_id = document.get('source', document.get('chunk_id', ''))
        
        if not doc_text:
            return None
        
        # Calculate semantic similarity
        similarity = self._calculate_semantic_similarity(claim.text, doc_text)
        
        if similarity < 0.3:  # Too dissimilar to be relevant
            return None
        
        # Extract relevant sentences from document
        relevant_sentences = self._find_relevant_sentences(claim.text, doc_text)
        
        if not relevant_sentences:
            return None
        
        # Determine support strength
        support_strength = self._calculate_support_strength(
            claim, relevant_sentences, similarity
        )
        
        # Calculate evidence confidence
        evidence_confidence = self._calculate_evidence_confidence(
            similarity, source_credibility, support_strength
        )
        
        evidence_text = ' '.join(relevant_sentences[:2])  # Top 2 relevant sentences
        
        return Evidence(
            text=evidence_text,
            source_id=source_id,
            chunk_id=document.get('chunk_id', ''),
            credibility=source_credibility or SourceCredibility(source_id, SourceType.UNKNOWN),
            support_strength=support_strength,
            confidence=evidence_confidence,
            semantic_similarity=similarity
        )
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts."""
        if not self.embedding_service:
            return self._lexical_similarity(text1, text2)
        
        try:
            emb1 = self.embedding_service.embed_text(text1)
            emb2 = self.embedding_service.embed_text(text2)
            
            vec1 = np.array(emb1.embedding)
            vec2 = np.array(emb2.embedding)
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return self._lexical_similarity(text1, text2)
    
    def _lexical_similarity(self, text1: str, text2: str) -> float:
        """Simple lexical similarity fallback."""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _find_relevant_sentences(self, claim_text: str, document_text: str) -> List[str]:
        """Find sentences in document that are relevant to the claim."""
        sentences = re.split(r'[.!?]+', document_text)
        relevant = []
        
        claim_words = set(re.findall(r'\b\w+\b', claim_text.lower()))
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(claim_words.intersection(sentence_words))
            
            if overlap >= 2:  # At least 2 words in common
                relevant.append(sentence)
        
        return relevant
    
    def _calculate_support_strength(
        self,
        claim: Claim,
        evidence_sentences: List[str],
        similarity: float
    ) -> float:
        """Calculate how strongly evidence supports or contradicts a claim."""
        
        evidence_text = ' '.join(evidence_sentences).lower()
        claim_lower = claim.text.lower()
        
        # Look for supporting patterns
        support_indicators = ['confirms', 'supports', 'validates', 'proves', 'demonstrates']
        contradict_indicators = ['contradicts', 'disproves', 'refutes', 'challenges', 'disputes']
        
        support_score = 0.0
        for indicator in support_indicators:
            if indicator in evidence_text:
                support_score += 0.2
        
        for indicator in contradict_indicators:
            if indicator in evidence_text:
                support_score -= 0.3
        
        # Check for negation patterns
        if claim.negated:
            # If claim is negated, look for opposing evidence
            negation_words = ['not', 'no', 'never', 'none', 'cannot', "can't", "won't"]
            evidence_negated = any(neg in evidence_text for neg in negation_words)
            
            if evidence_negated:
                support_score += similarity * 0.5
            else:
                support_score -= similarity * 0.3
        else:
            # If claim is positive, negated evidence contradicts
            negation_words = ['not', 'no', 'never', 'none', 'cannot', "can't", "won't"]
            evidence_negated = any(neg in evidence_text for neg in negation_words)
            
            if evidence_negated:
                support_score -= similarity * 0.4
            else:
                support_score += similarity * 0.3
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, support_score))
    
    def _calculate_evidence_confidence(
        self,
        similarity: float,
        source_credibility: Optional[SourceCredibility],
        support_strength: float
    ) -> float:
        """Calculate confidence in the evidence."""
        base_confidence = similarity * 0.4
        
        if source_credibility:
            base_confidence += source_credibility.overall_credibility * 0.4
        else:
            base_confidence += 0.2  # Default credibility
        
        # Stronger support/contradiction increases confidence
        base_confidence += abs(support_strength) * 0.2
        
        return max(0.0, min(1.0, base_confidence))
    
    def _calculate_consensus_score(
        self,
        supporting_evidence: List[Evidence],
        contradicting_evidence: List[Evidence]
    ) -> float:
        """Calculate consensus score across sources."""
        if not supporting_evidence and not contradicting_evidence:
            return 0.0
        
        # Weight evidence by credibility
        support_weight = sum(e.credibility.overall_credibility for e in supporting_evidence)
        contradict_weight = sum(e.credibility.overall_credibility for e in contradicting_evidence)
        
        total_weight = support_weight + contradict_weight
        if total_weight == 0:
            return 0.0
        
        # Consensus is the proportion of support vs contradiction
        consensus = (support_weight - contradict_weight) / total_weight
        return consensus  # Range: -1 to 1
    
    def _calculate_source_diversity(self, evidence_list: List[Evidence]) -> float:
        """Calculate diversity of source types."""
        if not evidence_list:
            return 0.0
        
        source_types = set(e.credibility.source_type for e in evidence_list)
        return len(source_types) / len(SourceType)
    
    def _determine_verification_status(
        self,
        supporting_evidence: List[Evidence],
        contradicting_evidence: List[Evidence],
        consensus_score: float
    ) -> VerificationStatus:
        """Determine verification status based on evidence."""
        
        total_evidence = len(supporting_evidence) + len(contradicting_evidence)
        
        if total_evidence < self.min_sources_for_verification:
            return VerificationStatus.INSUFFICIENT
        
        if consensus_score >= 0.6:
            return VerificationStatus.VERIFIED
        elif consensus_score >= 0.2:
            return VerificationStatus.PARTIALLY_VERIFIED
        elif consensus_score <= -0.4:
            return VerificationStatus.DISPUTED
        elif len(supporting_evidence) == 0:
            return VerificationStatus.UNVERIFIED
        else:
            return VerificationStatus.PARTIALLY_VERIFIED
    
    def _calculate_verification_confidence(
        self,
        status: VerificationStatus,
        consensus_score: float,
        source_diversity: float,
        supporting_evidence: List[Evidence],
        contradicting_evidence: List[Evidence]
    ) -> float:
        """Calculate overall verification confidence."""
        
        base_confidence = 0.5
        
        # Status-based confidence
        status_confidence = {
            VerificationStatus.VERIFIED: 0.9,
            VerificationStatus.PARTIALLY_VERIFIED: 0.6,
            VerificationStatus.DISPUTED: 0.7,  # High confidence in dispute
            VerificationStatus.UNVERIFIED: 0.4,
            VerificationStatus.INSUFFICIENT: 0.2
        }
        
        base_confidence = status_confidence.get(status, 0.5)
        
        # Adjust based on consensus strength
        base_confidence *= (0.5 + 0.5 * abs(consensus_score))
        
        # Adjust based on source diversity
        base_confidence *= (0.7 + 0.3 * source_diversity)
        
        # Adjust based on number of sources
        total_sources = len(supporting_evidence) + len(contradicting_evidence)
        if total_sources >= 5:
            base_confidence *= 1.1
        elif total_sources >= 3:
            base_confidence *= 1.05
        
        return max(0.0, min(1.0, base_confidence))
    
    def _generate_verification_explanation(
        self,
        status: VerificationStatus,
        support_count: int,
        contradict_count: int,
        consensus_score: float
    ) -> str:
        """Generate human-readable explanation of verification."""
        
        if status == VerificationStatus.VERIFIED:
            return f"Verified by {support_count} sources with strong consensus (score: {consensus_score:.2f})"
        elif status == VerificationStatus.PARTIALLY_VERIFIED:
            return f"Partially verified by {support_count} sources, {contradict_count} contradicting sources"
        elif status == VerificationStatus.DISPUTED:
            return f"Disputed claim with {support_count} supporting and {contradict_count} contradicting sources"
        elif status == VerificationStatus.UNVERIFIED:
            return f"No supporting evidence found across available sources"
        else:
            return f"Insufficient sources for verification ({support_count + contradict_count} sources found)"


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    verifier = CrossSourceFactVerifier()
    
    # Sample claims and documents
    claims = [
        Claim(
            text="Vector databases improve search performance by 50%",
            subject="Vector databases",
            predicate="improve",
            object="search performance by 50%",
            source="claim_source.pdf",
            chunk_id="claim_001"
        )
    ]
    
    documents = [
        {
            'text': 'Our benchmarks show that vector databases can improve search speed by up to 60% compared to traditional methods.',
            'source': 'benchmark_study.pdf',
            'chunk_id': 'chunk_001',
            'metadata': {'publication_date': '2024-01-15', 'type': 'academic'}
        },
        {
            'text': 'In practice, vector databases showed only marginal improvements of 10-15% in real-world applications.',
            'source': 'industry_report.pdf', 
            'chunk_id': 'chunk_002',
            'metadata': {'publication_date': '2024-02-01', 'type': 'corporate'}
        }
    ]
    
    verifications = verifier.verify_claims(claims, documents)
    
    for verification in verifications:
        print(f"\nClaim: {verification.original_claim.text}")
        print(f"Status: {verification.verification_status.value}")
        print(f"Confidence: {verification.confidence_score:.2f}")
        print(f"Consensus: {verification.consensus_score:.2f}")
        print(f"Evidence: {verification.get_evidence_summary()}")
        print(f"Explanation: {verification.explanation}")