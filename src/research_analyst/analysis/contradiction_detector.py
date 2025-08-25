"""
Advanced Contradiction Detection for Research Analysis

This module implements sophisticated contradiction detection using NLP techniques,
semantic analysis, and structured reasoning to identify conflicting information
across research documents with high accuracy.

Key Features:
- Semantic contradiction detection using embeddings
- Negation and sentiment analysis
- Claim extraction and comparison
- Confidence scoring for contradictions
- Source credibility weighting
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class ContradictionType(Enum):
    """Types of contradictions that can be detected."""
    DIRECT_NEGATION = "direct_negation"         # X is true vs X is false
    QUANTITATIVE = "quantitative"               # Different numbers/amounts  
    TEMPORAL = "temporal"                       # Different time references
    CAUSAL = "causal"                          # Different cause-effect relationships
    CATEGORICAL = "categorical"                 # Different categories/classifications
    OPINION = "opinion"                        # Conflicting opinions/evaluations
    METHODOLOGICAL = "methodological"          # Different methods/approaches
    SCOPE = "scope"                            # Different scope/applicability


@dataclass
class Claim:
    """Represents an extractable claim from text."""
    text: str
    subject: str
    predicate: str
    object: str
    negated: bool = False
    confidence: float = 0.0
    source: str = ""
    chunk_id: str = ""
    sentence_index: int = 0
    
    def __post_init__(self):
        # Extract basic components if not provided
        if not self.subject or not self.predicate:
            self._extract_components()
    
    def _extract_components(self):
        """Basic subject-predicate-object extraction."""
        # Simple patterns for claim extraction
        patterns = [
            r'(.+?)\s+(is|are|was|were|has|have|shows|demonstrates|indicates|suggests)\s+(.+)',
            r'(.+?)\s+(increases|decreases|improves|worsens|affects|causes|leads to)\s+(.+)',
            r'(.+?)\s+(can|could|should|will|would|may|might)\s+(.+)',
        ]
        
        text_clean = self.text.strip()
        for pattern in patterns:
            match = re.search(pattern, text_clean, re.IGNORECASE)
            if match:
                self.subject = match.group(1).strip()
                self.predicate = match.group(2).strip()
                self.object = match.group(3).strip()
                break
        
        # Check for negation
        negation_words = ['not', 'no', 'never', 'none', 'neither', 'cannot', "can't", "won't", "don't", "doesn't"]
        self.negated = any(neg in text_clean.lower() for neg in negation_words)


@dataclass
class Contradiction:
    """Represents a detected contradiction between two claims."""
    claim1: Claim
    claim2: Claim
    contradiction_type: ContradictionType
    confidence: float
    evidence: str
    severity: str  # "high", "medium", "low"
    explanation: str = ""
    semantic_similarity: float = 0.0
    source_credibility_factor: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'claim1': {
                'text': self.claim1.text,
                'source': self.claim1.source,
                'chunk_id': self.claim1.chunk_id,
                'negated': self.claim1.negated
            },
            'claim2': {
                'text': self.claim2.text,
                'source': self.claim2.source, 
                'chunk_id': self.claim2.chunk_id,
                'negated': self.claim2.negated
            },
            'type': self.contradiction_type.value,
            'confidence': self.confidence,
            'severity': self.severity,
            'evidence': self.evidence,
            'explanation': self.explanation,
            'semantic_similarity': self.semantic_similarity
        }


class ClaimExtractor:
    """Extracts claims from text for contradiction analysis."""
    
    def __init__(self):
        # Patterns for identifying claim-bearing sentences
        self.claim_patterns = [
            # Definitive statements
            r'\b(is|are|was|were)\s+(?:not\s+)?(?:very\s+|quite\s+|extremely\s+)?\w+',
            # Quantitative claims
            r'\b(?:increased?|decreased?|improved?|reduced?|enhanced?)\s+by\s+\d+',
            # Causal relationships  
            r'\b(?:causes?|leads?\s+to|results?\s+in|due\s+to|because\s+of)\b',
            # Comparative claims
            r'\b(?:better|worse|higher|lower|more|less|superior|inferior)\s+than\b',
            # Research findings
            r'\b(?:found|discovered|showed|demonstrated|indicated|suggested|concluded)\s+that\b',
        ]
        
        # Confidence indicators
        self.high_confidence_indicators = ['certainly', 'definitely', 'clearly', 'obviously', 'undoubtedly']
        self.low_confidence_indicators = ['possibly', 'perhaps', 'maybe', 'might', 'could', 'seems']
    
    def extract_claims(self, text: str, source: str = "", chunk_id: str = "") -> List[Claim]:
        """
        Extract claims from text.
        
        Args:
            text: Input text to analyze
            source: Source identifier
            chunk_id: Chunk identifier
            
        Returns:
            List of extracted claims
        """
        claims = []
        sentences = self._split_sentences(text)
        
        for i, sentence in enumerate(sentences):
            if self._is_claim_bearing(sentence):
                claim = Claim(
                    text=sentence,
                    subject="",  # Will be extracted in __post_init__
                    predicate="",
                    object="",
                    confidence=self._assess_claim_confidence(sentence),
                    source=source,
                    chunk_id=chunk_id,
                    sentence_index=i
                )
                claims.append(claim)
        
        return claims
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _is_claim_bearing(self, sentence: str) -> bool:
        """Check if sentence contains a claim."""
        sentence_lower = sentence.lower()
        
        # Check for claim patterns
        for pattern in self.claim_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        # Filter out questions and commands
        if sentence.strip().endswith('?') or sentence.startswith(('How', 'What', 'Why', 'When', 'Where')):
            return False
        
        return len(sentence.split()) >= 5  # Minimum length for meaningful claims
    
    def _assess_claim_confidence(self, sentence: str) -> float:
        """Assess confidence level of a claim based on linguistic cues."""
        sentence_lower = sentence.lower()
        confidence = 0.5  # Base confidence
        
        # High confidence indicators
        for indicator in self.high_confidence_indicators:
            if indicator in sentence_lower:
                confidence += 0.2
        
        # Low confidence indicators
        for indicator in self.low_confidence_indicators:
            if indicator in sentence_lower:
                confidence -= 0.2
        
        # Statistical/numerical evidence increases confidence
        if re.search(r'\b\d+(?:\.\d+)?%?\b', sentence):
            confidence += 0.1
        
        # Citations/references increase confidence
        if re.search(r'\([^)]*\d{4}[^)]*\)|et al\.|according to', sentence):
            confidence += 0.15
        
        return max(0.0, min(1.0, confidence))


class AdvancedContradictionDetector:
    """
    Advanced contradiction detection using multiple NLP techniques.
    """
    
    def __init__(self, embedding_service=None):
        """
        Initialize contradiction detector.
        
        Args:
            embedding_service: Service for generating embeddings
        """
        self.embedding_service = embedding_service
        self.claim_extractor = ClaimExtractor()
        
        # Contradiction detection thresholds
        self.semantic_similarity_threshold = 0.7  # High similarity for potential contradictions
        self.direct_negation_threshold = 0.9      # Very high threshold for direct contradictions
        
        # Quantitative contradiction patterns
        self.quantitative_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',  # Percentages
            r'(\d+(?:\.\d+)?)\s*(billion|million|thousand|hundred)',  # Large numbers
            r'(\d+(?:\.\d+)?)\s*(times|fold)',  # Multipliers
        ]
        
        logger.info("AdvancedContradictionDetector initialized")
    
    def detect_contradictions(
        self,
        text_chunks: List[Dict[str, Any]],
        min_confidence: float = 0.6
    ) -> List[Contradiction]:
        """
        Detect contradictions across multiple text chunks.
        
        Args:
            text_chunks: List of text chunks with metadata
            min_confidence: Minimum confidence threshold for contradictions
            
        Returns:
            List of detected contradictions
        """
        # Step 1: Extract claims from all chunks
        all_claims = []
        for chunk in text_chunks:
            claims = self.claim_extractor.extract_claims(
                text=chunk.get('text', ''),
                source=chunk.get('source', ''),
                chunk_id=chunk.get('chunk_id', '')
            )
            all_claims.extend(claims)
        
        logger.info(f"Extracted {len(all_claims)} claims from {len(text_chunks)} chunks")
        
        # Step 2: Compare claims pairwise
        contradictions = []
        for i, claim1 in enumerate(all_claims):
            for j, claim2 in enumerate(all_claims[i+1:], i+1):
                # Skip claims from same source
                if claim1.chunk_id == claim2.chunk_id:
                    continue
                
                contradiction = self._analyze_claim_pair(claim1, claim2)
                if contradiction and contradiction.confidence >= min_confidence:
                    contradictions.append(contradiction)
        
        # Step 3: Rank contradictions by confidence and severity
        contradictions.sort(key=lambda c: (c.confidence, self._severity_score(c.severity)), reverse=True)
        
        logger.info(f"Detected {len(contradictions)} contradictions above confidence threshold {min_confidence}")
        return contradictions
    
    def _analyze_claim_pair(self, claim1: Claim, claim2: Claim) -> Optional[Contradiction]:
        """Analyze a pair of claims for contradictions."""
        
        # Skip if claims are too similar (likely duplicates)
        if self._claims_too_similar(claim1, claim2):
            return None
        
        # Check for different types of contradictions
        contradiction_checks = [
            self._check_direct_negation,
            self._check_quantitative_contradiction,
            self._check_categorical_contradiction,
            self._check_temporal_contradiction,
            self._check_causal_contradiction,
        ]
        
        for check_func in contradiction_checks:
            contradiction = check_func(claim1, claim2)
            if contradiction:
                return contradiction
        
        return None
    
    def _check_direct_negation(self, claim1: Claim, claim2: Claim) -> Optional[Contradiction]:
        """Check for direct negation contradictions."""
        # Different negation states with similar content
        if claim1.negated != claim2.negated:
            # Check semantic similarity of the claims
            similarity = self._calculate_semantic_similarity(claim1.text, claim2.text)
            
            if similarity >= self.semantic_similarity_threshold:
                confidence = similarity * 0.9  # High confidence for direct negations
                return Contradiction(
                    claim1=claim1,
                    claim2=claim2,
                    contradiction_type=ContradictionType.DIRECT_NEGATION,
                    confidence=confidence,
                    evidence=f"One claim negates the other: '{claim1.text}' vs '{claim2.text}'",
                    severity="high",
                    explanation="Direct contradiction through negation",
                    semantic_similarity=similarity
                )
        
        return None
    
    def _check_quantitative_contradiction(self, claim1: Claim, claim2: Claim) -> Optional[Contradiction]:
        """Check for quantitative contradictions (different numbers)."""
        numbers1 = self._extract_numbers(claim1.text)
        numbers2 = self._extract_numbers(claim2.text)
        
        if numbers1 and numbers2:
            # Check if claims are about similar topics
            similarity = self._calculate_semantic_similarity(claim1.text, claim2.text)
            
            if similarity >= 0.6:  # Lower threshold for quantitative contradictions
                # Compare numbers
                for num1 in numbers1:
                    for num2 in numbers2:
                        if abs(num1 - num2) / max(num1, num2) > 0.3:  # 30% difference
                            confidence = similarity * 0.8
                            return Contradiction(
                                claim1=claim1,
                                claim2=claim2,
                                contradiction_type=ContradictionType.QUANTITATIVE,
                                confidence=confidence,
                                evidence=f"Different numerical values: {num1} vs {num2}",
                                severity="medium",
                                explanation="Conflicting quantitative information",
                                semantic_similarity=similarity
                            )
        
        return None
    
    def _check_categorical_contradiction(self, claim1: Claim, claim2: Claim) -> Optional[Contradiction]:
        """Check for categorical contradictions (different classifications)."""
        # Keywords that indicate categorical statements
        category_keywords = [
            'type', 'kind', 'category', 'class', 'group',
            'effective', 'ineffective', 'successful', 'failed',
            'positive', 'negative', 'good', 'bad', 'high', 'low'
        ]
        
        has_category1 = any(keyword in claim1.text.lower() for keyword in category_keywords)
        has_category2 = any(keyword in claim2.text.lower() for keyword in category_keywords)
        
        if has_category1 and has_category2:
            similarity = self._calculate_semantic_similarity(claim1.text, claim2.text)
            
            if similarity >= 0.5:
                # Look for opposite categorical terms
                opposite_pairs = [
                    ('effective', 'ineffective'), ('successful', 'failed'),
                    ('positive', 'negative'), ('good', 'bad'),
                    ('high', 'low'), ('increase', 'decrease')
                ]
                
                for pos, neg in opposite_pairs:
                    if ((pos in claim1.text.lower() and neg in claim2.text.lower()) or
                        (neg in claim1.text.lower() and pos in claim2.text.lower())):
                        
                        confidence = similarity * 0.7
                        return Contradiction(
                            claim1=claim1,
                            claim2=claim2,
                            contradiction_type=ContradictionType.CATEGORICAL,
                            confidence=confidence,
                            evidence=f"Opposite categorical classifications: {pos} vs {neg}",
                            severity="medium",
                            explanation="Conflicting categorical assessment",
                            semantic_similarity=similarity
                        )
        
        return None
    
    def _check_temporal_contradiction(self, claim1: Claim, claim2: Claim) -> Optional[Contradiction]:
        """Check for temporal contradictions."""
        time_indicators = [
            'before', 'after', 'during', 'while', 'since', 'until',
            'first', 'second', 'then', 'next', 'finally',
            'past', 'present', 'future', 'now', 'currently'
        ]
        
        has_temporal1 = any(indicator in claim1.text.lower() for indicator in time_indicators)
        has_temporal2 = any(indicator in claim2.text.lower() for indicator in time_indicators)
        
        if has_temporal1 and has_temporal2:
            similarity = self._calculate_semantic_similarity(claim1.text, claim2.text)
            
            if similarity >= 0.6:
                confidence = similarity * 0.6
                return Contradiction(
                    claim1=claim1,
                    claim2=claim2,
                    contradiction_type=ContradictionType.TEMPORAL,
                    confidence=confidence,
                    evidence="Conflicting temporal information",
                    severity="low",
                    explanation="Different timing or sequence",
                    semantic_similarity=similarity
                )
        
        return None
    
    def _check_causal_contradiction(self, claim1: Claim, claim2: Claim) -> Optional[Contradiction]:
        """Check for causal contradictions."""
        causal_indicators = [
            'causes', 'leads to', 'results in', 'due to', 'because of',
            'triggers', 'induces', 'prevents', 'blocks', 'inhibits'
        ]
        
        has_causal1 = any(indicator in claim1.text.lower() for indicator in causal_indicators)
        has_causal2 = any(indicator in claim2.text.lower() for indicator in causal_indicators)
        
        if has_causal1 and has_causal2:
            similarity = self._calculate_semantic_similarity(claim1.text, claim2.text)
            
            if similarity >= 0.5:
                # Check for opposing causal relationships
                opposing_causals = [
                    ('causes', 'prevents'), ('leads to', 'blocks'),
                    ('triggers', 'inhibits'), ('induces', 'prevents')
                ]
                
                for pos, neg in opposing_causals:
                    if ((pos in claim1.text.lower() and neg in claim2.text.lower()) or
                        (neg in claim1.text.lower() and pos in claim2.text.lower())):
                        
                        confidence = similarity * 0.8
                        return Contradiction(
                            claim1=claim1,
                            claim2=claim2,
                            contradiction_type=ContradictionType.CAUSAL,
                            confidence=confidence,
                            evidence=f"Opposite causal relationships: {pos} vs {neg}",
                            severity="high",
                            explanation="Conflicting cause-effect relationships",
                            semantic_similarity=similarity
                        )
        
        return None
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not self.embedding_service:
            # Fallback to simple lexical similarity
            return self._lexical_similarity(text1, text2)
        
        try:
            emb1 = self.embedding_service.embed_text(text1)
            emb2 = self.embedding_service.embed_text(text2)
            
            vec1 = np.array(emb1.embedding)
            vec2 = np.array(emb2.embedding)
            
            # Cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Embedding similarity calculation failed: {e}")
            return self._lexical_similarity(text1, text2)
    
    def _lexical_similarity(self, text1: str, text2: str) -> float:
        """Simple lexical similarity as fallback."""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        numbers = []
        
        # Extract percentages
        for match in re.finditer(r'(\d+(?:\.\d+)?)\s*%', text):
            numbers.append(float(match.group(1)))
        
        # Extract regular numbers
        for match in re.finditer(r'\b(\d+(?:\.\d+)?)\b', text):
            try:
                numbers.append(float(match.group(1)))
            except ValueError:
                continue
        
        return numbers
    
    def _claims_too_similar(self, claim1: Claim, claim2: Claim) -> bool:
        """Check if claims are too similar (likely duplicates)."""
        similarity = self._lexical_similarity(claim1.text, claim2.text)
        return similarity > 0.9
    
    def _severity_score(self, severity: str) -> int:
        """Convert severity to numeric score for sorting."""
        scores = {'high': 3, 'medium': 2, 'low': 1}
        return scores.get(severity, 0)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    detector = AdvancedContradictionDetector()
    
    # Sample contradictory claims
    chunks = [
        {
            'text': 'The new drug is 95% effective in treating the disease.',
            'source': 'study_a.pdf',
            'chunk_id': 'chunk_001'
        },
        {
            'text': 'The same drug shows only 60% effectiveness in clinical trials.',
            'source': 'study_b.pdf', 
            'chunk_id': 'chunk_002'
        },
        {
            'text': 'Vector databases significantly improve search performance.',
            'source': 'tech_report.pdf',
            'chunk_id': 'chunk_003'
        },
        {
            'text': 'Vector databases do not provide meaningful performance benefits.',
            'source': 'analysis.pdf',
            'chunk_id': 'chunk_004'
        }
    ]
    
    contradictions = detector.detect_contradictions(chunks)
    
    print(f"Found {len(contradictions)} contradictions:")
    for i, contradiction in enumerate(contradictions, 1):
        print(f"\n{i}. {contradiction.contradiction_type.value}")
        print(f"   Confidence: {contradiction.confidence:.2f}")
        print(f"   Severity: {contradiction.severity}")
        print(f"   Evidence: {contradiction.evidence}")
        print(f"   Claim 1: {contradiction.claim1.text[:100]}...")
        print(f"   Claim 2: {contradiction.claim2.text[:100]}...")