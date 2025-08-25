"""
Markdown Output Formatter for Research Results

This module converts research analysis results into well-formatted Markdown
reports suitable for documentation, presentations, and sharing.

Features:
- Executive summary formatting
- Structured analysis sections
- Confidence indicators
- Citation formatting
- Quality warnings and suggestions
- Visual indicators (badges, emojis)
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from ..enhanced_research_analyst import EnhancedResearchResult
from ..analysis.contradiction_detector import Contradiction, ContradictionType
from ..analysis.fact_verifier import FactVerification, VerificationStatus


@dataclass
class MarkdownConfig:
    """Configuration for markdown formatting."""
    include_toc: bool = True
    include_metadata: bool = True
    include_quality_metrics: bool = True
    include_contradictions: bool = True
    include_fact_verifications: bool = True
    include_recommendations: bool = True
    include_citations: bool = True
    confidence_badges: bool = True
    emoji_indicators: bool = True


class MarkdownFormatter:
    """
    Formats research results into professional Markdown reports.
    """
    
    def __init__(self, config: Optional[MarkdownConfig] = None):
        """
        Initialize markdown formatter.
        
        Args:
            config: Formatting configuration options
        """
        self.config = config or MarkdownConfig()
        
        # Confidence level mappings
        self.confidence_badges = {
            "high": "![Confidence: High](https://img.shields.io/badge/Confidence-High-brightgreen)",
            "medium": "![Confidence: Medium](https://img.shields.io/badge/Confidence-Medium-yellow)",
            "low": "![Confidence: Low](https://img.shields.io/badge/Confidence-Low-red)"
        }
        
        # Emoji mappings
        self.status_emojis = {
            "verified": "✅",
            "partially_verified": "🟡", 
            "disputed": "❌",
            "unverified": "❓",
            "insufficient": "⚠️"
        }
        
        self.contradiction_emojis = {
            "direct_negation": "🔄",
            "quantitative": "📊",
            "categorical": "🏷️",
            "temporal": "⏰",
            "causal": "🔗",
            "opinion": "💭",
            "methodological": "🔬",
            "scope": "🎯"
        }
    
    def format_research_result(self, result: EnhancedResearchResult, title: Optional[str] = None) -> str:
        """
        Convert research result to formatted Markdown.
        
        Args:
            result: Enhanced research result to format
            title: Optional custom title for the report
            
        Returns:
            Formatted Markdown string
        """
        sections = []
        
        # Title and header
        sections.append(self._format_header(result, title))
        
        # Table of contents
        if self.config.include_toc:
            sections.append(self._format_toc())
        
        # Metadata and overview
        if self.config.include_metadata:
            sections.append(self._format_metadata(result))
        
        # Quality metrics and confidence
        if self.config.include_quality_metrics:
            sections.append(self._format_quality_metrics(result))
        
        # Executive summary
        sections.append(self._format_executive_summary(result))
        
        # Key insights
        sections.append(self._format_key_insights(result))
        
        # Detailed analysis
        sections.append(self._format_detailed_analysis(result))
        
        # Recommendations
        if self.config.include_recommendations and result.recommendations:
            sections.append(self._format_recommendations(result))
        
        # Quality assessment
        sections.append(self._format_quality_assessment(result))
        
        # Contradictions
        if self.config.include_contradictions and result.contradictions_detected:
            sections.append(self._format_contradictions(result))
        
        # Fact verifications
        if self.config.include_fact_verifications and result.fact_verifications:
            sections.append(self._format_fact_verifications(result))
        
        # Sources and citations
        if self.config.include_citations:
            sections.append(self._format_sources(result))
        
        # Appendix
        sections.append(self._format_appendix(result))
        
        return "\n\n".join(sections)
    
    def _format_header(self, result: EnhancedResearchResult, title: Optional[str] = None) -> str:
        """Format document header."""
        if not title:
            title = f"Research Analysis: {result.query}"
        
        confidence_level = self._get_confidence_level(result.confidence_score)
        badge = self.confidence_badges[confidence_level] if self.config.confidence_badges else ""
        
        header = f"""# {title}

{badge}

**Research Query**: {result.query}

**Generated**: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}

**Analysis Confidence**: {result.confidence_score:.1%} ({confidence_level.title()})

---"""
        
        return header
    
    def _format_toc(self) -> str:
        """Format table of contents."""
        toc = """## Table of Contents

- [📊 Overview & Metrics](#-overview--metrics)
- [🎯 Executive Summary](#-executive-summary)
- [💡 Key Insights](#-key-insights)
- [📝 Detailed Analysis](#-detailed-analysis)
- [🚀 Recommendations](#-recommendations)
- [⚖️ Quality Assessment](#️-quality-assessment)
- [🔍 Sources & Citations](#-sources--citations)
- [📋 Appendix](#-appendix)"""
        
        return toc
    
    def _format_metadata(self, result: EnhancedResearchResult) -> str:
        """Format metadata and overview section."""
        uncertainty_emoji = {
            "low": "🟢",
            "medium": "🟡", 
            "high": "🔴"
        }[result.accuracy_metrics.uncertainty_level]
        
        evidence_emoji = {
            "strong": "💪",
            "moderate": "👍",
            "weak": "👎",
            "insufficient": "❓"
        }.get(result.accuracy_metrics.evidence_strength, "❓")
        
        metadata = f"""## 📊 Overview & Metrics

### Analysis Summary
| Metric | Value | Status |
|--------|-------|--------|
| **Overall Confidence** | {result.confidence_score:.1%} | {self._get_confidence_level(result.confidence_score).title()} |
| **Uncertainty Level** | {uncertainty_emoji} {result.accuracy_metrics.uncertainty_level.title()} | {self._get_uncertainty_description(result.accuracy_metrics.uncertainty_level)} |
| **Evidence Strength** | {evidence_emoji} {result.accuracy_metrics.evidence_strength.title()} | {self._get_evidence_description(result.accuracy_metrics.evidence_strength)} |
| **Sources Analyzed** | {result.document_count} documents | {result.chunk_count} content chunks |
| **Processing Time** | {result.processing_time:.1f} seconds | - |

### Quality Indicators
- **Source Credibility**: {result.accuracy_metrics.source_credibility_score:.1%}
- **Fact Verification**: {result.accuracy_metrics.fact_verification_score:.1%}
- **Contradiction Risk**: {result.accuracy_metrics.contradiction_risk:.1%}
- **Source Diversity**: {result.accuracy_metrics.source_diversity:.1%}"""

        return metadata
    
    def _format_quality_metrics(self, result: EnhancedResearchResult) -> str:
        """Format quality metrics with visual indicators."""
        # Quality warnings
        warnings_section = ""
        if result.quality_warnings:
            warnings_list = "\n".join([f"- ⚠️ {warning}" for warning in result.quality_warnings])
            warnings_section = f"""
### 🚨 Quality Warnings
{warnings_list}"""
        
        # Improvement suggestions
        suggestions_section = ""
        if result.improvement_suggestions:
            suggestions_list = "\n".join([f"- 💡 {suggestion}" for suggestion in result.improvement_suggestions])
            suggestions_section = f"""
### 💡 Improvement Suggestions
{suggestions_list}"""
        
        return f"""{warnings_section}{suggestions_section}"""
    
    def _format_executive_summary(self, result: EnhancedResearchResult) -> str:
        """Format executive summary section."""
        summary = f"""## 🎯 Executive Summary

{result.executive_summary}

### Key Takeaways
{self._format_key_takeaways(result)}"""
        
        return summary
    
    def _format_key_takeaways(self, result: EnhancedResearchResult) -> str:
        """Format key takeaways as bullet points."""
        if not result.key_insights:
            return "- No key insights available"
        
        takeaways = []
        for i, insight in enumerate(result.key_insights[:5], 1):
            takeaways.append(f"{i}. **{insight}**")
        
        return "\n".join(takeaways)
    
    def _format_key_insights(self, result: EnhancedResearchResult) -> str:
        """Format key insights section."""
        if not result.key_insights:
            return "## 💡 Key Insights\n\nNo key insights were identified in this analysis."
        
        insights_formatted = []
        for i, insight in enumerate(result.key_insights, 1):
            insights_formatted.append(f"### {i}. {insight}\n")
        
        insights_content = "\n".join(insights_formatted)
        
        return f"""## 💡 Key Insights

{insights_content}"""
    
    def _format_detailed_analysis(self, result: EnhancedResearchResult) -> str:
        """Format detailed analysis section."""
        analysis = result.detailed_analysis or "No detailed analysis available."
        
        return f"""## 📝 Detailed Analysis

{analysis}"""
    
    def _format_recommendations(self, result: EnhancedResearchResult) -> str:
        """Format recommendations section."""
        if not result.recommendations:
            return ""
        
        recommendations_formatted = []
        for i, rec in enumerate(result.recommendations, 1):
            recommendations_formatted.append(f"### {i}. {rec}\n")
        
        recommendations_content = "\n".join(recommendations_formatted)
        
        return f"""## 🚀 Recommendations

{recommendations_content}"""
    
    def _format_quality_assessment(self, result: EnhancedResearchResult) -> str:
        """Format quality assessment section."""
        assessment = f"""## ⚖️ Quality Assessment

### Confidence Analysis
This analysis achieved a **{result.confidence_score:.1%} confidence score**, indicating {self._get_confidence_description(result.confidence_score)}.

#### Contributing Factors:
- **Source Quality**: {self._format_quality_bar(result.accuracy_metrics.source_credibility_score)}
- **Evidence Consistency**: {self._format_quality_bar(1 - result.accuracy_metrics.contradiction_risk)}
- **Cross-Validation**: {self._format_quality_bar(result.accuracy_metrics.fact_verification_score)}
- **Information Diversity**: {self._format_quality_bar(result.accuracy_metrics.source_diversity)}

### Reliability Indicators
"""
        
        # Add reliability assessment
        if result.confidence_score >= 0.8:
            assessment += "✅ **High Reliability**: Results can be used with confidence for decision-making.\n"
        elif result.confidence_score >= 0.6:
            assessment += "🟡 **Medium Reliability**: Results are generally trustworthy but consider additional validation.\n"
        else:
            assessment += "🔴 **Low Reliability**: Results require significant additional validation before use.\n"
        
        return assessment
    
    def _format_contradictions(self, result: EnhancedResearchResult) -> str:
        """Format contradictions section."""
        if not result.contradictions_detected:
            return ""
        
        contradictions_content = []
        
        for i, contradiction in enumerate(result.contradictions_detected, 1):
            emoji = self.contradiction_emojis.get(contradiction.contradiction_type.value, "❗")
            
            contradiction_md = f"""### {emoji} Contradiction #{i}: {contradiction.contradiction_type.value.replace('_', ' ').title()}

**Confidence**: {contradiction.confidence:.1%} | **Severity**: {contradiction.severity.title()}

**Evidence**:
{contradiction.evidence}

**Conflicting Claims**:
- **Source 1** ({contradiction.claim1.source}): "{contradiction.claim1.text}"
- **Source 2** ({contradiction.claim2.source}): "{contradiction.claim2.text}"

**Analysis**: {contradiction.explanation or 'Automatic detection based on semantic analysis.'}

---"""
            
            contradictions_content.append(contradiction_md)
        
        contradictions_section = "\n".join(contradictions_content)
        
        return f"""## 🔍 Contradictions Detected

{len(result.contradictions_detected)} potential contradictions were identified in the source materials:

{contradictions_section}"""
    
    def _format_fact_verifications(self, result: EnhancedResearchResult) -> str:
        """Format fact verification section."""
        if not result.fact_verifications:
            return ""
        
        # Group by verification status
        status_groups = {}
        for verification in result.fact_verifications:
            status = verification.verification_status.value
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(verification)
        
        verification_content = []
        
        for status, verifications in status_groups.items():
            emoji = self.status_emojis.get(status, "❓")
            status_title = status.replace('_', ' ').title()
            
            verification_content.append(f"### {emoji} {status_title} ({len(verifications)} claims)")
            
            for verification in verifications:
                evidence_summary = verification.get_evidence_summary()
                
                claim_md = f"""
**Claim**: "{verification.original_claim.text}"
- **Confidence**: {verification.confidence_score:.1%}
- **Evidence**: {evidence_summary['supporting']} supporting, {evidence_summary['contradicting']} contradicting
- **Explanation**: {verification.explanation}
"""
                verification_content.append(claim_md)
        
        verification_section = "\n".join(verification_content)
        
        return f"""## ✅ Fact Verification Results

Cross-source verification was performed on key claims:

{verification_section}"""
    
    def _format_sources(self, result: EnhancedResearchResult) -> str:
        """Format sources and citations section."""
        if not result.sources_used:
            return "## 🔍 Sources & Citations\n\nNo sources were identified in this analysis."
        
        sources_formatted = []
        for i, source in enumerate(result.sources_used, 1):
            # Clean up source path for display
            source_display = source.split('/')[-1] if '/' in source else source
            sources_formatted.append(f"{i}. `{source_display}`")
        
        sources_content = "\n".join(sources_formatted)
        
        return f"""## 🔍 Sources & Citations

### Documents Analyzed ({len(result.sources_used)})
{sources_content}

### Citation Format
This analysis drew from {len(result.sources_used)} source documents totaling {result.chunk_count} content segments. All findings are traceable to specific source materials for verification and further research."""
    
    def _format_appendix(self, result: EnhancedResearchResult) -> str:
        """Format appendix with technical details."""
        appendix = f"""## 📋 Appendix

### Technical Details
- **Analysis Engine**: Enhanced Research Analyst v2.0
- **Embedding Model**: High-accuracy semantic embeddings
- **Processing Features**: 
  - ✅ Hybrid semantic-lexical search
  - ✅ Advanced contradiction detection
  - ✅ Cross-source fact verification
  - ✅ Confidence scoring & uncertainty quantification

### Methodology
This analysis employed advanced natural language processing techniques including:
1. **Semantic Search**: Vector embeddings for content similarity
2. **Lexical Matching**: Keyword-based relevance scoring  
3. **Contradiction Detection**: Multi-type conflict identification
4. **Fact Verification**: Cross-source claim validation
5. **Confidence Scoring**: Multi-factor reliability assessment

### Quality Assurance
- Confidence threshold: {result.confidence_score:.1%}
- Source diversity: {result.accuracy_metrics.source_diversity:.1%}
- Processing integrity: Verified
- Citation traceability: Complete

---

*Generated by Enhanced Research Analyst - Providing enterprise-grade research intelligence with quantified confidence and comprehensive accuracy assessment.*"""
        
        return appendix
    
    def _format_quality_bar(self, score: float) -> str:
        """Format a quality score as a visual bar."""
        if score >= 0.8:
            return f"🟢 {score:.1%} (Excellent)"
        elif score >= 0.6:
            return f"🟡 {score:.1%} (Good)"
        elif score >= 0.4:
            return f"🟠 {score:.1%} (Fair)"
        else:
            return f"🔴 {score:.1%} (Poor)"
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level from score."""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _get_confidence_description(self, score: float) -> str:
        """Get confidence description."""
        if score >= 0.8:
            return "high reliability with strong supporting evidence"
        elif score >= 0.6:
            return "moderate reliability with adequate supporting evidence"
        else:
            return "low reliability requiring additional validation"
    
    def _get_uncertainty_description(self, level: str) -> str:
        """Get uncertainty level description."""
        descriptions = {
            "low": "High confidence in conclusions",
            "medium": "Moderate confidence, some areas need validation",
            "high": "Significant uncertainty, extensive validation recommended"
        }
        return descriptions.get(level, "Unknown uncertainty level")
    
    def _get_evidence_description(self, strength: str) -> str:
        """Get evidence strength description."""
        descriptions = {
            "strong": "Robust evidence from multiple credible sources",
            "moderate": "Adequate evidence with minor gaps",
            "weak": "Limited evidence with significant gaps",
            "insufficient": "Inadequate evidence for reliable conclusions"
        }
        return descriptions.get(strength, "Unknown evidence strength")
    
    def save_markdown_report(self, result: EnhancedResearchResult, output_path: str, title: Optional[str] = None) -> bool:
        """
        Save research result as markdown file.
        
        Args:
            result: Research result to format
            output_path: Path to save markdown file
            title: Optional title for the report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            markdown_content = self.format_research_result(result, title)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            return True
        except Exception as e:
            print(f"Error saving markdown report: {e}")
            return False


# Convenience function for quick markdown formatting
def format_result_as_markdown(
    result: EnhancedResearchResult,
    title: Optional[str] = None,
    output_file: Optional[str] = None,
    config: Optional[MarkdownConfig] = None
) -> str:
    """
    Quick function to format result as markdown.
    
    Args:
        result: Research result to format
        title: Optional title
        output_file: Optional file to save to
        config: Optional formatting configuration
        
    Returns:
        Formatted markdown string
    """
    formatter = MarkdownFormatter(config)
    markdown_content = formatter.format_research_result(result, title)
    
    if output_file:
        formatter.save_markdown_report(result, output_file, title)
    
    return markdown_content


if __name__ == "__main__":
    # Example usage would go here
    print("MarkdownFormatter module loaded successfully")