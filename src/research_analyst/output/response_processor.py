"""
Response Post-Processor for LLM Output Quality Enhancement

This module cleans up and formats LLM responses to produce more professional,
consistent output for the research analyst system.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessedResponse:
    """Container for processed LLM response."""
    cleaned_text: str
    extracted_sections: Dict[str, str]
    key_insights: List[str]
    confidence_indicators: Dict[str, str]
    citations: List[str]
    formatting_applied: List[str]


class ResponseProcessor:
    """
    Advanced response processor that cleans up LLM output and enhances formatting.
    """
    
    def __init__(self):
        self.formatting_patterns = {
            # Remove thinking tags and internal reasoning
            'thinking_tags': r'<think>.*?</think>',
            'meta_comments': r'\[.*?thinking.*?\]',
            'internal_notes': r'\(.*?internal.*?\)',
            
            # Clean up section markers
            'section_markers': r'^#+\s*',
            'bullet_inconsistencies': r'^[•\-\*]+\s*',
            
            # Fix formatting issues
            'double_spaces': r'\s{2,}',
            'line_breaks': r'\n{3,}',
            'trailing_whitespace': r'\s+$',
        }
        
        self.professional_replacements = {
            # Make language more professional
            'i think': 'the analysis indicates',
            'i believe': 'the evidence suggests',
            'maybe': 'potentially',
            'probably': 'likely',
            'seems like': 'appears to be',
            'kind of': 'somewhat',
            'sort of': 'somewhat',
            
            # Strengthen weak language
            'might be': 'may be',
            'could be': 'may be',
            'should be': 'is likely to be',
        }
        
        self.section_headers = {
            'methodology': ['method', 'approach', 'technique'],
            'findings': ['result', 'finding', 'discovery', 'outcome'],
            'conclusions': ['conclusion', 'summary', 'implication'],
            'limitations': ['limitation', 'constraint', 'weakness'],
            'recommendations': ['recommendation', 'suggestion', 'next step']
        }
    
    def process_response(
        self,
        raw_response: str,
        response_type: str = "research_analysis"
    ) -> ProcessedResponse:
        """
        Process and enhance an LLM response for professional output.
        
        Args:
            raw_response: Raw LLM output text
            response_type: Type of response (research_analysis, executive_summary, etc.)
            
        Returns:
            ProcessedResponse with cleaned and enhanced text
        """
        logger.info(f"Processing {response_type} response ({len(raw_response)} chars)")
        
        # Apply sequential processing steps
        cleaned_text = self._remove_thinking_artifacts(raw_response)
        cleaned_text = self._enhance_professional_language(cleaned_text)
        cleaned_text = self._fix_formatting_issues(cleaned_text)
        cleaned_text = self._standardize_structure(cleaned_text, response_type)
        
        # Extract structured components
        sections = self._extract_sections(cleaned_text)
        insights = self._extract_insights(cleaned_text)
        confidence = self._analyze_confidence_indicators(cleaned_text)
        citations = self._extract_citations(cleaned_text)
        
        # Track applied formatting
        formatting_applied = [
            "thinking_artifacts_removed",
            "professional_language_enhanced", 
            "formatting_standardized",
            "structure_optimized"
        ]
        
        return ProcessedResponse(
            cleaned_text=cleaned_text,
            extracted_sections=sections,
            key_insights=insights,
            confidence_indicators=confidence,
            citations=citations,
            formatting_applied=formatting_applied
        )
    
    def _remove_thinking_artifacts(self, text: str) -> str:
        """Remove LLM thinking artifacts and internal reasoning."""
        # Remove <think> tags and their content
        text = re.sub(self.formatting_patterns['thinking_tags'], '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove meta-commentary
        text = re.sub(self.formatting_patterns['meta_comments'], '', text, flags=re.IGNORECASE)
        text = re.sub(self.formatting_patterns['internal_notes'], '', text, flags=re.IGNORECASE)
        
        # Remove phrases that indicate internal reasoning
        thinking_phrases = [
            r'let me think about this[^.]*\.',
            r'okay, so i need to[^.]*\.',
            r'looking at this[^,]*,',
            r'now, considering[^,]*,',
            r'hmm[^.]*\.',
            r'well[^,]*,'
        ]
        
        for pattern in thinking_phrases:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def _enhance_professional_language(self, text: str) -> str:
        """Enhance language to be more professional and authoritative."""
        # Apply professional replacements
        for casual, professional in self.professional_replacements.items():
            text = re.sub(rf'\b{re.escape(casual)}\b', professional, text, flags=re.IGNORECASE)
        
        # Strengthen hedging language appropriately
        text = re.sub(r'\bmight suggest\b', 'suggests', text, flags=re.IGNORECASE)
        text = re.sub(r'\bcould indicate\b', 'indicates', text, flags=re.IGNORECASE)
        text = re.sub(r'\bseems to show\b', 'demonstrates', text, flags=re.IGNORECASE)
        
        # Remove filler words
        filler_words = ['basically', 'actually', 'really', 'quite', 'rather', 'fairly']
        for word in filler_words:
            text = re.sub(rf'\b{word}\s+', '', text, flags=re.IGNORECASE)
        
        return text
    
    def _fix_formatting_issues(self, text: str) -> str:
        """Fix common formatting problems in LLM output."""
        # Fix spacing issues
        text = re.sub(self.formatting_patterns['double_spaces'], ' ', text)
        text = re.sub(self.formatting_patterns['line_breaks'], '\n\n', text)
        text = re.sub(self.formatting_patterns['trailing_whitespace'], '', text, flags=re.MULTILINE)
        
        # Standardize bullet points
        text = re.sub(r'^[\s]*[•\-\*]\s*', '- ', text, flags=re.MULTILINE)
        
        # Fix numbered lists
        text = re.sub(r'^[\s]*(\d+)[\.\)]\s*', r'\1. ', text, flags=re.MULTILINE)
        
        # Ensure proper capitalization after periods
        text = re.sub(r'(\. )([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        return text.strip()
    
    def _standardize_structure(self, text: str, response_type: str) -> str:
        """Standardize response structure based on type."""
        lines = text.split('\n')
        structured_lines = []
        
        current_section = None
        section_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if section_content:
                    structured_lines.append('')
                continue
            
            # Check if this line is a section header
            section_type = self._identify_section_type(line)
            if section_type:
                # Save previous section
                if current_section and section_content:
                    structured_lines.append(f"## {current_section.title()}")
                    structured_lines.extend(section_content)
                    structured_lines.append('')
                
                # Start new section
                current_section = section_type
                section_content = []
            else:
                section_content.append(line)
        
        # Add final section
        if current_section and section_content:
            structured_lines.append(f"## {current_section.title()}")
            structured_lines.extend(section_content)
        
        return '\n'.join(structured_lines)
    
    def _identify_section_type(self, line: str) -> Optional[str]:
        """Identify if a line represents a section header."""
        line_lower = line.lower().strip(':').strip()
        
        for section, keywords in self.section_headers.items():
            for keyword in keywords:
                if keyword in line_lower:
                    return section
        
        # Check for explicit section markers
        if any(marker in line_lower for marker in ['methodology:', 'findings:', 'conclusions:']):
            return line_lower.split(':')[0]
        
        return None
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract structured sections from the processed text."""
        sections = {}
        current_section = None
        current_content = []
        
        for line in text.split('\n'):
            if line.startswith('## '):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line[3:].strip().lower()
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save final section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _extract_insights(self, text: str) -> List[str]:
        """Extract key insights from the text."""
        insights = []
        
        # Look for insight indicators
        insight_patterns = [
            r'key insight[s]?:?\s*(.+?)(?:\n|$)',
            r'important finding[s]?:?\s*(.+?)(?:\n|$)',
            r'significant result[s]?:?\s*(.+?)(?:\n|$)',
            r'main takeaway[s]?:?\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in insight_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                insight = match.group(1).strip()
                if len(insight) > 20:  # Filter out very short insights
                    insights.append(insight)
        
        # Also look for numbered/bulleted insights
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if (line.startswith(('- ', '• ')) or re.match(r'^\d+\.', line)) and len(line) > 30:
                clean_line = re.sub(r'^[\d\.\-•\s]+', '', line).strip()
                if clean_line and len(clean_line) > 20:
                    insights.append(clean_line)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_insights = []
        for insight in insights:
            if insight not in seen:
                seen.add(insight)
                unique_insights.append(insight)
        
        return unique_insights[:10]  # Limit to top 10 insights
    
    def _analyze_confidence_indicators(self, text: str) -> Dict[str, str]:
        """Analyze confidence indicators in the text."""
        confidence_indicators = {
            'overall': 'moderate',
            'evidence_strength': 'moderate',
            'certainty_level': 'moderate'
        }
        
        # High confidence indicators
        high_conf_patterns = [
            r'strong evidence', r'clearly demonstrates', r'definitively shows',
            r'robust findings', r'consistent across', r'well-established'
        ]
        
        # Low confidence indicators  
        low_conf_patterns = [
            r'limited evidence', r'preliminary findings', r'unclear',
            r'insufficient data', r'uncertain', r'may suggest'
        ]
        
        text_lower = text.lower()
        
        high_count = sum(1 for pattern in high_conf_patterns if re.search(pattern, text_lower))
        low_count = sum(1 for pattern in low_conf_patterns if re.search(pattern, text_lower))
        
        if high_count > low_count:
            confidence_indicators['overall'] = 'high'
            confidence_indicators['evidence_strength'] = 'strong'
        elif low_count > high_count:
            confidence_indicators['overall'] = 'low'
            confidence_indicators['evidence_strength'] = 'weak'
        
        return confidence_indicators
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation information from the text."""
        citations = []
        
        # Look for citation patterns
        citation_patterns = [
            r'\[([^\]]+\.pdf[^\]]*)\]',
            r'\[([^\]]+, p\. \d+[^\]]*)\]',
            r'Source: ([^,\n]+)',
            r'according to ([^,\n]+)',
            r'based on ([^,\n]+)'
        ]
        
        for pattern in citation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citation = match.group(1).strip()
                if citation and len(citation) > 3:
                    citations.append(citation)
        
        return list(set(citations))  # Remove duplicates


def create_professional_summary(
    processed_response: ProcessedResponse,
    research_query: str,
    metadata: Dict[str, Any]
) -> str:
    """
    Create a professional summary from processed response.
    
    Args:
        processed_response: Processed LLM response
        research_query: Original research question
        metadata: Additional metadata (processing time, docs count, etc.)
        
    Returns:
        Formatted professional summary
    """
    summary_parts = [
        f"# Research Analysis Report\n",
        f"## Query\n**{research_query}**\n",
        f"## Executive Summary"
    ]
    
    # Add key insights if available
    if processed_response.key_insights:
        summary_parts.append("### Key Findings")
        for i, insight in enumerate(processed_response.key_insights[:5], 1):
            summary_parts.append(f"{i}. {insight}")
        summary_parts.append("")
    
    # Add main content
    if processed_response.extracted_sections:
        for section_name, content in processed_response.extracted_sections.items():
            if content.strip():
                summary_parts.append(f"## {section_name.title()}")
                summary_parts.append(content)
                summary_parts.append("")
    else:
        # Fall back to cleaned text
        summary_parts.append("## Analysis")
        summary_parts.append(processed_response.cleaned_text)
        summary_parts.append("")
    
    # Add citations if available
    if processed_response.citations:
        summary_parts.append("## Sources Referenced")
        for i, citation in enumerate(processed_response.citations, 1):
            summary_parts.append(f"{i}. {citation}")
        summary_parts.append("")
    
    # Add metadata
    summary_parts.extend([
        "## Analysis Metrics",
        f"- **Processing Time**: {metadata.get('processing_time', 0):.1f}s",
        f"- **Documents Analyzed**: {metadata.get('document_count', 0)}",
        f"- **Text Chunks**: {metadata.get('chunk_count', 0)}",
        f"- **Confidence Level**: {processed_response.confidence_indicators.get('overall', 'moderate').title()}",
        "",
        "---",
        "*Generated by AI Research Analyst with Enhanced Processing*"
    ])
    
    return "\n".join(summary_parts)


if __name__ == "__main__":
    # Test the response processor
    sample_raw_response = """
    <think>
    Let me analyze this research paper. I need to figure out what the key methodologies are.
    </think>
    
    **Final Answer:**
    
    I think the main findings are quite interesting. The paper basically shows that:
    
    1. The methodology seems to use some kind of reinforcement learning approach
    2. They probably found some significant results 
    3. Maybe there are some limitations but it's hard to say
    
    Actually, looking at this more carefully, the evidence suggests stronger conclusions.
    """
    
    processor = ResponseProcessor()
    result = processor.process_response(sample_raw_response, "research_analysis")
    
    print("Original:", repr(sample_raw_response))
    print("\nCleaned:", result.cleaned_text)
    print("\nInsights:", result.key_insights)
    print("\nConfidence:", result.confidence_indicators)