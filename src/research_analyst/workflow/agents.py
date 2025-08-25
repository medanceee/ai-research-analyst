"""
AI Agents for Advanced Analysis in Research Workflow

This module implements specialized AI agents that can be integrated with
the LangGraph workflow to provide more sophisticated analysis capabilities
using LLMs for summarization, critical review, and executive reporting.
"""

from typing import Dict, List, Any, Optional
import logging
from abc import ABC, abstractmethod

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)


class BaseAnalysisAgent(ABC):
    """Base class for analysis agents."""
    
    def __init__(
        self,
        agent_name: str,
        model_name: str = "deepseek-r1:1.5b",
        temperature: float = 0.1
    ):
        self.agent_name = agent_name
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM (using Ollama for local deployment)
        try:
            self.llm = Ollama(
                model=model_name,
                temperature=temperature
            )
            logger.info(f"Initialized {agent_name} with model {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM for {agent_name}: {e}")
            self.llm = None
    
    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis on the provided context."""
        pass
    
    def _invoke_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Safely invoke the LLM with retry logic."""
        if not self.llm:
            return f"[{self.agent_name}] LLM not available - using fallback analysis"
        
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                return response.strip()
            except Exception as e:
                logger.warning(f"LLM invocation failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return f"[{self.agent_name}] Analysis unavailable due to LLM error"
        
        return f"[{self.agent_name}] Analysis failed after {max_retries} attempts"


class ResearchAnalyst(BaseAnalysisAgent):
    """
    Research Analyst Agent - Specializes in extracting insights and themes
    from research documents with structured analysis.
    """
    
    def __init__(self, model_name: str = "deepseek-r1:1.5b"):
        super().__init__("ResearchAnalyst", model_name, temperature=0.2)
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze research context and extract structured insights.
        
        Args:
            context: Dictionary containing:
                - research_query: str
                - relevant_chunks: List[Dict]
                - document_summaries: Dict[str, str]
        
        Returns:
            Dictionary with extracted insights and themes
        """
        research_query = context.get("research_query", "")
        relevant_chunks = context.get("relevant_chunks", [])
        document_summaries = context.get("document_summaries", {})
        
        # Combine context for analysis
        context_text = self._prepare_context_text(relevant_chunks, document_summaries)
        
        # Create analysis prompt
        prompt = self._create_analysis_prompt(research_query, context_text)
        
        # Get LLM analysis
        analysis_text = self._invoke_llm(prompt)
        
        # Parse and structure the response
        structured_analysis = self._parse_analysis_response(analysis_text)
        
        return {
            "agent": self.agent_name,
            "analysis_type": "research_insights",
            "key_themes": structured_analysis.get("themes", []),
            "main_insights": structured_analysis.get("insights", []),
            "evidence_strength": structured_analysis.get("evidence_strength", "moderate"),
            "research_gaps": structured_analysis.get("gaps", []),
            "raw_analysis": analysis_text
        }
    
    def _prepare_context_text(self, chunks: List[Dict], summaries: Dict[str, str]) -> str:
        """Prepare context text from chunks and summaries."""
        context_parts = []
        
        # Add document summaries
        if summaries:
            context_parts.append("DOCUMENT SUMMARIES:")
            for source, summary in summaries.items():
                context_parts.append(f"Source: {source}")
                context_parts.append(f"Summary: {summary}")
                context_parts.append("")
        
        # Add relevant chunks
        if chunks:
            context_parts.append("RELEVANT EXCERPTS:")
            for i, chunk in enumerate(chunks[:10], 1):  # Limit to top 10
                context_parts.append(f"{i}. Source: {chunk.get('source', 'Unknown')}")
                context_parts.append(f"   Text: {chunk['text']}")
                context_parts.append(f"   Relevance: {chunk.get('similarity', 0):.2f}")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _create_analysis_prompt(self, research_query: str, context_text: str) -> str:
        """Create enhanced structured analysis prompt for the LLM."""
        return f"""You are a senior research analyst with expertise in critical thinking and evidence evaluation. Your task is to perform a comprehensive analysis of research materials to answer a specific query with high accuracy.

RESEARCH QUERY: {research_query}

AVAILABLE CONTEXT:
{context_text}

ANALYSIS FRAMEWORK:
Follow this systematic approach for maximum accuracy:

1. EVIDENCE EVALUATION:
   - Assess the quality and credibility of each source
   - Identify primary vs secondary sources
   - Note any methodological limitations
   - Flag potential biases or conflicts of interest

2. THEMATIC ANALYSIS:
   - Identify 3-5 dominant themes with supporting evidence
   - Note recurring patterns across sources
   - Highlight unique perspectives that deviate from consensus

3. KEY INSIGHTS:
   - Extract 5-7 actionable insights that directly address the query
   - Prioritize insights by strength of evidence
   - Include quantitative data where available
   - Note confidence level for each insight (High/Medium/Low)

4. EVIDENCE STRENGTH ASSESSMENT:
   - Rate overall evidence strength: STRONG/MODERATE/WEAK
   - Explain reasoning based on:
     * Source credibility and authority
     * Sample sizes and methodological rigor
     * Consistency across multiple sources
     * Recency and relevance of findings

5. CRITICAL GAPS & LIMITATIONS:
   - Identify information gaps that limit conclusions
   - Note areas where evidence is conflicting or insufficient
   - Suggest specific research directions for addressing gaps

6. SYNTHESIS & IMPLICATIONS:
   - Synthesize findings into coherent narrative
   - Draw logical connections between insights
   - Consider broader implications and applications

QUALITY STANDARDS:
- Be precise and avoid overgeneralization
- Distinguish between correlation and causation
- Acknowledge uncertainty where evidence is limited
- Use specific citations and page references when possible
- Maintain objectivity and highlight potential biases

ANALYSIS:"""
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured format."""
        sections = {
            "themes": [],
            "insights": [],
            "evidence_strength": "moderate",
            "gaps": []
        }
        
        try:
            # Simple parsing based on section headers
            current_section = None
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                if "THEMES:" in line.upper():
                    current_section = "themes"
                elif "INSIGHTS:" in line.upper():
                    current_section = "insights"
                elif "EVIDENCE_STRENGTH:" in line.upper():
                    current_section = "evidence_strength"
                elif "RESEARCH_GAPS:" in line.upper() or "GAPS:" in line.upper():
                    current_section = "gaps"
                elif current_section:
                    # Extract content
                    if current_section == "evidence_strength":
                        if "strong" in line.lower():
                            sections["evidence_strength"] = "strong"
                        elif "weak" in line.lower():
                            sections["evidence_strength"] = "weak"
                    else:
                        # Remove bullet points and numbering
                        clean_line = line.lstrip('•-*123456789. ')
                        if clean_line and len(clean_line) > 10:
                            sections[current_section].append(clean_line)
        
        except Exception as e:
            logger.warning(f"Failed to parse analysis response: {e}")
        
        return sections


class CriticalReviewer(BaseAnalysisAgent):
    """
    Critical Reviewer Agent - Specializes in identifying contradictions,
    biases, and limitations in research materials.
    """
    
    def __init__(self, model_name: str = "deepseek-r1:1.5b"):
        super().__init__("CriticalReviewer", model_name, temperature=0.3)
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform critical analysis to identify issues and contradictions.
        
        Args:
            context: Dictionary containing research context
        
        Returns:
            Dictionary with critical analysis findings
        """
        research_query = context.get("research_query", "")
        relevant_chunks = context.get("relevant_chunks", [])
        contradictions = context.get("contradictions", [])
        
        # Create critical review prompt
        context_text = self._prepare_critical_context(relevant_chunks, contradictions)
        prompt = self._create_critical_prompt(research_query, context_text)
        
        # Get critical analysis
        analysis_text = self._invoke_llm(prompt)
        
        # Parse response
        critical_analysis = self._parse_critical_response(analysis_text)
        
        return {
            "agent": self.agent_name,
            "analysis_type": "critical_review",
            "identified_contradictions": critical_analysis.get("contradictions", []),
            "potential_biases": critical_analysis.get("biases", []),
            "limitations": critical_analysis.get("limitations", []),
            "confidence_assessment": critical_analysis.get("confidence", "moderate"),
            "recommendations": critical_analysis.get("recommendations", []),
            "raw_analysis": analysis_text
        }
    
    def _prepare_critical_context(self, chunks: List[Dict], contradictions: List[Dict]) -> str:
        """Prepare context for critical analysis."""
        context_parts = []
        
        # Add research content
        if chunks:
            context_parts.append("RESEARCH CONTENT:")
            for i, chunk in enumerate(chunks[:8], 1):
                context_parts.append(f"{i}. Source: {chunk.get('source', 'Unknown')}")
                context_parts.append(f"   Content: {chunk['text'][:300]}...")
                context_parts.append("")
        
        # Add identified contradictions
        if contradictions:
            context_parts.append("IDENTIFIED CONTRADICTIONS:")
            for i, contradiction in enumerate(contradictions, 1):
                context_parts.append(f"{i}. Type: {contradiction.get('type', 'Unknown')}")
                context_parts.append(f"   Evidence A: {contradiction.get('chunk1', {}).get('text', '')[:200]}...")
                context_parts.append(f"   Evidence B: {contradiction.get('chunk2', {}).get('text', '')[:200]}...")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _create_critical_prompt(self, research_query: str, context_text: str) -> str:
        """Create enhanced prompt for critical analysis with systematic evaluation."""
        return f"""You are an expert critical reviewer with advanced training in research methodology, bias detection, and evidence evaluation. Your role is to rigorously assess research quality and identify potential issues that could compromise accuracy.

RESEARCH QUERY: {research_query}

MATERIALS TO REVIEW:
{context_text}

SYSTEMATIC CRITICAL EVALUATION FRAMEWORK:

1. SOURCE CREDIBILITY ASSESSMENT:
   - Evaluate author credentials and institutional affiliations
   - Assess publication venue quality (peer-reviewed vs non-peer-reviewed)
   - Check for conflicts of interest or funding sources
   - Verify citation patterns and scholarly impact

2. METHODOLOGICAL RIGOR:
   - Examine sample sizes and selection methods
   - Assess experimental design and controls
   - Evaluate statistical methods and significance testing
   - Check for reproducibility and replication attempts

3. BIAS DETECTION:
   - Selection bias: Are certain groups/cases overrepresented?
   - Confirmation bias: Cherry-picking supportive evidence?
   - Survivorship bias: Missing negative cases/failures?
   - Reporting bias: Selective presentation of results?
   - Temporal bias: Outdated information presented as current?

4. LOGICAL CONSISTENCY:
   - Check for internal contradictions within sources
   - Identify gaps in reasoning or logic jumps
   - Flag unsupported causal claims
   - Note correlation vs causation conflation

5. EVIDENCE QUALITY:
   - Distinguish between anecdotal and systematic evidence
   - Assess representativeness of samples/examples
   - Evaluate strength of causal vs correlational claims
   - Check for appropriate statistical power

6. CONTRADICTION ANALYSIS:
   - Identify direct contradictions between sources
   - Note subtle disagreements in conclusions
   - Assess whether contradictions are reconcilable
   - Evaluate which sources are more credible in conflicts

7. CONFIDENCE CALIBRATION:
   - High: Multiple high-quality sources, consistent findings, robust methods
   - Moderate: Some quality sources, minor inconsistencies, adequate methods
   - Low: Limited sources, significant contradictions, weak methods

CRITICAL EVALUATION CHECKLIST:
□ Are the conclusions supported by the evidence presented?
□ Are alternative explanations adequately considered?
□ Is the scope of claims appropriate to the evidence?
□ Are limitations and uncertainties clearly acknowledged?
□ Would independent reviewers likely reach similar conclusions?

CRITICAL ANALYSIS:"""
    
    def _parse_critical_response(self, response: str) -> Dict[str, Any]:
        """Parse critical analysis response."""
        sections = {
            "contradictions": [],
            "biases": [],
            "limitations": [],
            "confidence": "moderate",
            "recommendations": []
        }
        
        try:
            current_section = None
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Section headers
                if "CONTRADICTIONS:" in line.upper():
                    current_section = "contradictions"
                elif "BIASES:" in line.upper():
                    current_section = "biases"
                elif "LIMITATIONS:" in line.upper():
                    current_section = "limitations"
                elif "CONFIDENCE:" in line.upper():
                    current_section = "confidence"
                elif "RECOMMENDATIONS:" in line.upper():
                    current_section = "recommendations"
                elif current_section:
                    if current_section == "confidence":
                        if "high" in line.lower():
                            sections["confidence"] = "high"
                        elif "low" in line.lower():
                            sections["confidence"] = "low"
                    else:
                        clean_line = line.lstrip('•-*123456789. ')
                        if clean_line and len(clean_line) > 10:
                            sections[current_section].append(clean_line)
        
        except Exception as e:
            logger.warning(f"Failed to parse critical response: {e}")
        
        return sections


class ExecutiveSummarizer(BaseAnalysisAgent):
    """
    Executive Summarizer Agent - Creates executive-level summaries
    and recommendations for business decision makers.
    """
    
    def __init__(self, model_name: str = "deepseek-r1:1.5b"):
        super().__init__("ExecutiveSummarizer", model_name, temperature=0.1)
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create executive summary and strategic recommendations.
        
        Args:
            context: Complete analysis context
        
        Returns:
            Executive summary and recommendations
        """
        research_query = context.get("research_query", "")
        key_insights = context.get("key_insights", [])
        contradictions = context.get("contradictions", [])
        confidence_scores = context.get("confidence_scores", {})
        
        # Prepare executive context
        exec_context = self._prepare_executive_context(
            key_insights, contradictions, confidence_scores
        )
        
        # Create executive prompt
        prompt = self._create_executive_prompt(research_query, exec_context)
        
        # Generate executive summary
        summary_text = self._invoke_llm(prompt)
        
        # Parse executive response
        executive_analysis = self._parse_executive_response(summary_text)
        
        return {
            "agent": self.agent_name,
            "analysis_type": "executive_summary",
            "executive_summary": executive_analysis.get("summary", ""),
            "strategic_recommendations": executive_analysis.get("recommendations", []),
            "key_decisions": executive_analysis.get("decisions", []),
            "risk_assessment": executive_analysis.get("risks", ""),
            "next_steps": executive_analysis.get("next_steps", []),
            "raw_summary": summary_text
        }
    
    def _prepare_executive_context(
        self,
        insights: List[str],
        contradictions: List[Dict],
        confidence: Dict[str, float]
    ) -> str:
        """Prepare context for executive summary."""
        context_parts = []
        
        # Key insights
        if insights:
            context_parts.append("KEY INSIGHTS:")
            for i, insight in enumerate(insights[:5], 1):
                context_parts.append(f"{i}. {insight}")
            context_parts.append("")
        
        # Critical issues
        if contradictions:
            context_parts.append("CRITICAL ISSUES:")
            context_parts.append(f"- {len(contradictions)} contradictions identified")
            for contradiction in contradictions[:3]:
                context_parts.append(f"- {contradiction.get('type', 'Conflict')} detected")
            context_parts.append("")
        
        # Confidence assessment
        avg_confidence = sum(confidence.values()) / len(confidence) if confidence else 0.5
        context_parts.append(f"OVERALL CONFIDENCE: {avg_confidence:.2f}/1.0")
        context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _create_executive_prompt(self, research_query: str, context: str) -> str:
        """Create enhanced executive summary prompt with strategic focus."""
        return f"""You are a senior strategic consultant with extensive experience advising C-suite executives. Your task is to synthesize complex research into actionable strategic intelligence for decision-makers.

STRATEGIC QUESTION: {research_query}

RESEARCH INTELLIGENCE:
{context}

EXECUTIVE BRIEFING FRAMEWORK:
Structure your briefing using the following strategic analysis model:

1. EXECUTIVE SUMMARY (2-3 paragraphs):
   - Lead with the most critical finding that impacts business strategy
   - Quantify business impact where possible (ROI, market size, cost implications)
   - Present a clear "bottom line" recommendation
   - Address the "so what?" for the organization

2. STRATEGIC IMPLICATIONS:
   - Market positioning implications
   - Competitive advantage considerations
   - Resource allocation impacts
   - Timeline considerations for implementation

3. ACTIONABLE RECOMMENDATIONS:
   Provide 3-5 specific, prioritized recommendations:
   - IMMEDIATE (0-3 months): Quick wins and urgent actions
   - SHORT-TERM (3-12 months): Strategic initiatives  
   - LONG-TERM (1-3 years): Transformational opportunities
   
   For each recommendation include:
   * Expected impact/benefit
   * Resource requirements
   * Success metrics
   * Implementation difficulty (Low/Medium/High)

4. CRITICAL DECISIONS REQUIRED:
   - Investment decisions with financial implications
   - Strategic direction choices with market impact
   - Resource prioritization decisions
   - Partnership or acquisition considerations
   - Regulatory or compliance decisions

5. RISK ASSESSMENT & MITIGATION:
   - High-impact risks that could derail strategy
   - Market/competitive risks
   - Implementation/execution risks
   - Financial/resource risks
   - Mitigation strategies for each major risk

6. SUCCESS METRICS & MONITORING:
   - Key performance indicators to track
   - Milestones and checkpoints
   - Early warning indicators
   - Review and adjustment mechanisms

7. IMMEDIATE NEXT STEPS (Next 30 days):
   - Specific actions with ownership
   - Resource allocation decisions
   - Stakeholder communications needed
   - Additional analysis required

EXECUTIVE COMMUNICATION STANDARDS:
- Lead with conclusions, not methodology
- Use business language, minimize jargon
- Include specific numbers and timelines
- Address budget and resource implications
- Consider competitive/market context
- Acknowledge key uncertainties and how to address them

EXECUTIVE BRIEFING:"""
    
    def _parse_executive_response(self, response: str) -> Dict[str, Any]:
        """Parse executive summary response."""
        sections = {
            "summary": "",
            "recommendations": [],
            "decisions": [],
            "risks": "",
            "next_steps": []
        }
        
        try:
            current_section = None
            current_content = []
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Section headers
                if "EXECUTIVE_SUMMARY:" in line.upper():
                    current_section = "summary"
                    current_content = []
                elif "STRATEGIC_RECOMMENDATIONS:" in line.upper() or "RECOMMENDATIONS:" in line.upper():
                    if current_section == "summary":
                        sections["summary"] = " ".join(current_content)
                    current_section = "recommendations"
                    current_content = []
                elif "KEY_DECISIONS:" in line.upper():
                    current_section = "decisions"
                    current_content = []
                elif "RISK_ASSESSMENT:" in line.upper():
                    current_section = "risks"
                    current_content = []
                elif "NEXT_STEPS:" in line.upper():
                    if current_section == "risks":
                        sections["risks"] = " ".join(current_content)
                    current_section = "next_steps"
                    current_content = []
                elif line and current_section:
                    if current_section in ["recommendations", "decisions", "next_steps"]:
                        clean_line = line.lstrip('•-*123456789. ')
                        if clean_line and len(clean_line) > 5:
                            sections[current_section].append(clean_line)
                    else:
                        current_content.append(line)
            
            # Handle final section
            if current_section == "summary" and current_content:
                sections["summary"] = " ".join(current_content)
            elif current_section == "risks" and current_content:
                sections["risks"] = " ".join(current_content)
        
        except Exception as e:
            logger.warning(f"Failed to parse executive response: {e}")
        
        return sections


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Sample context for testing
    test_context = {
        "research_query": "What are the benefits of vector databases for AI applications?",
        "relevant_chunks": [
            {
                "text": "Vector databases provide efficient storage and retrieval of high-dimensional embeddings.",
                "source": "technical_paper.pdf",
                "similarity": 0.85
            },
            {
                "text": "ChromaDB offers persistent storage with built-in similarity search capabilities.",
                "source": "database_guide.md",
                "similarity": 0.78
            }
        ],
        "document_summaries": {
            "technical_paper.pdf": "Discusses vector database architecture and performance benefits.",
            "database_guide.md": "Provides practical guide to using ChromaDB for AI applications."
        }
    }
    
    # Test Research Analyst
    analyst = ResearchAnalyst()
    research_analysis = analyst.analyze(test_context)
    print("Research Analysis:", research_analysis)
    
    # Test Critical Reviewer
    reviewer = CriticalReviewer()
    critical_analysis = reviewer.analyze(test_context)
    print("Critical Analysis:", critical_analysis)
    
    # Test Executive Summarizer
    summarizer = ExecutiveSummarizer()
    executive_summary = summarizer.analyze({
        **test_context,
        "key_insights": research_analysis["main_insights"],
        "contradictions": [],
        "confidence_scores": {"overall": 0.8}
    })
    print("Executive Summary:", executive_summary)