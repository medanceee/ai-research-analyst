"""
LangGraph Workflow Nodes for Research Analysis Pipeline

This module implements the individual nodes that make up the research workflow,
each handling a specific step in the analysis process.
"""

from typing import Dict, List, Any, Optional
import logging
import time
from abc import ABC, abstractmethod

from langchain_core.messages import AIMessage

from .workflow import WorkflowState
from ..rag.rag_pipeline import RAGPipeline
from ..rag.retrieval import RetrievalContext

logger = logging.getLogger(__name__)


class BaseWorkflowNode(ABC):
    """Base class for workflow nodes."""
    
    def __init__(self, node_name: str):
        self.node_name = node_name
    
    @abstractmethod
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute the node's functionality."""
        pass
    
    def _log_step_start(self, state: WorkflowState) -> float:
        """Log step start and return start time."""
        logger.info(f"Starting {self.node_name} step")
        state["current_step"] = self.node_name
        return time.time()
    
    def _log_step_end(self, state: WorkflowState, start_time: float, message: str = ""):
        """Log step completion and record timing."""
        duration = time.time() - start_time
        state["step_timings"][self.node_name] = duration
        
        log_message = f"Completed {self.node_name} step in {duration:.2f}s"
        if message:
            log_message += f" - {message}"
        logger.info(log_message)
    
    def _add_message(self, state: WorkflowState, content: str):
        """Add an AI message to the state."""
        state["messages"].append(AIMessage(content=f"[{self.node_name}] {content}"))


class IngestionNode(BaseWorkflowNode):
    """
    Document Ingestion Node - Processes and stores documents in the RAG system.
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        super().__init__("ingestion")
        self.rag_pipeline = rag_pipeline
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Process and ingest all documents."""
        start_time = self._log_step_start(state)
        
        documents = state["documents"]
        if not documents:
            state["errors"].append("No documents provided for ingestion")
            return state
        
        try:
            # Ingest documents using the RAG pipeline
            ingestion_results = []
            ingested_doc_ids = []
            
            for doc_path in documents:
                try:
                    result = self.rag_pipeline.add_document(
                        file_path=doc_path,
                        metadata={"research_query": state["research_query"]}
                    )
                    
                    if result.success:
                        ingestion_results.append(result)
                        ingested_doc_ids.append(result.doc_id)
                        logger.info(f"Successfully ingested: {doc_path} -> {result.doc_id}")
                    else:
                        error_msg = f"Failed to ingest {doc_path}: {result.error_message}"
                        state["errors"].append(error_msg)
                        logger.error(error_msg)
                        
                except Exception as e:
                    error_msg = f"Exception ingesting {doc_path}: {str(e)}"
                    state["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Update state
            state["ingestion_results"] = ingestion_results
            state["ingested_doc_ids"] = ingested_doc_ids
            state["sources_used"] = [result.source_path for result in ingestion_results]
            
            # Add status message
            success_count = len(ingested_doc_ids)
            total_chunks = sum(result.chunks_created for result in ingestion_results)
            
            message = f"Successfully ingested {success_count}/{len(documents)} documents, created {total_chunks} chunks"
            self._add_message(state, message)
            self._log_step_end(state, start_time, message)
            
        except Exception as e:
            error_msg = f"Ingestion node failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state


class RetrievalNode(BaseWorkflowNode):
    """
    Information Retrieval Node - Finds relevant information for the research query.
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        super().__init__("retrieval")
        self.rag_pipeline = rag_pipeline
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Retrieve relevant information for the research query."""
        start_time = self._log_step_start(state)
        
        research_query = state["research_query"]
        if not state["ingested_doc_ids"]:
            state["errors"].append("No documents available for retrieval")
            return state
        
        try:
            # Query the RAG system
            query_result = self.rag_pipeline.query(
                question=research_query,
                n_results=10,  # Get more results for comprehensive analysis
                similarity_threshold=state.get("similarity_threshold", 0.3)  # Use config value, default lower
            )
            
            # Extract relevant chunks with enhanced citation tracking
            relevant_chunks = []
            for result in query_result.retrieval_context.results:
                chunk_data = {
                    "text": result.text,
                    "similarity": result.similarity,
                    "source": result.source_document,
                    "metadata": result.metadata,
                    "chunk_id": getattr(result, 'chunk_id', None),
                    "page_number": result.metadata.get('page_number'),
                    "section": result.metadata.get('section'),
                    "citation": self._generate_citation(result)
                }
                relevant_chunks.append(chunk_data)
            
            # Update state
            state["retrieval_context"] = query_result.retrieval_context
            state["relevant_chunks"] = relevant_chunks
            state["confidence_scores"]["retrieval"] = query_result.retrieval_context.avg_similarity
            
            # Add status message
            message = f"Retrieved {len(relevant_chunks)} relevant chunks (avg similarity: {query_result.retrieval_context.avg_similarity:.3f})"
            self._add_message(state, message)
            self._log_step_end(state, start_time, message)
            
        except Exception as e:
            error_msg = f"Retrieval node failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    def _generate_citation(self, result) -> str:
        """Generate proper citation for a search result."""
        metadata = result.metadata
        source = result.source_document
        
        if not source:
            return "Unknown source"
        
        # Extract filename without path
        from pathlib import Path
        source_name = Path(source).stem
        
        # Add page number if available
        page = metadata.get('page_number')
        if page:
            return f"{source_name}, p. {page}"
        
        # Add section if available
        section = metadata.get('section')
        if section:
            return f"{source_name}, {section}"
        
        return source_name


class SummarizationNode(BaseWorkflowNode):
    """
    Summarization Node - Extracts key insights and creates document summaries.
    """
    
    def __init__(self):
        super().__init__("summarization")
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Generate summaries and extract key insights using LLM."""
        start_time = self._log_step_start(state)
        
        relevant_chunks = state["relevant_chunks"]
        if not relevant_chunks:
            state["errors"].append("No relevant chunks available for summarization")
            return state
        
        try:
            # Try LLM-powered analysis first, fallback to rule-based
            try:
                from ..workflow.agents import ResearchAnalyst
                from ..output.response_processor import ResponseProcessor
                
                analyst = ResearchAnalyst()
                processor = ResponseProcessor()
                
                # Prepare context for LLM analysis
                context = {
                    "research_query": state["research_query"],
                    "relevant_chunks": relevant_chunks,
                    "document_summaries": {}
                }
                
                # Get LLM analysis
                analysis_result = analyst.analyze(context)
                
                # Process and enhance the LLM response
                processed_response = processor.process_response(
                    analysis_result.get("raw_analysis", ""), 
                    "research_analysis"
                )
                
                # Extract results from processed LLM output with enhanced citation tracking
                document_summaries = {"AI_Analysis": processed_response.cleaned_text}
                
                # Use processed insights if available, otherwise extract from analysis
                llm_insights = processed_response.key_insights or analysis_result.get("main_insights", [])
                key_insights = self._add_citations_to_insights(llm_insights, relevant_chunks)
                
                # Determine confidence from processed indicators
                conf_level = processed_response.confidence_indicators.get("overall", "moderate")
                confidence = 0.9 if conf_level == "high" else 0.7 if conf_level == "moderate" else 0.5
                
                logger.info("✅ Using enhanced LLM-powered summarization with post-processing")
                
            except Exception as llm_error:
                logger.warning(f"LLM analysis failed, using fallback: {llm_error}")
                # Fallback to original rule-based method
                
                # Group chunks by source document
                chunks_by_source = {}
                for chunk in relevant_chunks:
                    source = chunk.get("source", "unknown")
                    if source not in chunks_by_source:
                        chunks_by_source[source] = []
                    chunks_by_source[source].append(chunk)
                
                # Generate summaries for each document with citations
                document_summaries = {}
                all_insights = []
                
                for source, chunks in chunks_by_source.items():
                    combined_text = " ".join([chunk["text"] for chunk in chunks])
                    key_sentences = self._extract_key_sentences(combined_text)
                    summary = " ".join(key_sentences[:3])  # Top 3 sentences
                    
                    # Add source citations to summary
                    from pathlib import Path
                    summary_with_citation = f"{summary} [Source: {Path(source).stem}]"
                    document_summaries[source] = summary_with_citation
                    
                    # Add insights with citations
                    for sentence in key_sentences[:5]:
                        insight_with_citation = f"{sentence} [Source: {Path(source).stem}]"
                        all_insights.append(insight_with_citation)
                
                # Deduplicate and rank insights
                unique_insights = list(set(all_insights))
                key_insights = unique_insights[:10]  # Top 10 overall insights
                confidence = 0.7
            
            # Update state
            state["document_summaries"] = document_summaries
            state["key_insights"] = key_insights
            state["confidence_scores"]["summarization"] = confidence
            
            # Add status message
            message = f"Generated summaries for {len(document_summaries)} sources, extracted {len(key_insights)} key insights"
            self._add_message(state, message)
            self._log_step_end(state, start_time, message)
            
        except Exception as e:
            error_msg = f"Summarization node failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    def _extract_key_sentences(self, text: str) -> List[str]:
        """Extract key sentences from text (simplified implementation)."""
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Simple ranking by length and keyword presence
        keywords = ["important", "significant", "key", "main", "primary", "essential", "critical"]
        
        scored_sentences = []
        for sentence in sentences:
            score = len(sentence) / 100  # Length factor
            for keyword in keywords:
                if keyword.lower() in sentence.lower():
                    score += 0.5
            scored_sentences.append((score, sentence))
        
        # Sort by score and return
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        return [sentence for _, sentence in scored_sentences[:20]]
    
    def _add_citations_to_insights(self, insights: List[str], chunks: List[Dict]) -> List[str]:
        """Add proper citations to insights based on source chunks."""
        cited_insights = []
        
        for insight in insights:
            # Find the most relevant chunk for this insight
            best_match = None
            best_score = 0
            
            for chunk in chunks:
                # Simple keyword matching to find relevant source
                insight_words = set(insight.lower().split())
                chunk_words = set(chunk["text"].lower().split())
                overlap = len(insight_words & chunk_words)
                
                if overlap > best_score:
                    best_score = overlap
                    best_match = chunk
            
            if best_match:
                citation = best_match.get("citation", "Unknown source")
                cited_insight = f"{insight} [{citation}]"
            else:
                cited_insight = insight
            
            cited_insights.append(cited_insight)
        
        return cited_insights


class ReviewNode(BaseWorkflowNode):
    """
    Critical Review Node - Performs critical analysis and finds contradictions.
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        super().__init__("review")
        self.rag_pipeline = rag_pipeline
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Perform critical analysis and contradiction detection."""
        start_time = self._log_step_start(state)
        
        if not state["key_insights"]:
            state["warnings"].append("No insights available for critical review")
            return state
        
        try:
            # Find contradictions using the query engine
            contradictions = []
            
            # Use the query engine's contradiction detection
            query_engine = self.rag_pipeline.query_engine if hasattr(self.rag_pipeline, 'query_engine') else None
            
            if query_engine:
                try:
                    contradictions = query_engine.find_contradictions(state["research_query"])
                except AttributeError:
                    # Fallback: simple contradiction detection
                    contradictions = self._simple_contradiction_detection(state["relevant_chunks"])
            else:
                contradictions = self._simple_contradiction_detection(state["relevant_chunks"])
            
            # Update state
            state["contradictions"] = contradictions
            state["confidence_scores"]["review"] = 0.7 if contradictions else 0.9
            
            # Add status message
            message = f"Critical review completed, found {len(contradictions)} potential contradictions"
            self._add_message(state, message)
            self._log_step_end(state, start_time, message)
            
        except Exception as e:
            error_msg = f"Review node failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    def _simple_contradiction_detection(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple contradiction detection between chunks."""
        contradictions = []
        
        # Contradiction word pairs
        contradiction_pairs = [
            ("increases", "decreases"), ("improves", "worsens"),
            ("effective", "ineffective"), ("positive", "negative"),
            ("supports", "opposes"), ("confirms", "contradicts"),
            ("better", "worse"), ("higher", "lower")
        ]
        
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                text1 = chunk1["text"].lower()
                text2 = chunk2["text"].lower()
                
                for pos_word, neg_word in contradiction_pairs:
                    if (pos_word in text1 and neg_word in text2) or \
                       (neg_word in text1 and pos_word in text2):
                        contradictions.append({
                            "type": f"{pos_word} vs {neg_word}",
                            "chunk1": {
                                "text": chunk1["text"],
                                "source": chunk1.get("source"),
                                "similarity": chunk1.get("similarity"),
                                "citation": chunk1.get("citation", "Unknown source")
                            },
                            "chunk2": {
                                "text": chunk2["text"],
                                "source": chunk2.get("source"),
                                "similarity": chunk2.get("similarity"),
                                "citation": chunk2.get("citation", "Unknown source")
                            },
                            "source1": chunk1.get("citation", "Unknown source"),
                            "source2": chunk2.get("citation", "Unknown source")
                        })
                        break
        
        return contradictions[:5]  # Limit to top 5 contradictions


class ReportNode(BaseWorkflowNode):
    """
    Report Generation Node - Creates executive summary and final report.
    """
    
    def __init__(self):
        super().__init__("report")
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Generate final executive report."""
        start_time = self._log_step_start(state)
        
        try:
            # Generate executive summary
            executive_summary = self._generate_executive_summary(state)
            
            # Generate detailed analysis
            detailed_analysis = self._generate_detailed_analysis(state)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(state)
            
            # Calculate overall confidence score
            confidence_scores = state["confidence_scores"]
            overall_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
            
            # Update state
            state["executive_summary"] = executive_summary
            state["detailed_analysis"] = detailed_analysis
            state["recommendations"] = recommendations
            state["confidence_scores"]["overall"] = overall_confidence
            
            # Add status message
            message = f"Executive report generated (confidence: {overall_confidence:.2f})"
            self._add_message(state, message)
            self._log_step_end(state, start_time, message)
            
        except Exception as e:
            error_msg = f"Report node failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    def _generate_executive_summary(self, state: WorkflowState) -> str:
        """Generate executive summary."""
        research_query = state["research_query"]
        key_insights = state["key_insights"]
        document_count = len(state["sources_used"])
        
        summary_parts = [
            f"Research Query: {research_query}",
            f"Documents Analyzed: {document_count}",
            "",
            "Key Findings:"
        ]
        
        for i, insight in enumerate(key_insights[:5], 1):
            # Truncate insight but preserve citation
            if '[' in insight and ']' in insight:
                text_part = insight.split('[')[0].strip()
                citation_part = '[' + insight.split('[')[1]
                truncated = f"{text_part[:150]}..." if len(text_part) > 150 else text_part
                summary_parts.append(f"{i}. {truncated} {citation_part}")
            else:
                summary_parts.append(f"{i}. {insight[:200]}...")
        
        if state["contradictions"]:
            summary_parts.extend([
                "",
                f"Critical Issues: {len(state['contradictions'])} potential contradictions identified requiring further analysis."
            ])
        
        return "\n".join(summary_parts)
    
    def _generate_detailed_analysis(self, state: WorkflowState) -> str:
        """Generate detailed analysis."""
        analysis_parts = [
            f"Detailed Analysis for: {state['research_query']}",
            "=" * 50,
            ""
        ]
        
        # Document summaries
        analysis_parts.append("Document Summaries:")
        for source, summary in state["document_summaries"].items():
            analysis_parts.extend([
                f"Source: {source}",
                f"Summary: {summary}",
                ""
            ])
        
        # Key insights
        analysis_parts.extend([
            "Key Insights:",
            ""
        ])
        for i, insight in enumerate(state["key_insights"], 1):
            analysis_parts.append(f"{i}. {insight}")
        
        # Contradictions
        if state["contradictions"]:
            analysis_parts.extend([
                "",
                "Contradictions and Conflicts:",
                ""
            ])
            for i, contradiction in enumerate(state["contradictions"], 1):
                analysis_parts.extend([
                    f"{i}. {contradiction['type']}",
                    f"   Evidence 1: {contradiction['chunk1']['text'][:100]}... [{contradiction.get('source1', 'Unknown')}]",
                    f"   Evidence 2: {contradiction['chunk2']['text'][:100]}... [{contradiction.get('source2', 'Unknown')}]",
                    ""
                ])
        
        return "\n".join(analysis_parts)
    
    def _generate_recommendations(self, state: WorkflowState) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Based on confidence scores
        avg_confidence = state["confidence_scores"].get("overall", 0.0)
        
        if avg_confidence > 0.8:
            recommendations.append("High confidence in findings - proceed with implementation")
        elif avg_confidence > 0.6:
            recommendations.append("Moderate confidence - consider additional validation")
        else:
            recommendations.append("Low confidence - conduct further research before decisions")
        
        # Based on contradictions
        if state["contradictions"]:
            recommendations.append("Resolve identified contradictions through expert consultation")
            recommendations.append("Prioritize analysis of conflicting sources")
        
        # Based on document coverage
        if len(state["sources_used"]) < 3:
            recommendations.append("Expand document base for more comprehensive analysis")
        
        # General recommendations
        recommendations.extend([
            "Monitor developments in this area for emerging insights",
            "Consider stakeholder input for complete perspective"
        ])
        
        return recommendations[:5]  # Limit to top 5 recommendations