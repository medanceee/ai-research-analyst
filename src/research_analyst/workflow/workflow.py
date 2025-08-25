"""
Main LangGraph Workflow for Research Analysis Pipeline

This module implements the core workflow orchestration using LangGraph,
integrating document ingestion, retrieval, analysis, and reporting into
a structured multi-step process.
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
import logging
import time
import json
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..rag.rag_pipeline import RAGPipeline, QueryResult, IngestionResult
from ..rag.retrieval import RetrievalContext

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """
    State object passed between workflow nodes.
    Contains all data needed for the research analysis pipeline.
    """
    # Input data
    research_query: str
    documents: List[str]  # File paths or URLs
    
    # Processing state
    messages: Annotated[List[BaseMessage], add_messages]
    current_step: str
    
    # Document ingestion results
    ingestion_results: List[IngestionResult]
    ingested_doc_ids: List[str]
    
    # Retrieval results
    retrieval_context: Optional[RetrievalContext]
    relevant_chunks: List[Dict[str, Any]]
    
    # Analysis results
    document_summaries: Dict[str, str]  # doc_id -> summary
    key_insights: List[str]
    contradictions: List[Dict[str, Any]]
    
    # Final outputs
    executive_summary: str
    detailed_analysis: str
    recommendations: List[str]
    
    # Metadata
    workflow_start_time: float
    step_timings: Dict[str, float]
    sources_used: List[str]
    confidence_scores: Dict[str, float]
    
    # Error handling
    errors: List[str]
    warnings: List[str]


@dataclass
class ResearchResult:
    """Final result from the research workflow."""
    query: str
    executive_summary: str
    detailed_analysis: str
    key_insights: List[str]
    recommendations: List[str]
    sources_used: List[str]
    contradictions: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    document_count: int
    chunk_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "executive_summary": self.executive_summary,
            "detailed_analysis": self.detailed_analysis,
            "key_insights": self.key_insights,
            "recommendations": self.recommendations,
            "sources_used": self.sources_used,
            "contradictions": self.contradictions,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count
        }
    
    def save_report(self, output_path: str) -> bool:
        """Save research result to file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Research report saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return False


class ResearchWorkflow:
    """
    Main research workflow orchestrator using LangGraph.
    
    Implements a 5-node workflow:
    1. Ingestion Node - Process and store documents
    2. Retrieval Node - Find relevant information
    3. Summarization Node - Extract key insights
    4. Review Node - Critical analysis and contradictions
    5. Report Node - Generate executive briefing
    """
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize research workflow.
        
        Args:
            rag_pipeline: RAG pipeline for document processing and retrieval
            config: Workflow configuration options
        """
        self.rag_pipeline = rag_pipeline
        self.config = config or {}
        
        # Workflow configuration
        self.max_retrieval_chunks = self.config.get('max_retrieval_chunks', 10)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.6)
        self.enable_contradiction_detection = self.config.get('enable_contradiction_detection', True)
        
        # Build the workflow graph
        self.workflow_graph = self._build_workflow()
        
        logger.info("ResearchWorkflow initialized with LangGraph")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(WorkflowState)
        
        # Import nodes (we'll create these next)
        from .nodes import (
            IngestionNode, RetrievalNode, SummarizationNode, 
            ReviewNode, ReportNode
        )
        
        # Initialize nodes
        ingestion_node = IngestionNode(self.rag_pipeline)
        retrieval_node = RetrievalNode(self.rag_pipeline)
        summarization_node = SummarizationNode()
        review_node = ReviewNode(self.rag_pipeline)
        report_node = ReportNode()
        
        # Add nodes to workflow
        workflow.add_node("ingestion", ingestion_node.execute)
        workflow.add_node("retrieval", retrieval_node.execute)
        workflow.add_node("summarization", summarization_node.execute)
        workflow.add_node("review", review_node.execute)
        workflow.add_node("report", report_node.execute)
        
        # Define workflow edges
        workflow.add_edge("ingestion", "retrieval")
        workflow.add_edge("retrieval", "summarization")
        workflow.add_edge("summarization", "review")
        workflow.add_edge("review", "report")
        workflow.add_edge("report", END)
        
        # Set entry point
        workflow.set_entry_point("ingestion")
        
        return workflow.compile()
    
    def run_research(
        self,
        research_query: str,
        documents: List[str],
        save_report: Optional[str] = None
    ) -> ResearchResult:
        """
        Run the complete research analysis workflow.
        
        Args:
            research_query: Research question or topic
            documents: List of document paths or URLs to analyze
            save_report: Optional path to save the final report
            
        Returns:
            ResearchResult with comprehensive analysis
        """
        start_time = time.time()
        
        # Initialize workflow state
        initial_state: WorkflowState = {
            "research_query": research_query,
            "documents": documents,
            "messages": [HumanMessage(content=f"Research query: {research_query}")],
            "current_step": "initialization",
            "similarity_threshold": self.similarity_threshold,
            "ingestion_results": [],
            "ingested_doc_ids": [],
            "retrieval_context": None,
            "relevant_chunks": [],
            "document_summaries": {},
            "key_insights": [],
            "contradictions": [],
            "executive_summary": "",
            "detailed_analysis": "",
            "recommendations": [],
            "workflow_start_time": start_time,
            "step_timings": {},
            "sources_used": [],
            "confidence_scores": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Execute the workflow
            logger.info(f"Starting research workflow for query: {research_query}")
            final_state = self.workflow_graph.invoke(initial_state)
            
            # Create result object
            processing_time = time.time() - start_time
            
            result = ResearchResult(
                query=research_query,
                executive_summary=final_state.get("executive_summary", ""),
                detailed_analysis=final_state.get("detailed_analysis", ""),
                key_insights=final_state.get("key_insights", []),
                recommendations=final_state.get("recommendations", []),
                sources_used=final_state.get("sources_used", []),
                contradictions=final_state.get("contradictions", []),
                confidence_score=final_state.get("confidence_scores", {}).get("overall", 0.0),
                processing_time=processing_time,
                document_count=len(final_state.get("ingested_doc_ids", [])),
                chunk_count=len(final_state.get("relevant_chunks", []))
            )
            
            # Save report if requested
            if save_report:
                result.save_report(save_report)
            
            logger.info(f"Research workflow completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            # Return error result
            return ResearchResult(
                query=research_query,
                executive_summary=f"Workflow failed: {str(e)}",
                detailed_analysis="",
                key_insights=[],
                recommendations=[],
                sources_used=[],
                contradictions=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                document_count=0,
                chunk_count=0
            )
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get workflow configuration and status."""
        return {
            "max_retrieval_chunks": self.max_retrieval_chunks,
            "similarity_threshold": self.similarity_threshold,
            "enable_contradiction_detection": self.enable_contradiction_detection,
            "rag_pipeline_stats": self.rag_pipeline.get_pipeline_stats()
        }


def create_research_workflow(
    collection_name: str = "research_workflow",
    persist_directory: str = "./workflow_data",
    embedding_model: str = "all-MiniLM-L6-v2",
    config: Optional[Dict[str, Any]] = None
) -> ResearchWorkflow:
    """
    Factory function to create a complete research workflow.
    
    Args:
        collection_name: Name for the document collection
        persist_directory: Directory for persistent storage
        embedding_model: Embedding model to use
        config: Additional configuration options
        
    Returns:
        Configured ResearchWorkflow instance
    """
    # Create RAG pipeline
    rag_pipeline = RAGPipeline(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        config=config
    )
    
    # Create workflow
    workflow = ResearchWorkflow(rag_pipeline, config)
    
    logger.info(f"Created research workflow with collection: {collection_name}")
    return workflow


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create workflow
    workflow = create_research_workflow(
        collection_name="demo_research",
        persist_directory="./demo_workflow_data"
    )
    
    # Example research query
    research_query = "What are the key benefits and limitations of vector databases for AI applications?"
    
    # Example documents (in a real scenario, these would be actual file paths)
    documents = [
        "sample_doc_1.txt",  # These would be real files
        "sample_doc_2.pdf"
    ]
    
    # Note: This example won't run without actual documents
    # In practice, you would provide real file paths or URLs
    print("ResearchWorkflow created successfully!")
    print(f"Ready to analyze: {research_query}")
    print("Workflow status:", workflow.get_workflow_status())