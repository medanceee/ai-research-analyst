"""
LangGraph Workflow Orchestration for RAG Pipeline

This package provides workflow orchestration using LangGraph to chain together
RAG components into a structured, step-wise analysis pipeline.

Key Components:
- ResearchWorkflow: Main workflow orchestrator
- WorkflowState: State management between nodes
- Analysis nodes for summarization, review, and reporting
- Integration with existing RAG components
"""

from .workflow import (
    ResearchWorkflow,
    WorkflowState,
    ResearchResult,
    create_research_workflow
)

from .nodes import (
    IngestionNode,
    RetrievalNode,
    SummarizationNode,
    ReviewNode,
    ReportNode
)

from .agents import (
    ResearchAnalyst,
    CriticalReviewer,
    ExecutiveSummarizer
)

__version__ = "1.0.0"

__all__ = [
    # Main workflow
    'ResearchWorkflow',
    'WorkflowState', 
    'ResearchResult',
    'create_research_workflow',
    
    # Workflow nodes
    'IngestionNode',
    'RetrievalNode',
    'SummarizationNode', 
    'ReviewNode',
    'ReportNode',
    
    # AI agents
    'ResearchAnalyst',
    'CriticalReviewer',
    'ExecutiveSummarizer'
]