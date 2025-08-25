#!/usr/bin/env python3
"""
🧠 AI Research Analyst - Command Line Interface
Production-ready CLI that produces two key outputs:
1. System Analytics JSON (performance, metrics, diagnostics)
2. User Research Report (findings, insights, recommendations)
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.research_analyst.workflow import create_research_workflow, ResearchResult
    from src.research_analyst.rag.rag_pipeline import RAGPipeline
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    sys.exit(1)


class SystemAnalytics:
    """System performance and diagnostic analytics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            "processing_times": {},
            "document_stats": {},
            "retrieval_performance": {},
            "llm_analysis": {},
            "error_tracking": [],
            "memory_usage": {},
            "workflow_health": {}
        }
    
    def record_timing(self, operation: str, duration: float):
        """Record operation timing."""
        self.metrics["processing_times"][operation] = {
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        }
    
    def record_document_stats(self, doc_count: int, chunk_count: int, avg_chunk_size: float):
        """Record document processing statistics."""
        self.metrics["document_stats"] = {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "average_chunk_size": avg_chunk_size,
            "processing_rate": chunk_count / self.get_total_time() if self.get_total_time() > 0 else 0
        }
    
    def record_retrieval_performance(self, query_count: int, avg_similarity: float, retrieval_time: float):
        """Record RAG retrieval performance."""
        self.metrics["retrieval_performance"] = {
            "total_queries": query_count,
            "average_similarity_score": avg_similarity,
            "retrieval_time_seconds": retrieval_time,
            "queries_per_second": query_count / retrieval_time if retrieval_time > 0 else 0
        }
    
    def record_workflow_health(self, success_rate: float, errors: List[str], warnings: List[str]):
        """Record overall workflow health."""
        self.metrics["workflow_health"] = {
            "success_rate": success_rate,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
            "overall_status": "healthy" if success_rate > 0.8 and len(errors) == 0 else "degraded"
        }
    
    def get_total_time(self) -> float:
        """Get total processing time."""
        return time.time() - self.start_time
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate complete analytics report."""
        return {
            "system_info": {
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "total_processing_time": self.get_total_time(),
                "python_version": sys.version,
                "platform": sys.platform
            },
            "performance_metrics": self.metrics,
            "efficiency_scores": {
                "processing_efficiency": self._calculate_processing_efficiency(),
                "retrieval_accuracy": self._calculate_retrieval_accuracy(),
                "overall_system_health": self._calculate_system_health()
            },
            "recommendations": self._generate_system_recommendations()
        }
    
    def _calculate_processing_efficiency(self) -> float:
        """Calculate processing efficiency score."""
        doc_stats = self.metrics.get("document_stats", {})
        if not doc_stats:
            return 0.0
        
        processing_rate = doc_stats.get("processing_rate", 0)
        # Normalize to 0-1 scale (assume 10 chunks/second is excellent)
        return min(processing_rate / 10.0, 1.0)
    
    def _calculate_retrieval_accuracy(self) -> float:
        """Calculate retrieval accuracy score."""
        retrieval = self.metrics.get("retrieval_performance", {})
        return retrieval.get("average_similarity_score", 0.0)
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score."""
        workflow = self.metrics.get("workflow_health", {})
        success_rate = workflow.get("success_rate", 0.0)
        error_penalty = min(workflow.get("error_count", 0) * 0.1, 0.5)
        return max(success_rate - error_penalty, 0.0)
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate system optimization recommendations."""
        recommendations = []
        
        # Processing efficiency recommendations
        efficiency = self._calculate_processing_efficiency()
        if efficiency < 0.5:
            recommendations.append("Consider optimizing document chunking strategy for better processing speed")
        
        # Retrieval accuracy recommendations
        accuracy = self._calculate_retrieval_accuracy()
        if accuracy < 0.7:
            recommendations.append("Improve similarity threshold or embedding model for better retrieval accuracy")
        
        # Error handling recommendations
        errors = self.metrics.get("workflow_health", {}).get("error_count", 0)
        if errors > 0:
            recommendations.append("Address workflow errors to improve system reliability")
        
        if not recommendations:
            recommendations.append("System is performing optimally - no immediate recommendations")
        
        return recommendations


class UserReportGenerator:
    """Generate user-facing research reports."""
    
    def __init__(self, result: ResearchResult):
        self.result = result
    
    def generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report with enhanced processing."""
        try:
            # Import response processor for enhanced formatting
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from src.research_analyst.output.response_processor import ResponseProcessor, create_professional_summary, ProcessedResponse
            
            processor = ResponseProcessor()
            
            # Create a mock processed response for consistent formatting
            processed_response = ProcessedResponse(
                cleaned_text=self.result.detailed_analysis,
                extracted_sections={},
                key_insights=self.result.key_insights,
                confidence_indicators={"overall": "moderate"},
                citations=[str(Path(source).name) for source in self.result.sources_used],
                formatting_applied=["enhanced_formatting"]
            )
            
            # Generate professional summary
            metadata = {
                'processing_time': self.result.processing_time,
                'document_count': self.result.document_count,
                'chunk_count': self.result.chunk_count
            }
            
            report = create_professional_summary(processed_response, self.result.query, metadata)
            
        except Exception as e:
            # Fallback to original format if enhancement fails
            report = self._generate_basic_report()
        
        # Add strategic recommendations
        if self.result.recommendations:
            report += "\n## Strategic Recommendations\n"
            for i, rec in enumerate(self.result.recommendations, 1):
                report += f"{i}. {rec}\n"
        
        # Add contradictions if any
        if self.result.contradictions:
            report += "\n## Critical Issues & Contradictions\n"
            for i, contradiction in enumerate(self.result.contradictions, 1):
                report += f"{i}. **{contradiction.get('type', 'Conflict detected')}**\n"
                report += f"   - Evidence A: {contradiction.get('source1', 'Unknown')}\n"
                report += f"   - Evidence B: {contradiction.get('source2', 'Unknown')}\n\n"
        
        return report
    
    def _generate_basic_report(self) -> str:
        """Generate basic report as fallback."""
        report = f"""# Research Analysis Report

## Query
**{self.result.query}**

## Executive Summary
{self.result.executive_summary}

## Detailed Analysis
{self.result.detailed_analysis}

## Key Insights
"""
        
        for i, insight in enumerate(self.result.key_insights, 1):
            report += f"{i}. {insight}\n"
        
        report += f"""

## Sources Analyzed
"""
        for i, source in enumerate(self.result.sources_used, 1):
            source_name = Path(source).name
            report += f"{i}. {source_name}\n"
        
        report += f"""

## Analysis Metrics
- **Documents Processed**: {self.result.document_count}
- **Text Chunks Analyzed**: {self.result.chunk_count}
- **Processing Time**: {self.result.processing_time:.1f} seconds
- **Confidence Score**: {self.result.confidence_score:.2f}/1.0

---
*Report generated by AI Research Analyst v1.0.0*
"""
        return report
    
    def generate_json_report(self) -> Dict[str, Any]:
        """Generate structured JSON report."""
        return {
            "metadata": {
                "query": self.result.query,
                "timestamp": datetime.now().isoformat(),
                "processing_time": self.result.processing_time,
                "confidence_score": self.result.confidence_score
            },
            "analysis": {
                "executive_summary": self.result.executive_summary,
                "detailed_analysis": self.result.detailed_analysis,
                "key_insights": self.result.key_insights,
                "recommendations": self.result.recommendations
            },
            "evidence": {
                "contradictions": self.result.contradictions,
                "sources_analyzed": self.result.sources_used,
                "document_count": self.result.document_count,
                "chunk_count": self.result.chunk_count
            }
        }


def run_research_analysis(
    pdf_files: List[str],
    research_query: str,
    collection_name: str = "research_cli",
    similarity_threshold: float = 0.7,
    max_chunks: int = 15,
    analytics_file: str = "system_analytics.json",
    report_file: str = "research_report.md"
) -> tuple:
    """
    Run complete research analysis and generate output files.
    
    Returns:
        tuple: (ResearchResult, SystemAnalytics)
    """
    
    # Initialize system analytics
    analytics = SystemAnalytics()
    
    print("🧠 AI Research Analyst - Production CLI")
    print("=" * 50)
    
    # Validate input files
    valid_files = []
    for pdf_file in pdf_files:
        pdf_path = Path(pdf_file)
        if pdf_path.exists() and pdf_path.suffix.lower() == '.pdf':
            valid_files.append(str(pdf_path.absolute()))
            print(f"✅ {pdf_path.name}")
        else:
            print(f"❌ {pdf_file} (not found or not PDF)")
    
    if not valid_files:
        print("❌ No valid PDF files found!")
        return None, analytics
    
    print(f"\n📋 Configuration:")
    print(f"   Documents: {len(valid_files)}")
    print(f"   Query: {research_query}")
    print(f"   Similarity Threshold: {similarity_threshold}")
    print(f"   Max Retrieval Chunks: {max_chunks}")
    
    # Initialize workflow with enhanced configuration
    print(f"\n🔧 Initializing enhanced workflow...")
    config = {
        "max_retrieval_chunks": max_chunks,
        "similarity_threshold": similarity_threshold,
        "enable_contradiction_detection": True,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "enable_advanced_analysis": True
    }
    
    start_time = time.time()
    
    workflow = create_research_workflow(
        collection_name=collection_name,
        persist_directory=f"./data/{collection_name}",
        embedding_model="all-MiniLM-L6-v2",
        config=config
    )
    
    workflow_init_time = time.time() - start_time
    analytics.record_timing("workflow_initialization", workflow_init_time)
    
    # Run research analysis
    print(f"\n🚀 Starting research analysis...")
    print("⏳ Processing documents and generating insights...")
    
    analysis_start = time.time()
    
    result = workflow.run_research(
        research_query=research_query,
        documents=valid_files
    )
    
    analysis_time = time.time() - analysis_start
    analytics.record_timing("research_analysis", analysis_time)
    
    # Record performance metrics
    analytics.record_document_stats(
        result.document_count, 
        result.chunk_count,
        result.chunk_count / result.document_count if result.document_count > 0 else 0
    )
    
    # Generate outputs
    print(f"\n📊 Generating output files...")
    
    # Generate user report
    report_gen = UserReportGenerator(result)
    
    # Save user report (markdown)
    if report_file.endswith('.md'):
        user_report = report_gen.generate_markdown_report()
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(user_report)
    else:
        user_report = report_gen.generate_json_report()
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(user_report, f, indent=2, ensure_ascii=False)
    
    # Record final analytics
    analytics.record_workflow_health(
        success_rate=1.0 if result.confidence_score > 0.5 else 0.5,
        errors=[],
        warnings=[]
    )
    
    # Save system analytics
    analytics_report = analytics.generate_report()
    with open(analytics_file, 'w', encoding='utf-8') as f:
        json.dump(analytics_report, f, indent=2, ensure_ascii=False)
    
    # Display summary
    total_time = analytics.get_total_time()
    print(f"\n✅ Analysis Complete!")
    print(f"   Processing Time: {total_time:.1f}s")
    print(f"   Documents: {result.document_count}")
    print(f"   Chunks: {result.chunk_count}")
    print(f"   Confidence: {result.confidence_score:.2f}")
    
    print(f"\n📁 Output Files:")
    print(f"   📊 System Analytics: {analytics_file}")
    print(f"   📋 Research Report: {report_file}")
    
    return result, analytics


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🧠 AI Research Analyst - Production CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python research_cli.py paper.pdf -q "What are the main findings?"
  
  # Multiple papers with custom output
  python research_cli.py paper1.pdf paper2.pdf \\
    -q "Compare methodologies" \\
    -s 0.8 -m 20 \\
    --analytics system_metrics.json \\
    --report findings.md
  
  # Batch analysis
  python research_cli.py *.pdf \\
    -q "Identify research trends" \\
    --collection trend_analysis
        """
    )
    
    # Required arguments
    parser.add_argument(
        'pdf_files',
        nargs='+',
        help='PDF files to analyze'
    )
    
    parser.add_argument(
        '-q', '--query',
        required=True,
        help='Research question or analysis focus'
    )
    
    # Optional configuration
    parser.add_argument(
        '-c', '--collection',
        default='research_cli',
        help='ChromaDB collection name (default: research_cli)'
    )
    
    parser.add_argument(
        '-s', '--similarity',
        type=float,
        default=0.7,
        help='Similarity threshold 0.0-1.0 (default: 0.7)'
    )
    
    parser.add_argument(
        '-m', '--max-chunks',
        type=int,
        default=15,
        help='Maximum retrieval chunks (default: 15)'
    )
    
    # Output files
    parser.add_argument(
        '--analytics',
        default='system_analytics.json',
        help='System analytics output file (default: system_analytics.json)'
    )
    
    parser.add_argument(
        '--report',
        default='research_report.md',
        help='Research report output file (default: research_report.md)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='AI Research Analyst CLI v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Validate parameters
    if not 0.0 <= args.similarity <= 1.0:
        print("❌ Similarity threshold must be 0.0-1.0")
        sys.exit(1)
    
    if args.max_chunks < 1:
        print("❌ Max chunks must be ≥ 1")
        sys.exit(1)
    
    # Run analysis
    try:
        result, analytics = run_research_analysis(
            pdf_files=args.pdf_files,
            research_query=args.query,
            collection_name=args.collection,
            similarity_threshold=args.similarity,
            max_chunks=args.max_chunks,
            analytics_file=args.analytics,
            report_file=args.report
        )
        
        if result is None:
            sys.exit(1)
            
        print(f"\n🎯 Research analysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()