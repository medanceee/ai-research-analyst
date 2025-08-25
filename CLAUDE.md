# 📊 AI-Powered Research Analyst (RAG + LangGraph)

## 🎯 The Vision
Building an AI-powered research analyst that acts like a junior consultant or enterprise research associate. Instead of simply summarizing PDFs, the system:

- **Ingests scattered sources** (PDFs, websites, internal docs)
- **Uses Retrieval-Augmented Generation (RAG)** to fetch accurate context
- **Chains tasks with LangGraph** to produce structured, step-wise analysis
- **Delivers executive-ready briefings** with citations, critical reviews, and structured insights

**Essentially:** "Upload 50 reports → wake up to a consulting-grade briefing deck."

## 🚨 The Problem It Solves

- **Fragmented sources:** Information locked in PDFs, web pages, databases, internal wikis
- **Manual synthesis takes hours:** Analysts spend 40–60% of project time on consolidation
- **Quality inconsistency:** Output depends on individual skill; enterprises want standardization
- **Slow turnaround:** Strategic insights can take weeks — slowing decision-making

## 💡 The Core Solution
An LLM workflow agent that automates the research-to-briefing pipeline:

### 🛠️ Workflow Steps (via LangGraph)

**1. Ingestion Layer**
- Accepts PDFs, websites, internal docs
- Converts to text + metadata
- Stores embeddings in vector DB

**2. Retrieval & Context Building (RAG)**
- For each query/topic, retrieves semantically relevant chunks
- Builds multi-source context with source references

**3. Summarization & Structuring**
- Summarizes each source individually → produces knowledge base
- Organizes information into structured categories (Market Trends, Risks, Opportunities)

**4. Critical Review & Synthesis**
- Cross-analyzes different documents
- Highlights contradictions, gaps, or consensus
- Extracts "so what" insights

**5. Executive Briefing Generation**
- Outputs in business-ready formats:
  - Executive summary
  - SWOT / Porter's 5 Forces / Risk matrix
  - Recommendations
- Includes citations & traceability

## Project Overview
This enterprise-grade system transforms scattered research into actionable intelligence through automated RAG workflows and structured analysis pipelines.

## System Architecture

### Core Components
1. **Document Ingestion Engine** - Multi-format document processing
2. **RAG Pipeline** - Vector embeddings + semantic search  
3. **LangGraph Orchestrator** - Task chaining workflow
4. **Analysis Engine** - Summarization + critical review
5. **Executive Briefing Generator** - Structured output with citations

### Tech Stack
- **LangChain/LangGraph** - Workflow orchestration (5-node sequential pipeline)
- **ChromaDB** - Vector database
- **DeepSeek LLM** - Local LLM processing via Ollama
- **CLI Interface** - Production command-line tool
- **PyPDF2** - Document parsing
- **Sentence Transformers** - Embeddings

### Workflow
```
Input Sources → Document Processing → Embedding → Vector Store
                                                     ↓
Enhanced Reports ← Critical Review ← Summarization ← RAG Retrieval
```

### LangGraph Chain (5-Node Pipeline)
```
Input → [INGESTION] → [RETRIEVAL] → [SUMMARIZATION] → [REVIEW] → [REPORT] → Output
         ↓            ↓             ↓               ↓         ↓
    Process PDFs   Find Relevant   Extract Key    Critical  Generate
    Store in RAG   Chunks         Insights       Analysis  Final Report
```

## Current Status
**PRODUCTION-READY CLI SYSTEM** ✅

### Completed Components:
✅ **Core RAG Pipeline** - ChromaDB vector store, embedding service, semantic search
✅ **Document Processing** - Multi-format support (PDF, DOCX, TXT, URLs)
✅ **LangGraph Workflow** - 5-node orchestration pipeline
✅ **Enhanced LLM Processing** - Professional response post-processing
✅ **CLI Interface** - Production command-line tool
✅ **Dual Output System** - System analytics JSON + user research report
✅ **Citation Tracking** - Page-level citations with source attribution
✅ **Response Enhancement** - Professional language processing and cleanup

### Current Focus:
🔄 **IMPROVING REPORT DETAIL AND DEPTH** - Working on comprehensive detailed reports

### Recent Achievements:
- ✅ Removed all UI components (Streamlit, web interface)
- ✅ Streamlined to CLI-only operation
- ✅ Enhanced LLM response quality with post-processing
- ✅ Improved citation tracking and source references
- ✅ Professional report formatting and structure
- ✅ 24 unnecessary files and 11 directories cleaned up
- ✅ System produces enterprise-grade analysis

### Next Priority:
🎯 **Enhanced Report Detail** - Creating more comprehensive, detailed analysis reports with:
- Multi-perspective analysis framework
- Quantitative data extraction
- Deeper methodological analysis
- Extended contextual insights

## Built Components:
✅ **Production CLI Interface** (`research_cli.py`)
✅ **Enhanced Document Processing** - Page-level metadata extraction
✅ **Complete RAG System** - Vector embeddings, semantic search, chunking strategies
✅ **LangGraph Workflow** - Sequential 5-node analysis pipeline
✅ **Response Post-Processing** - Professional language enhancement
✅ **System Analytics** - Performance metrics and diagnostics

## Implementation Checklist
- [x] Set up core dependencies and requirements  
- [x] Implement document ingestion module (PDFs, websites, docs)
- [x] Build RAG system with vector database
- [x] Create LangGraph workflow for task chaining
- [x] Implement retrieval and summarization components
- [x] Build critical review and executive briefing generator
- [x] Add citation tracking and structured output formatting
- [x] Remove UI components and optimize for CLI-only
- [x] Enhance LLM response quality and post-processing
- [ ] **IN PROGRESS: Create more detailed comprehensive reports**
- [ ] Add quantitative analysis extraction
- [ ] Implement multi-perspective analysis framework

## Development Approach
- **CLI-First:** Production command-line interface
- **Local Development:** DeepSeek R1 1.5B via Ollama + sentence-transformers
- **Enterprise-Ready:** Professional analysis with citations
- **Quality-Focused:** Post-processed LLM responses with cleanup

## User Usage
```bash
# Basic analysis
python research_cli.py paper.pdf -q "What are the main findings?"

# Advanced analysis with custom outputs  
python research_cli.py *.pdf -q "Compare methodologies" \
  --analytics system_report.json --report findings.md
```

## Hardware Setup
- RTX 3050 Laptop GPU (4GB VRAM)
- DeepSeek R1 1.5B model via Ollama
- Sentence-transformers for embeddings
- ChromaDB for vector storage

## Output Files
1. **system_analytics.json** - Performance metrics, system health, processing stats
2. **research_report.md** - Professional analysis with citations and insights

## Notes
- **CLI-only operation** - All UI components removed
- **Enhanced LLM responses** - Professional post-processing applied
- **Citation tracking** - Page-level source attribution
- **Enterprise-ready** - Production deployment ready
- **Performance optimized** - Clean, streamlined codebase

# Current Session Status
🔄 **WORKING ON:** Improving report detail and comprehensiveness
🎯 **GOAL:** Generate more thorough, detailed analysis reports
📊 **STATUS:** System is production-ready, enhancing output quality

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.