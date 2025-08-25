# 📊 AI-Powered Research Analyst

**Production-ready CLI system** that transforms scattered research documents into structured, executive-ready analysis reports using RAG and LangGraph orchestration.

## 🎯 Overview

Building an AI-powered research analyst that acts like a junior consultant or enterprise research associate. The system:

- **Ingests scattered sources** (PDFs, websites, internal docs)
- **Uses Retrieval-Augmented Generation (RAG)** for accurate context retrieval
- **Chains tasks with LangGraph** for structured, step-wise analysis
- **Delivers executive-ready briefings** with citations and critical insights

**Essentially:** "Upload research papers → receive consulting-grade briefing reports."

## ✅ Current Status: Production Ready

- ✅ **Core RAG Pipeline** - ChromaDB vector store, embedding service, semantic search
- ✅ **Document Processing** - Multi-format support (PDF, DOCX, TXT, URLs)
- ✅ **LangGraph Workflow** - 5-node orchestration pipeline
- ✅ **Enhanced LLM Processing** - Professional response post-processing
- ✅ **CLI Interface** - Production command-line tool
- ✅ **Citation Tracking** - Page-level source attribution

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Or create conda environment
conda env create -f environment.yml
conda activate research_analyst
```

### 2. Local LLM Setup (DeepSeek via Ollama)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull DeepSeek model
ollama pull deepseek-r1:1.5b
```

### 3. Run Analysis
```bash
# Basic analysis
python research_cli.py paper.pdf -q "What are the main findings?"

# Advanced analysis with custom outputs
python research_cli.py *.pdf -q "Compare methodologies" \
  --analytics system_report.json --report findings.md
```

### 4. Output Files
- **system_analytics.json** - Performance metrics, system diagnostics
- **research_report.md** - Professional analysis with citations

## 🏗️ System Architecture

### Core Components
1. **Document Ingestion Engine** - Multi-format document processing
2. **RAG Pipeline** - Vector embeddings + semantic search  
3. **LangGraph Orchestrator** - Task chaining workflow
4. **Analysis Engine** - Summarization + critical review
5. **Executive Briefing Generator** - Structured output with citations

### Tech Stack
- **LangChain/LangGraph** - Workflow orchestration
- **ChromaDB** - Vector database
- **DeepSeek LLM** - Local LLM processing via Ollama
- **Sentence Transformers** - Embeddings
- **CLI Interface** - Production command-line tool

### LangGraph Workflow (5-Node Pipeline)
```
Input → [INGESTION] → [RETRIEVAL] → [SUMMARIZATION] → [REVIEW] → [REPORT] → Output
         ↓            ↓             ↓               ↓         ↓
    Process PDFs   Find Relevant   Extract Key    Critical  Generate
    Store in RAG   Chunks         Insights       Analysis  Final Report
```

## 📁 Project Structure

```
research_analyst/
├── research_cli.py             # Main CLI interface
├── src/
│   ├── research_analyst/       # Core system
│   │   ├── enhanced_research_analyst.py
│   │   ├── rag/               # RAG pipeline
│   │   │   ├── rag_pipeline.py
│   │   │   ├── vector_store.py
│   │   │   ├── embeddings.py
│   │   │   └── retrieval.py
│   │   ├── workflow/          # LangGraph orchestration
│   │   │   ├── workflow.py
│   │   │   ├── nodes.py
│   │   │   └── agents.py
│   │   ├── output/            # Report generation
│   │   │   ├── response_processor.py
│   │   │   └── markdown_formatter.py
│   │   └── analysis/          # Advanced analysis
│   │       ├── fact_verifier.py
│   │       └── contradiction_detector.py
│   ├── core/                  # Document processing
│   │   ├── document_ingestion.py
│   │   └── file_processors.py
│   └── config/                # Configuration
│       └── settings.py
├── tests/                     # Test suite
├── requirements.txt           # Dependencies
├── environment.yml           # Conda environment
├── CLAUDE.md                 # Project documentation
└── README.md                 # This file
```

## 🔧 Hardware Requirements

- **GPU**: RTX 3050+ (4GB VRAM recommended)
- **CPU**: Multi-core processor
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for models and embeddings

## 🚀 Features

### ✅ Production Ready
- **CLI-only operation** - Streamlined interface
- **Enhanced LLM responses** - Professional post-processing
- **Citation tracking** - Page-level source attribution
- **Enterprise-ready** - Production deployment ready
- **Performance optimized** - Clean, streamlined codebase

### 🔄 Current Development
- Improving report detail and comprehensiveness
- Enhanced quantitative analysis extraction
- Multi-perspective analysis framework

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request