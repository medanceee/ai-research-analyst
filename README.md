# AI Research Analyst

AI-powered research assistant that processes documents and generates structured analysis reports using RAG and LangGraph workflows.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Setup Ollama and DeepSeek model
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull deepseek-r1:1.5b
```

## Usage

```bash
# Basic analysis
python research_cli.py paper.pdf -q "What are the main findings?"

# Multiple documents
python research_cli.py *.pdf -q "Compare methodologies"
```

## Features

- Document processing (PDF, DOCX, TXT)
- RAG pipeline with ChromaDB
- LangGraph workflow orchestration
- Local LLM via Ollama (DeepSeek)
- Structured report generation