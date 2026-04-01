# 📄 Local RAG Application

A **Retrieval-Augmented Generation (RAG)** system that combines document retrieval, semantic search, and large language models to answer domain-specific questions with source citations and quality metrics.

## 🎯 Overview

This application enables users to upload PDF documents and ask questions about their contents. The system intelligently retrieves relevant document chunks, reranks them using a cross-encoder, and generates factual answers using an LLM—all while maintaining conversation history and evaluating response quality using RAGAS metrics.

**Key Achievement:** 76% average answer correctness across 10 benchmark queries.

## ✨ Features

- **Hybrid Retrieval:** Combines semantic search (FAISS) + keyword search (BM25) for comprehensive coverage
- **Intelligent Reranking:** Cross-encoder based document reordering for precision
- **Multi-turn Conversations:** Maintains conversation history (5 Q&A pairs) for contextual awareness
- **Source Citation:** Generated answers include PDF source and page references
- **Quality Evaluation:** RAGAS metrics for answer correctness, faithfulness, and retrieval quality
- **Local Execution:** Uses Ollama for local LLM inference (no external API calls)
- **User-Friendly Interface:** Streamlit-based web UI for easy interaction
- **Automatic Caching:** FAISS indexes saved locally for fast retrieval

## 🏗️ Architecture

```
┌─────────────────────┐
│   User Query (Q)    │
└──────────┬──────────┘
           ▼
┌───────────────────────────────────────┐   ┌────────────────────────────┐
│  Vector Search (FAISS, k=20)          │ + │  BM25 Keyword Search (k=20)|
│  - Semantic similarity                │   │  - TF-IDF term matching    |
└──────────────┬────────────────────────┘   └─────────────┬──────────────┘
               ▼                                          ▼
┌────────────────────────────────────────────────────────┐
│         Merge & Deduplicate Retrieved Docs             │
└──────────────────────┬─────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────┐
│      Rerank with CrossEncoder (top-k=12)               │
│   (ms-marco-MiniLM-L-6-v2)                             │
└──────────────────────┬─────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────┐
│         Format Context + Chat History                  │
└──────────────────────┬─────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────┐
│      Generate Answer (Llama 3.2, temp=0.2)             │
└──────────────────────┬─────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│    Evaluate with RAGAS + Update Chat History             │
│  Metrics: Faithfulness, Relevancy, Context Quality       │
└──────────────────────────────────────────────────────────┘
```

## 📋 Requirements

- **Python:** 3.8+
- **Ollama:** Local LLM server (download from [ollama.ai](https://ollama.ai))
- **Required Models:**
  - `llama3.2` - Main language model
  - `nomic-embed-text` - Embedding model

## 🚀 Installation

### 1. Set Up Ollama

```bash
# Download and install Ollama from https://ollama.ai
# Start the Ollama server
ollama serve
```

### 2. Pull Required Models

In a new terminal:

```bash
# Pull Llama 3.2
ollama pull llama3.2

# Pull embedding model
ollama pull nomic-embed-text
```

### 3. Clone and Set Up Project

```bash
cd ../RAGASSIGENMENT
```

### 4. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 5. Install Dependencies

```bash
cd backend
pip install -r req.txt
```

## 📦 Dependencies

Key packages (see [req.txt](req.txt)):

```
langchain>=0.1.0
langchain-ollama>=0.1.0
langchain-community>=0.1.0
faiss-cpu>=1.7.0
pymupdf>=1.23.0
streamlit>=1.28.0
ragas>=0.0.70
sentence-transformers>=2.2.0
datasets>=2.14.0
numpy>=1.24.0
```

## 💻 Usage

### Start the Application

```bash
cd backend
streamlit run main.py
```

The app opens at `http://localhost:8501`

### Workflow

1. **Upload PDFs:** Click "Upload PDFs" and select up to 5 PDF files
2. **Process PDFs:** Click "Process PDFs" button to create FAISS indexes
3. **Ask Questions:** Use the chat interface to ask about document contents
4. **View Results:** See answers with RAGAS evaluation scores in expandable section

### Example Queries

```
"Who are the authors of the paper Attention Is All You Need?"
"What is the Transformer architecture?"
"Explain multi-head attention"
"What modifications does RoBERTa make to BERT?"
```

## 📁 Project Structure

```
RAGASSIGENMENT/
├── README.md                          # This file
├── RAG_SYSTEM_REPORT.md              # Comprehensive evaluation report
│
├── backend/
│   ├── main.py                       # Streamlit UI
│   ├── rag_service.py                # Core RAG pipeline
│   ├── retriever.py                  # Hybrid retrieval + reranking
│   ├── rag_evaluator.py              # RAGAS evaluation framework
│   ├── vector_store.py               # PDF processing & FAISS indexing
│   ├── data.json                     # Ground truth Q&A pairs
│   ├── evaluation_results.json       # Test evaluation scores
│   ├── req.txt                       # Python dependencies
│   │
│   ├── faiss_indexes/                # Vector store indexes (auto-generated)
│   │   ├── Attention/
│   │   ├── BERT/
│   │   ├── Exploring/
│   │   ├── Language/
│   │   └── RoBERT/
│   │
│   ├── uploads/                      # Uploaded PDF files
│   └── __pycache__/                  # Python cache files
│
└── working/
    └── [Mirror of backend with additional files]
```

## ⚙️ Configuration

### Chat Memory Size

Modify in [rag_service.py](backend/rag_service.py):
```python
CHAT_MEMORY_SIZE = 5  # Number of previous Q&A pairs to retain
```

### Retrieval Parameters

Modify in [retriever.py](backend/retriever.py):
```python
# Vector search results
k_vector = 20

# BM25 search results
k_bm25 = 20

# Final reranked documents
top_k = 12
```

### LLM Temperature

Modify in [rag_service.py](backend/rag_service.py):
```python
llm = ChatOllama(
    model="llama3.2",
    temperature=0.2  # Lower = more deterministic, Higher = more creative
)
```

### Chunk Size & Overlap

Modify in [vector_store.py](backend/vector_store.py):
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Tokens per chunk
    chunk_overlap=200    # Token overlap between chunks
)
```

## 📊 Evaluation Metrics

The system uses **RAGAS** (Retrieval-Augmented Generation Assessment) with 5 metrics:

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| **Faithfulness** | 0-1 | Answer grounded in retrieved context |
| **Answer Relevancy** | 0-1 | Answer addresses the question |
| **Context Precision** | 0-1 | Retrieved docs contain relevant info |
| **Context Recall** | 0-1 | Retrieved docs cover ground truth |
| **Answer Correctness** | 0-1 | Answer matches expected response |

### Sample Results

```json
{
  "question": "What is multi-head attention and why is it used?",
  "answer": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
  "ragas_scores": {
    "faithfulness": 0.834,
    "answer_relevancy": 0.748,
    "context_precision": 0.834,
    "context_recall": 0.813,
    "answer_correctness": 0.880
  }
}
```

**Overall Performance:** Average correctness **0.757** across 10 benchmark queries

## 🔄 Data Flow

### 1. Document Ingestion
```
PDF Upload → PyMuPDFLoader → RecursiveCharacterTextSplitter 
→ Chunks (800 tokens, 200 overlap) → OllamaEmbeddings 
→ FAISS Index (local storage)
```

### 2. Query Processing
```
User Query → Hybrid Retrieval (Vector + BM25, k=20 each) 
→ Merge & Deduplicate → CrossEncoder Reranking (top-12) 
→ Format with Chat History → LLM Prompt
```

### 3. Response Generation
```
LLM Inference (Llama 3.2, temp=0.2) → Extract Answer Text 
→ RAGAS Evaluation → Store in Chat History (max 5 pairs) 
→ Return Answer + Scores
```

## 🛠️ Troubleshooting

### Issue: "Connection refused" for Ollama
**Solution:** Ensure Ollama server is running
```bash
ollama serve  # In a separate terminal
```

### Issue: FAISS index not found
**Solution:** Reprocess PDFs using "Process PDFs" button in UI

### Issue: Out of memory errors
**Solution:** Reduce `CHAT_MEMORY_SIZE` or `chunk_size` values

### Issue: Slow retrieval
**Solution:** Ensure FAISS indexes are already built; first run is slower during indexing

## 📈 Performance Optimization

- **Vector Search:** FAISS CPU-based (GPU support available with `faiss-gpu`)
- **Embeddings:** Cached locally after first computation
- **Reranking:** Efficient batch processing with CrossEncoder
- **LLM Inference:** Local Ollama avoids network latency

## 🧪 Testing

Run evaluation on benchmark queries:

```bash
cd backend
python -c "
from rag_service import ask_question
# Test with benchmark queries
result = ask_question('Who are the authors of Attention Is All You Need?')
print(result['ragas_scores'])
"
```

See [evaluation_results.json](evaluation_results.json) for detailed results.

## 📚 Key Files Reference

| File | Purpose |
|------|---------|
| [main.py](backend/main.py) | Streamlit UI and app orchestration |
| [rag_service.py](backend/rag_service.py) | Core RAG pipeline and LLM interaction |
| [retriever.py](backend/retriever.py) | Hybrid search and reranking logic |
| [rag_evaluator.py](backend/rag_evaluator.py) | RAGAS evaluation framework |
| [vector_store.py](backend/vector_store.py) | PDF processing and FAISS indexing |
| [data.json](backend/data.json) | Ground truth Q&A pairs |

## 🎓 Concepts

### Retrieval-Augmented Generation (RAG)
Combines document retrieval with LLM generation to produce factual, grounded answers by:
1. Retrieving relevant document chunks
2. Using them as context for LLM prompts
3. Generating answers constrained by retrieved content

### Hybrid Retrieval
- **Vector Search:** Captures semantic meaning (what the text means)
- **Keyword Search:** Captures specific terms (exact entity names, terminology)
- **Combined:** Reduces false negatives while maintaining precision

### Cross-Encoder Reranking
Uses a transformer model to rescore document relevance by considering:
- Query semantics
- Document content
- Semantic relationships
- Superior to embedding-only matching

## 📝 License

This project is provided as-is for educational and research purposes.

## 👤 Author

Shubham Garg  
Created: March 31, 2026

## 🤝 Contributing

To improve this system:
1. Test with additional PDFs and queries
2. Try different embedding models in `retriever.py`
3. Experiment with prompt templates in `rag_service.py`
4. Evaluate alternative reranker models
5. Submit findings and improvements

## 📞 Support

For issues or questions:
1. Check [RAG_SYSTEM_REPORT.md](RAG_SYSTEM_REPORT.md) for detailed analysis
2. Check Ollama status: `curl http://localhost:11434/api/tags`

## 🔗 References

- [LangChain Documentation](https://python.langchain.com/)
- [RAGAS Framework](https://github.com/explodinggradients/ragas)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Ollama Models](https://ollama.ai/library)
- [Streamlit Docs](https://docs.streamlit.io/)

---

**Version:** 1.0  
**Last Updated:** April 1, 2026  
**Status:** Production Ready
