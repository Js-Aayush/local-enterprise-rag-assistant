# Local Enterprise RAG Document Assistant

A local Retrieval-Augmented Generation (RAG) application for querying technical and industrial PDF documents using semantic search and a locally hosted Large Language Model.

The system retrieves relevant information from multiple documents and provides grounded answers with source file and page references.

## Project Overview

Traditional LLMs answer based mainly on knowledge learned during training and may not have access to private, domain-specific, or recently updated documents.

This project implements a RAG pipeline that allows users to ask questions about a custom collection of PDF documents.

The system:

- Extracts text from multiple PDF documents
- Splits documents into overlapping chunks
- Converts chunks into semantic embeddings
- Stores embeddings in a persistent ChromaDB vector database
- Converts user queries into embeddings
- Retrieves the most relevant document chunks
- Filters weak retrieval results using distance thresholds
- Sends retrieved context and the user question to a local LLM through Ollama
- Generates grounded answers
- Returns source document and page references

## Architecture

```text
PDF Documents
      ↓
Text Extraction
      ↓
Chunking
      ↓
Sentence Transformer Embeddings
      ↓
Persistent ChromaDB
      ↓
      ↓
User Question
      ↓
Query Embedding
      ↓
Vector Similarity Search
      ↓
Top-K Retrieval
      ↓
Distance Filtering
      ↓
Relevant Document Context
      ↓
Local LLM using Ollama
      ↓
Grounded Answer
      +
Source PDF / Page
```

## Technologies Used

- Python
- FastAPI
- ChromaDB
- Sentence Transformers
- Ollama
- Qwen3 4B
- PyPDF
- NumPy
- Scikit-learn
- Uvicorn
- Jupyter Notebook

## Embedding Model

The project uses:

```text
all-MiniLM-L6-v2
```

Each document chunk is converted into a 384-dimensional semantic embedding.

These embeddings allow the system to retrieve document sections based on meaning rather than only exact keyword matches.

## Local LLM

The generation component uses:

```text
qwen3:4b
```

running locally through Ollama.

This allows the RAG system to operate without depending on paid cloud LLM APIs.

## Document Processing

PDF documents are processed page-by-page.

Each page is divided into overlapping text chunks.

Example configuration:

```text
Chunk size: 1000 characters
Chunk overlap: 200 characters
```

Metadata is stored with every chunk:

```text
Source PDF
Page number
Original text
Embedding vector
```

This allows the final system to provide source references with generated answers.

## Retrieval Process

When a user asks a question:

1. The question is converted into an embedding.
2. ChromaDB searches for the closest document embeddings.
3. The top relevant chunks are retrieved.
4. Weak matches are removed using a distance threshold.
5. Retrieved text is combined into context.
6. The context and question are sent to the local LLM.
7. The LLM generates a concise answer using the retrieved information.

## RAG Evaluation

A small manually created evaluation set was used to verify document retrieval.

For each test question, the expected source document was compared with the documents returned in the top-k retrieval results.

Current result:

```text
Retrieval Hit Rate: 100%
```

This result applies only to the current small manually created evaluation set and should not be interpreted as overall model accuracy.

## API

The application exposes a FastAPI endpoint:

```text
POST /ask
```

Example request:

```json
{
  "question": "What is lockout tagout used for?"
}
```

Example response:

```json
{
  "answer": "Lockout/tagout is used to control hazardous energy and prevent unexpected machine startup during servicing or maintenance.",
  "sources": [
    {
      "file": "lockout_tagout.pdf",
      "page": 8
    }
  ]
}
```

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

Install Ollama locally and download the required model:

```bash
ollama run qwen3:4b
```

### 3. Add PDF documents

Place your PDF files inside:

```text
documents/
```

### 4. Run the notebook

Run:

```text
rag.ipynb
```

to extract document text, create chunks, generate embeddings, and populate ChromaDB.

### 5. Start the FastAPI application

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Use the `/ask` endpoint to test the RAG system.

## Project Structure

```text
local-enterprise-rag-assistant/
│
├── app.py
├── rag.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── documents/
│   ├── ppe_manual.pdf
│   ├── machine_guarding.pdf
│   └── lockout_tagout.pdf
│
└── chroma_db/
```

The `documents/` and `chroma_db/` folders can be excluded from Git because the database can be regenerated from the source documents.

## Key Concepts Demonstrated

- Retrieval-Augmented Generation
- Semantic embeddings
- Vector similarity search
- Document chunking
- Top-k retrieval
- Retrieval filtering
- Metadata-based source tracking
- Local LLM inference
- Prompt grounding
- Hallucination reduction
- RAG evaluation
- REST API development
- Local AI deployment

## Future Improvements

Possible future improvements include:

- Token-aware or semantic chunking
- Reranking retrieved documents
- Conversation history
- Metadata filtering
- Additional RAG evaluation metrics
- Web interface
- Docker deployment
- Cloud deployment
- Support for additional document formats

## Resume Summary

**Local Enterprise RAG Document Assistant | Python, ChromaDB, Sentence Transformers, Ollama, FastAPI**

Built a local Retrieval-Augmented Generation system for querying multiple technical PDF documents using semantic embeddings and vector similarity search. Implemented document chunking, persistent vector storage, top-k retrieval, distance filtering, source/page tracking, local LLM inference, and a FastAPI-based question-answering API.