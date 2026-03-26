# Medical RAG Assistant (Local LLM + Streamlit)

A lightweight **Retrieval-Augmented Generation (RAG)** system built with **Streamlit** and **local GGUF models**. This project enables intelligent question answering over custom documents using embeddings, vector search, and a local LLM pipeline.

---

## Features

-  Document-based Question Answering (RAG)
-  Local LLM support using GGUF models (no API dependency)
-  Custom document ingestion and chunking
-  Fast vector search using ChromaDB
-  Modular pipeline (Embedding → Retrieval → Generation)
-  Interactive UI with Streamlit
-  Offline-first architecture (privacy-friendly)

---

##  Project Architecture

```
User Query
    ↓
Retriever (Vector Search - ChromaDB)
    ↓
Relevant Chunks
    ↓
Prompt + Context
    ↓
Local LLM (GGUF Model)
    ↓
Generated Answer
```

---

## Project Structure

```.
├── chroma_db/                          # Vector database (Chroma)
├── data/docs/                          # Input documents
│ └── sample.txt
├── models/                             # Local GGUF models
│ └── openbiollm-llama3-8b.Q8_0.gguf
├── rag/
│ ├── embed.py                          # Embedding logic
│ ├── ingest.py                         # Document chunking
│ ├── pipeline.py                       # RAG pipeline
│ ├── retriever.py                      # Retrieval logic
│ └── vector_store.py                   # Vector DB creation/loading
├── utils/
│ └── loader.py                         # Document loader
├── app.py                              # Streamlit UI
├── config.py                           # Configuration settings
├── requirements.txt                    # Dependencies
└── README.md
```

---

## Tech Stack

- **Frontend**: Streamlit  
- **LLM**: GGUF Models (LLaMA-based, OpenBioLLM)  
- **Embeddings**: Local embedding models  
- **Vector DB**: ChromaDB  
- **Backend**: Python  

---

##  How It Works

1. **Load Documents** → From `data/docs/`
2. **Chunking** → Split into smaller text chunks (`ingest.py`)
3. **Embedding** → Convert chunks into vectors (`embed.py`)
4. **Store** → Save in ChromaDB (`vector_store.py`)
5. **Retrieve** → Fetch relevant chunks (`retriever.py`)
6. **Generate** → Pass context + query to LLM (`pipeline.py`)

---

## Configuration Setup

Before running the project, you need to create a configuration file.

### 1: Create `config.py`

In the root directory of the project, create a file named:

### 2: Add the following configuration

Copy and paste the following content into `config.py`:

```python
CHROMA_DIR = "./chroma_db"
MODEL_PATH = r"# Add your local model path here"   
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 2
```

Explanation

```
CHROMA_DIR → Directory where vector database will be stored
URL → Model download link (GGUF format)
MODEL_PATH → Path to your downloaded local model (required)
EMBED_MODEL → Embedding model used for vectorization
CHUNK_SIZE → Size of text chunks
CHUNK_OVERLAP → Overlap between chunks
TOP_K → Number of relevant chunks retrieved
```

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/sni-gdh/medical-rag-assistant.git
cd medical-rag-assistant
```

### 2. Create virtual environment

```bash
python -m venv rag_env
source rag_env/bin/activate   # Linux/Mac
rag_env\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Model Setup

```bash
mkdir models                  #create the model at your root directory
cd models
curl -L -o openbiollm-llama3-8b.Q4_K_M.gguf \
https://huggingface.co/aaditya/OpenBioLLM-Llama3-8B-GGUF/resolve/main/openbiollm-llama3-8b.Q8_0.gguf
```

### 5. Run the Application

```bash
streamlit run app.py
```

---

## Usage

- Add your documents inside:

```bash
data/docs/
```

- Run the app
- Ask questions in the UI
- Get contextual answers from your data

---

## Example Use Cases

- Biomedical document QA (OpenBioLLM)
- Research paper assistant
- Internal knowledge base chatbot
- Private offline AI assistant
