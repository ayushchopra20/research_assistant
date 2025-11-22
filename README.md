# Edge Computing Research Assistant (RAG)

This project implements a lightweight Retrieval-Augmented Generation (RAG) assistant over an edge computing knowledge corpus. It builds (or loads) a local FAISS vector index of the text and uses a locally hosted Large Language Model (LLM) via Ollama to answer questions grounded in retrieved context.

## Features
- Automatic index build if no existing FAISS index directory (`faiss_new_index/`).
- Deterministic retrieval with sentence-transformer embeddings (`all-MiniLM-L6-v2`).
- Local vector similarity search using FAISS (fast, in-memory, persistent to disk).
- Local LLM integration through Ollama (`llama3` model by default).
- Simple API (`answer_question`) for programmatic Q&A.

## Project Structure
```
project3/
  edge_computing.txt                # Raw corpus (original)
  edge_computing_cleaned.txt        # Cleaned corpus used for indexing
  research_assistant.py             # Core RAG logic (index + retrieval + LLM prompt)
  run_research_assistant.py         # Example script executing sample queries
  faiss_new_index/                  # Persisted FAISS index (created after first run)
  README.md                         # (This file)
```

## How It Works
1. On import/run, `run_research_assistant.py` checks if `faiss_new_index/` exists.
2. If absent: loads `edge_computing_cleaned.txt`, splits into overlapping chunks (size 500, overlap 50) using LangChain's `RecursiveCharacterTextSplitter`.
3. Generates embeddings for each chunk with HuggingFace sentence-transformers.
4. Builds and saves a FAISS index locally.
5. For a query, performs similarity search (top-k, default 5) and concatenates retrieved chunk text into a prompt.
6. Sends the augmented prompt to a local Ollama LLM (`llama3`) via `ChatOllama` and returns the answer.

## Dependencies
- `langchain-text-splitters`
- `langchain-community`
- `langchain-huggingface`
- `langchain-ollama`
- `sentence-transformers`
- `faiss-cpu` (for CPU environments)

You also need Ollama installed (see https://ollama.com/). After installation, pull the model:
```powershell
ollama pull llama3
```

Install Python libraries:
```powershell
pip install langchain-text-splitters langchain-community langchain-huggingface langchain-ollama sentence-transformers faiss-cpu
```
(Optional) create a `requirements.txt` for reproducibility:
```
langchain-text-splitters
langchain-community
langchain-huggingface
langchain-ollama
sentence-transformers
faiss-cpu
```

## Usage
Run the sample query script:
```powershell
python run_research_assistant.py
```
Or import in your own script:
```python
import research_assistant as RA
print(RA.answer_question("What is Edge Computing?"))
```
If you modify the corpus, delete or rename `faiss_new_index/` to force a rebuild.

## Configuration
Adjust in `research_assistant.py`:
- `chunk_size`, `chunk_overlap` in the `RecursiveCharacterTextSplitter`.
- `model_name` for embeddings (`all-MiniLM-L6-v2` can be swapped for larger models if resources allow).
- LLM model: change `ChatOllama(model="llama3")` to another local model tag.
- Retrieval depth: parameter `k` in `answer_question(question, k=5)`.

## Security & Safety Notes
- The load call uses `allow_dangerous_deserialization=True` to read FAISS index metadata. Only keep this if you fully trust the index directory contents.
- Large contexts may increase prompt size; consider truncation for very large corpora.

## Troubleshooting
- Index not building: ensure `edge_computing_cleaned.txt` exists and is readable.
- Embedding errors: confirm `sentence-transformers` installed and internet access for first model download (or pre-cache locally).
- Ollama errors: verify daemon running (`ollama list`).
- Slow responses: reduce `k` or use a smaller embedding/model.

## Extending
- Add metadata (e.g., source section) per chunk for more structured answers.
- Replace FAISS with another vector store (e.g., Chroma) if you need filtering.
- Add a CLI or simple FastAPI endpoint to expose the assistant.

## Acknowledgements
Built with LangChain, HuggingFace sentence-transformers, FAISS, and Ollama.

---
Feel free to request a refactor, CLI wrapper, or API service layer.
