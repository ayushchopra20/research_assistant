from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
import os

# Initialize embeddings model (needed for either loading or building index)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_dir = "faiss_new_index"
if os.path.isdir(index_dir):
    print(f"FAISS index directory '{index_dir}' exists. Loading existing index; skipping text processing.")
    faiss_index = FAISS.load_local(
        index_dir,
        embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    print(f"FAISS index directory '{index_dir}' not found. Building new index...")
    # Load and split only when index absent
    file_path = "edge_computing_cleaned.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- Chunk {i+1} ---")
        print(chunk)
        print()

    faiss_index = FAISS.from_texts(chunks, embedding_model)
    faiss_index.save_local(index_dir)
    print(f"FAISS index saved to '{index_dir}' directory.")

llm = ChatOllama(model="llama3")   # your local Llama-3 instance

def answer_question(question, k=5):
    docs = faiss_index.similarity_search(question, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = (
        "Use the following context to answer concisely:\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    return response.content