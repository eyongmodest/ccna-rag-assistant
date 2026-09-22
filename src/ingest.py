"""
ingest.py
---------
Loads all PDF files from data/raw, splits them into chunks, converts each
chunk into an embedding (a numeric representation of its meaning), and
stores those embeddings in a local ChromaDB vector database.

Run this once whenever you add new PDFs to data/raw/.

Usage:
    python src/ingest.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Step 1: Load environment variables ---
load_dotenv()
# Note: no OpenAI key check here anymore — embeddings now run locally
# and free, using a Hugging Face model instead of OpenAI's API.

# --- Step 2: Configuration ---
RAW_DATA_DIR = "data/raw"
VECTORDB_DIR = "data/vectordb"
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 150     # characters shared between consecutive chunks

# --- Step 3: Load all PDFs from data/raw ---
print(f"Loading PDFs from {RAW_DATA_DIR}...")
loader = PyPDFDirectoryLoader(RAW_DATA_DIR)
documents = loader.load()
print(f"Loaded {len(documents)} pages total from your PDF files.")

# --- Step 4: Split documents into chunks ---
# Why we chunk: LLMs can only process a limited amount of text at once,
# and smaller, focused chunks let the retrieval step pull back exactly
# the relevant passage instead of an entire chapter.
# CHUNK_OVERLAP exists so that a sentence or config example split across
# two chunks isn't cut off with no context on either side.
print(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

# --- Step 5: Embed chunks and store them in ChromaDB ---
# Each chunk of text gets converted into a vector (list of numbers)
# using a free, open-source embedding model that runs locally on your
# CPU. The first run will download the model (~80MB) automatically;
# after that it's cached and works fully offline.
print("Loading local embedding model (first run downloads ~80MB)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Embedding chunks and storing in ChromaDB (running locally, no API calls)...")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTORDB_DIR,
)

print(f"Done. Vector database saved to {VECTORDB_DIR}/")
print(f"Total chunks stored: {len(chunks)}")
