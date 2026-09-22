"""
query.py
--------
Asks a question against your CCNA vector database. Retrieves the most
relevant chunks from ChromaDB, then sends them along with your question
to a free Groq-hosted LLM to generate a grounded answer -- along with
the source PDF(s) and page number(s) the answer was drawn from.

Usage:
    python src/query.py
    (then type your question when prompted)
"""

import os
import sys
import io

# --- Force UTF-8 output, robustly ---
# Windows terminals often default to an older encoding (cp1252) that
# can't display certain characters (smart quotes, special dashes, etc.)
# that LLMs commonly produce. Wrapping stdout like this guarantees
# UTF-8 output regardless of the terminal's default settings.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- Step 1: Load environment variables ---
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Make sure your .env file exists "
        "in the project root and contains GROQ_API_KEY=your-key-here"
    )

# --- Step 2: Configuration ---
VECTORDB_DIR = "data/vectordb"
NUM_CHUNKS_TO_RETRIEVE = 4  # how many relevant chunks to pull per question

# --- Step 3: Load the same embedding model used during ingestion ---
# This MUST match ingest.py's model, otherwise question and document
# vectors won't be comparable to each other.
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Step 4: Connect to the existing ChromaDB vector database ---
vectordb = Chroma(
    persist_directory=VECTORDB_DIR,
    embedding_function=embeddings,
)
retriever = vectordb.as_retriever(search_kwargs={"k": NUM_CHUNKS_TO_RETRIEVE})

# --- Step 5: Set up the Groq LLM ---
# openai/gpt-oss-20b is the model available on this account's Groq
# key -- fast, general-purpose, and free-tier friendly.
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

# --- Step 6: Build the prompt template ---
# temperature=0.2 keeps answers focused and factual rather than
# creative, which suits studying from reference material.
prompt = ChatPromptTemplate.from_template(
    """You are a helpful CCNA study assistant. Answer the question using
ONLY the context below. If the answer isn't in the context, say you
don't have enough information in the provided material, rather than
guessing.

Context:
{context}

Question:
{question}

Answer:"""
)


def format_sources(retrieved_docs):
    """
    Build a clean, de-duplicated list of "filename - page X" strings
    from the retrieved chunks' metadata, so the user can see exactly
    which PDFs (and which pages) an answer was drawn from.
    """
    seen = set()
    sources = []
    for doc in retrieved_docs:
        # PyPDFDirectoryLoader stores the full file path in "source"
        # and the zero-indexed page number in "page". We just want
        # the filename (not the full path) and a human page number.
        full_path = doc.metadata.get("source", "Unknown file")
        filename = os.path.basename(full_path)
        page_number = doc.metadata.get("page")
        # Pages are 0-indexed internally; add 1 so it matches what
        # you'd actually see printed on the PDF page itself.
        page_display = f"page {page_number + 1}" if page_number is not None else "page unknown"

        entry = f"{filename} ({page_display})"
        if entry not in seen:
            seen.add(entry)
            sources.append(entry)
    return sources


# --- Step 7: Simple question loop ---
print("\nCCNA Study Assistant ready. Type your question, or 'exit' to quit.\n")

while True:
    question = input("Your question: ").strip()
    if question.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    if not question:
        continue

    # Retrieve the most relevant chunks for this question
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Build the final prompt and get the answer
    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)

    # Replace any character that still can't be displayed, instead of
    # crashing -- a last-resort safety net on top of the UTF-8 wrapping
    # above.
    answer_text = response.content.encode("utf-8", errors="replace").decode("utf-8")

    print(f"\nAnswer: {answer_text}\n")

    sources = format_sources(retrieved_docs)
    if sources:
        print("Sources:")
        for src in sources:
            print(f"  - {src}")
    print()
    print("-" * 60)
