# main.py
# Orchestrator: runs the full ingestion pipeline end to end.
#
#   1. Load    -> LangChain's ArxivLoader fetches + parses cs.AI / cs.LG
#                 papers into Document objects (replaces requests + XML)
#   2. Chunk   -> LangChain's RecursiveCharacterTextSplitter splits each
#                 abstract into RAG-ready chunks (replaces manual loop)
#   3. Save    -> write both raw papers and chunks to JSON files
#
# Run with: python main.py

from config import CATEGORIES, RAW_OUTPUT_PATH, CHUNKED_OUTPUT_PATH
from load_papers import load_all_categories
from chunk_papers import chunk_documents
from save_data import save_documents


def run_pipeline():
    print("Starting ArXiv ingestion pipeline (LangChain-powered)...\n")

    # Step 1: Fetch + parse papers for each category via ArxivLoader
    papers = load_all_categories(CATEGORIES)
    print(f"\nLoaded {len(papers)} papers total.")

    # Step 2: Split each paper's abstract into overlapping chunks
    chunks = chunk_documents(papers)
    print(f"Created {len(chunks)} text chunks via RecursiveCharacterTextSplitter.")

    # Step 3: Save both outputs to disk
    save_documents(papers, RAW_OUTPUT_PATH)
    save_documents(chunks, CHUNKED_OUTPUT_PATH)

    print("\nPipeline complete. Next step: embed chunks + load into a vector DB.")


if __name__ == "__main__":
    run_pipeline()