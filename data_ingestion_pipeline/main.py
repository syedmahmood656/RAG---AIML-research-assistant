# main.py
# Orchestrator: runs the full ingestion pipeline end to end.
#
#   1. Load    -> `arxiv` package + PyMuPDF fetch cs.AI / cs.LG papers (full text,
#                 de-duplicated across categories) as LangChain Documents
#   2. Chunk   -> RecursiveCharacterTextSplitter splits each paper into chunks
#   3. Save    -> write raw papers and chunks to data/*.json
#
# Run from the project root:  python data_ingestion_pipeline/main.py

from config import CATEGORIES, RAW_OUTPUT_PATH, CHUNKED_OUTPUT_PATH
from load_papers import load_all_categories
from chunk_papers import chunk_documents
from save_data import save_documents


def run_pipeline():
    print("Starting ArXiv ingestion pipeline...\n")

    # Step 1: Fetch + parse papers for each category
    papers = load_all_categories(CATEGORIES)
    print(f"\nLoaded {len(papers)} papers total.")

    # Step 2: Split each paper's full text into overlapping chunks
    chunks = chunk_documents(papers)
    print(f"Created {len(chunks)} text chunks via RecursiveCharacterTextSplitter.")

    # Step 3: Save both outputs to disk
    save_documents(papers, RAW_OUTPUT_PATH)
    save_documents(chunks, CHUNKED_OUTPUT_PATH)

    print("\nPipeline complete. Next: python load_and_vectorbase/load.py")


if __name__ == "__main__":
    run_pipeline()