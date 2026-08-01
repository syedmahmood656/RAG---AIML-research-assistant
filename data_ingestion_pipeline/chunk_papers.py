# chunk_papers.py
# Step 3, now using LangChain's RecursiveCharacterTextSplitter instead of
# a hand-rolled loop. It's smarter than plain character-slicing: it tries
# to split on paragraph/sentence boundaries first, and only falls back to
# hard character cuts if it has to -> better chunk quality for RAG.

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(docs: list) -> list:
    """
    Takes the list of LangChain Documents (one per paper) and splits
    each one's text into smaller overlapping Document chunks.
    Metadata (title, category, pdf_url, etc.) is automatically copied
    onto every chunk by LangChain -> no manual bookkeeping needed.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)