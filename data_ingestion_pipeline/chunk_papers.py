# chunk_papers.py
# Split each paper into overlapping chunks. RecursiveCharacterTextSplitter prefers paragraph /
# sentence boundaries and only hard-cuts when it has to. Paper metadata (Title, Authors,
# entry_id, pdf_url, ...) is copied onto every chunk, and `start_index` records the chunk's
# character offset inside its paper.

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(docs)
