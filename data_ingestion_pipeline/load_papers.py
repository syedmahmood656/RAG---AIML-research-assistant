# load_papers.py
# Fetch + parse papers from arXiv.
#
# WHY THIS DOESN'T USE LangChain's ArxivLoader: its full-text mode runs
# query.replace(":", "") before searching, which turns "cat:cs.AI" into "catcs.AI" and silently
# destroys the category filter (0 results, no error). We call the `arxiv` package directly -
# the same package ArxivLoader wraps - and still return LangChain Documents.

import tempfile

import arxiv
import fitz  # pymupdf, for extracting text from downloaded PDFs
from langchain_core.documents import Document

from config import DOC_CONTENT_CHARS_MAX, PAPERS_PER_CATEGORY

# One shared client handles arXiv's rate limiting (3 s between requests) and retries.
client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=5)


def load_papers_for_category(category: str, seen_ids: set) -> list:
    """
    Fetch the newest papers of one category, download each PDF, extract its text and return
    LangChain Documents. Papers whose entry_id is already in `seen_ids` (cross-listed in an
    earlier category) are skipped, so the corpus never contains the same paper twice.
    """
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=PAPERS_PER_CATEGORY,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    docs = []
    for result in client.results(search):
        if result.entry_id in seen_ids:
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                pdf_path = result.download_pdf(dirpath=tmp)
                with fitz.open(pdf_path) as pdf:
                    text = "".join(page.get_text() for page in pdf)
        except Exception as e:  # noqa: BLE001
            print(f"  skipped one paper (download/parse failed): {e}")
            continue

        text = text.strip()
        if not text:  # scanned / image-only PDF
            print(f"  skipped '{result.title[:50]}' (no extractable text)")
            continue

        seen_ids.add(result.entry_id)
        docs.append(
            Document(
                page_content=text[:DOC_CONTENT_CHARS_MAX],
                metadata={
                    "Title": result.title,
                    "Authors": ", ".join(a.name for a in result.authors),
                    "Published": str(result.published.date()),
                    "category": category,
                    "entry_id": result.entry_id,
                    "pdf_url": result.pdf_url,
                },
            )
        )
    return docs


def load_all_categories(categories: list) -> list:
    """Loop over every category and return one flat, de-duplicated list of Documents."""
    all_docs, seen_ids = [], set()
    for category in categories:
        print(f"Loading full-text papers for category: {category}")
        docs = load_papers_for_category(category, seen_ids)
        print(f"  -> got {len(docs)} new papers")
        all_docs.extend(docs)
    return all_docs
