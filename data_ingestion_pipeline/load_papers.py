# load_papers.py
# Step 1+2 combined (fetch + parse).
#
# WHY THIS ISN'T USING ArxivLoader FOR FETCHING:
# LangChain's ArxivLoader.load() (full-text mode) has a real bug: before
# searching, it runs query.replace(":", "").replace("-", "") on the query
# string. For a category-filtered query like "cat:cs.AI" this mangles it
# into "catcs.AI", silently destroying the category filter -> 0 results,
# no error. There's no config flag to turn this off.
#
# So we call the `arxiv` package directly instead -> it's the exact same
# package ArxivLoader wraps internally, just without the buggy string
# mangling. We still use LangChain's Document schema below, and
# everything downstream (chunk_papers.py, save_data.py) is unchanged.
#
# Bonus: this also lets us sort by newest-first, which ArxivLoader never
# exposed in the first place.

import os
import arxiv
import fitz  # pymupdf, for extracting text from downloaded PDFs
from langchain_core.documents import Document
from config import PAPERS_PER_CATEGORY, DOC_CONTENT_CHARS_MAX

# One shared client handles ArXiv's rate limiting for us automatically
client = arxiv.Client()


def load_papers_for_category(category: str) -> list:
    """
    Fetches full-text papers for one ArXiv category, downloads each PDF,
    extracts its text with PyMuPDF, and returns a list of LangChain
    Document objects (page_content = full paper text).
    """
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=PAPERS_PER_CATEGORY,
        sort_by=arxiv.SortCriterion.SubmittedDate,  # newest papers first
        sort_order=arxiv.SortOrder.Descending,
    )

    docs = []
    for result in client.results(search):
        try:
            pdf_path = result.download_pdf()  # downloads to a temp file
            with fitz.open(pdf_path) as pdf:
                text = "".join(page.get_text() for page in pdf)
            os.remove(pdf_path)  # clean up after extracting text
        except Exception as e:
            # Some PDFs occasionally fail to download/parse -> skip and
            # keep going instead of crashing the whole run.
            print(f"  skipped one paper (download/parse failed): {e}")
            continue

        docs.append(Document(
            page_content=text[:DOC_CONTENT_CHARS_MAX],  # cap length per paper
            metadata={
                "Title": result.title,
                "Authors": ", ".join(a.name for a in result.authors),
                "Published": str(result.published.date()),
                "category": category,
                "entry_id": result.entry_id,
                "pdf_url": result.pdf_url,
            },
        ))

    return docs


def load_all_categories(categories: list) -> list:
    """
    Loops over every category we care about and returns one combined,
    flat list of LangChain Document objects across all categories.
    """
    all_docs = []
    for category in categories:
        print(f"Loading full-text papers for category: {category}")
        docs = load_papers_for_category(category)
        print(f"  -> got {len(docs)} papers")
        all_docs.extend(docs)
    return all_docs