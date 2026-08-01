# load_papers.py
# Step 1+2 combined (fetch + parse), now using LangChain's built-in
# ArxivLoader instead of hand-rolled requests + XML parsing.
#
# Why this is the "LangChain part": ArxivLoader wraps the ArXiv API call
# AND turns each result into a LangChain Document (page_content + metadata)
# for us -> exactly what our old fetch_arxiv.py + parse_papers.py did
# manually, in one built-in class.

from langchain_community.document_loaders import ArxivLoader
from data_ingestion_pipeline.config import MAX_RESULTS_PER_CATEGORY

def load_papers_for_category(category: str) -> list:
    """
    Uses LangChain's ArxivLoader to fetch + parse papers for one category.
    Returns a list of LangChain Document objects (page_content = abstract).
    """
    query = f"cat:{category}"  # same ArXiv query syntax as the raw API

    loader = ArxivLoader(query=query, load_max_docs=MAX_RESULTS_PER_CATEGORY)

    # get_summaries_as_docs() is a lightweight fetch: it grabs abstracts
    # only (not full PDFs), matching what our original pipeline did.
    docs = loader.get_summaries_as_docs()

    # LangChain's metadata doesn't include our category or a direct PDF
    # link, so we add those ourselves -> keeps output shape consistent
    # with the earlier version of this pipeline.
    for doc in docs:
        doc.metadata["category"] = category
        entry_id = doc.metadata.get("Entry ID", "")
        doc.metadata["pdf_url"] = entry_id.replace("/abs/", "/pdf/")

    return docs


def load_all_categories(categories: list) -> list:
    """
    Loops over every category we care about and returns one combined,
    flat list of LangChain Document objects across all categories.
    """
    all_docs = []
    for category in categories:
        print(f"Loading latest papers for category: {category} (via LangChain ArxivLoader)")
        all_docs.extend(load_papers_for_category(category))
    return all_docs


# checking that the loader works as expected

result = load_papers_for_category("cs.CL")
print(f"Loaded {len(result)} papers for category cs.CL")
print(result[0].metadata)
print(result[0].page_content[:200])  # print first 200 chars of abstract