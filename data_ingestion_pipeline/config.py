# config.py
# All the "knobs" for the pipeline live here so you don't have to
# dig through code to change categories, result count, etc.

# ArXiv subject categories we care about (AI + ML)
CATEGORIES = ["cs.AI", "cs.LG"]  # cs.LG is the official code for cs.ML papers

# How many papers to pull per category, per run.
# ~75 x 2 categories = ~150 papers total (quick-demo sized corpus).
# NOTE: this maps to ArxivLoader's `top_k_results` param. Its
# `load_max_docs` param LOOKS like it should control this but is dead
# code in the current langchain-community version -> it's declared but
# never actually used anywhere in the fetch logic. Don't rely on it.
PAPERS_PER_CATEGORY = 75

# Full-text mode: we now download each paper's PDF and extract the full
# text (via PyMuPDF) instead of just the ~200-word abstract. This gives
# WAY more content per paper -> more, richer chunks for the vector DB.
# Caps each paper's extracted text at this many characters, so one huge
# 40-page paper can't bloat the dataset. ~20000 chars =~ 8-10 pages.
DOC_CONTENT_CHARS_MAX = 20000

# Text splitting settings for LangChain's RecursiveCharacterTextSplitter
CHUNK_SIZE = 1000     # bigger chunks now that we have full-text, not just abstracts
CHUNK_OVERLAP = 150   # overlap between chunks so context isn't lost at edges

# Where raw + processed data get saved
RAW_OUTPUT_PATH = "data/raw_papers.json"
CHUNKED_OUTPUT_PATH = "data/chunked_papers.json"

# NOTE: LangChain's ArxivLoader does not expose ArXiv's sortBy/sortOrder
# params (it always sorts by relevance under the hood). If "newest first"
# ordering matters for your demo, that's a known trade-off of using the
# LangChain abstraction vs. calling the raw API directly.
#
# NOTE: full-text mode downloads a PDF per paper and parses it, so this
# run will take noticeably longer than abstract-only mode (expect several
# minutes for ~150 papers, plus ArXiv's own rate limiting).