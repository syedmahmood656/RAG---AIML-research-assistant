# config.py
# All the "knobs" for the pipeline live here so you don't have to
# dig through code to change categories, result count, etc.

# ArXiv subject categories we care about (AI + ML)
CATEGORIES = ["cs.AI", "cs.LG"]  # cs.LG is the official code for cs.ML papers

# How many papers to pull per category, per run
# (passed to LangChain's ArxivLoader as load_max_docs)
MAX_RESULTS_PER_CATEGORY = 20

# Text splitting settings for LangChain's RecursiveCharacterTextSplitter
CHUNK_SIZE = 300      # characters per chunk
CHUNK_OVERLAP = 50    # overlap between chunks so context isn't lost at edges

# Where raw + processed data get saved
RAW_OUTPUT_PATH = "data/raw_papers.json"
CHUNKED_OUTPUT_PATH = "data/chunked_papers.json"

# NOTE: LangChain's ArxivLoader does not expose ArXiv's sortBy/sortOrder
# params (it always sorts by relevance under the hood). If "newest first"
# ordering matters for your demo, that's a known trade-off of using the
# LangChain abstraction vs. calling the raw API directly.