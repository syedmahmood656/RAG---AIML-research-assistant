# config.py
# All the "knobs" for the pipeline live here.

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # project root

# ArXiv subject categories we care about (AI + ML)
CATEGORIES = ["cs.AI", "cs.LG"]  # cs.LG is the official code for cs.ML papers

# Papers pulled per category per run. Papers cross-listed in both categories are
# de-duplicated by arXiv id, so the final corpus is a bit under 2 x this number.
PAPERS_PER_CATEGORY = 75

# Full-text mode: each paper's PDF is downloaded and its text extracted with PyMuPDF.
# Each paper's text is capped at this many characters (~8-10 pages) to keep the corpus small.
DOC_CONTENT_CHARS_MAX = 20000

# RecursiveCharacterTextSplitter settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Absolute paths, so the pipeline behaves the same from any working directory.
RAW_OUTPUT_PATH = str(_ROOT / "data" / "raw_papers.json")
CHUNKED_OUTPUT_PATH = str(_ROOT / "data" / "chunked_papers.json")
