# save_data.py
# Step 4: persist data to disk as JSON so later pipeline stages
# (embedding, vector DB loading) can pick it up.

import json
import os


def save_json(data, path: str):
    """Saves any Python list/dict to a JSON file, creating folders if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved {len(data)} records -> {path}")


def save_documents(docs: list, path: str):
    """
    LangChain Document objects aren't plain dicts, so we convert each one
    to {page_content, metadata} first, then reuse save_json() above.
    """
    plain_records = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    save_json(plain_records, path)