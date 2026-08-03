from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Initialize the local embedding model
# 'all-MiniLM-L6-v2' is widely used because it is fast, lightweight, and highly capable
# It will download the model weights to your machine the first time you run this
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}, # Change to 'cuda' if you have a GPU
    encode_kwargs={'normalize_embeddings': True}
)

# 2. Define the JSON Loader
loader = JSONLoader(
    file_path="./data/chunked_papers.json",
    jq_schema='.[]',
    content_key='page_content'
)

# Generator helper function to chunk the lazy loader into batches
def batch_generator(lazy_iterable, batch_size=500):
    batch = []
    for item in lazy_iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# 3. Stream & Embed in Batches
db = None
BATCH_SIZE = 500  # Adjust based on your system's memory capacity

print("Starting batch embedding...")
# Using loader.lazy_load() instead of loader.load()
for i, batch in enumerate(batch_generator(loader.lazy_load(), batch_size=BATCH_SIZE)):
    print(f"Processing batch {i + 1} ({len(batch)} chunks)...")
    
    if db is None:
        # Initialize FAISS with the first batch
        db = FAISS.from_documents(batch, embeddings)
    else:
        # Incrementally add subsequent batches to the same FAISS index
        db.add_documents(batch)

# 4. Save the index to disk
db.save_local("faiss_local_index")
print("All 35,000 chunks successfully embedded and saved!")
