from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Initialize the EXACT same embedding model used during creation
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. Load the FAISS database from your local folder
# Make sure the folder_path matches the folder containing your .faiss and .pkl files
print("Loading FAISS database...")
db = FAISS.load_local(
    folder_path="faiss_local_index", 
    embeddings=embeddings,
    allow_dangerous_deserialization=True  # Required by LangChain to load the .pkl file
)
print("Database loaded successfully!\n")

# 3. Define your test query
query = "What are the computational challenges of Seiberg Dualities?"

# 4. Perform the similarity search
# k=3 retrieves the top 3 most relevant chunks
print(f"Searching for: '{query}'\n")
results = db.similarity_search(query, k=3)

# 5. Display the results and their metadata
for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    # Print the first 1000 characters of the content so it doesn't flood your terminal
    print(f"Content: {doc.page_content[:1000]}...\n") 
    print("\n" + "="*50 + "\n")



    