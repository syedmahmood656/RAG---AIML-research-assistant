from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os

# 1. Initialize the EXACT same embedding model used during creation
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. Load the FAISS database from your local folder
print("Loading FAISS database...")
db = FAISS.load_local(
    folder_path="faiss_local_index", 
    embeddings=embeddings,
    allow_dangerous_deserialization=True  # Required by LangChain to load the .pkl file
)
print("Database loaded successfully!\n")

# 3. Define your test query
query = "What are the computational challenges of Seiberg Dualities?"

#here we have used the most basic type of retriver where it uses vector store as a retriver and retrives "k" no of most simalar docs from the vector base
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4. Perform the similarity search
# k=3 retrieves the top 3 most relevant chunks
print(f"Searching for: '{query}'\n")
retrieved_docs = retriever.invoke(query)
# now this retrival will fetch 3 most similar chunks of data from the vector base

""" UN-COMMENT THIS SECTION IF YOU WANT TO SEE THE RETRIEVED RESULTS IN THE TERMINAL

# 5. Display the results
for i, doc in enumerate(retrieved_docs,1):
    print(f"--- Result {i} ---")
    # Print the first 1000 characters of the content so it doesn't flood your terminal
    print(f"Content: {doc.page_content[:1000]}...\n") 
    print("\n" + "="*50 + "\n")
"""

# making a prompt template
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context, this context is fetched form research papers.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

# invoking prompt
#final_prompt = prompt.invoke({"context": retrieved_docs, "question": query})



# setting up hugging face free chatmodel to use for text generation

# Set the token in environment variables specifically for the hub client
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'API KEY'  # Replace with your actual Hugging Face API token

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
    temperature=0.2
)

# Some wrappers need the token explicitly passed if environment sync is slow
model = ChatHuggingFace(llm=llm, huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'])


## building CHAINS

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(lambda docs: "\n".join([doc.page_content for doc in docs])),
    'question': RunnablePassthrough()
})

# parsing the output to string
parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

main_chain_output = main_chain.invoke(query)
print(main_chain_output)