from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

print("Starting RAG Customer Support Assistant...")

knowledge_base = """
Return Policy: Customers can return products within 7 days.
Refund Policy: Refund processed within 5 business days.
Shipping Charges: Free shipping above Rs.999.
Order Cancellation: Orders can be cancelled before dispatch.
Damaged Product: Replacement within 3 days.
Customer Support: Complex issues go to human support agent.
"""

# Step 1: Chunking
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.create_documents([knowledge_base])

# Step 2: Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 3: Chroma Vector DB
db = Chroma.from_documents(docs, embedding)
retriever = db.as_retriever(search_kwargs={"k": 2})

# Step 4: LLM
groq_api = "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"

llm = ChatGroq(
    groq_api_key=groq_api,
    model_name="llama-3.1-8b-instant"
)

print("RAG Bot Ready! Type exit to stop.\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    if "angry" in query.lower() or "complaint" in query.lower():
        print("Bot: Escalated to Human Support Agent\n")
        continue

    retrieved_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f'''
Use only this support information:

{context}

Answer customer question:
{query}
'''

    response = llm.invoke(prompt)
    print("Bot:", response.content, "\n")