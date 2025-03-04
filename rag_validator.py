import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Base directory
persist_dir = "./chroma_db"

# Custom prompt template for RAG
custom_prompt_template = """
You are an expert assistant with deep knowledge about the content in the provided context.
Answer the question based only on the provided context.

Context:
{context}

Question: {question}

Your answer (be specific and cite sources when possible):
"""


# Function to create a RAG chain for a specific collection
def create_rag_chain(collection_name):
    # Initialize the collection
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    # Set up the retriever
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Create the prompt
    PROMPT = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    # Create the chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain


# Function to query across all collections and combine results
def query_all_collections(query):
    # Get list of collections - updated for Chroma v0.6.0
    import chromadb
    client = chromadb.PersistentClient(path=persist_dir)
    collection_names = client.list_collections()

    print(f"Found collections: {collection_names}")

    all_responses = {}

    # Query each collection
    for collection_name in collection_names:
        print(f"\nQuerying '{collection_name}' for: {query}")
        chain = create_rag_chain(collection_name)

        # Use .invoke() instead of calling the chain directly
        response = chain.invoke({"query": query})

        all_responses[collection_name] = {
            "answer": response["result"],
            "sources": [doc.metadata.get('source', 'Unknown') for doc in response["source_documents"]]
        }

        print(f"Answer from {collection_name}:")
        print(response["result"])
        print("Sources:", [doc.metadata.get('source', 'Unknown') for doc in response["source_documents"]])

    return all_responses


# Demo usage
if __name__ == "__main__":
    # Example questions
    questions = [
        "What is ArgoCD and how is it used?",
        "What qualities make an effective leader?",
        "How can I implement continuous deployment?"
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)
        results = query_all_collections(question)