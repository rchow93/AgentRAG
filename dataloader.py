import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Base directories and embeddings
docs_dir = "./docs"
persist_dir = "./chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# File type configurations
FILE_LOADERS = {
    "pdf": PyPDFLoader,
    "epub": UnstructuredEPubLoader
}

# Process each subdirectory
for subdir in os.listdir(docs_dir):
    subdir_path = os.path.join(docs_dir, subdir)

    # Skip if not a directory
    if not os.path.isdir(subdir_path):
        continue

    print(f"Processing collection: {subdir}")
    all_docs = []

    # Process all supported files
    for file in os.listdir(subdir_path):
        file_ext = file.split('.')[-1].lower()

        if file_ext in FILE_LOADERS:
            file_path = os.path.join(subdir_path, file)
            print(f"  Loading {file_ext.upper()}: {file}")

            # Load document with appropriate loader
            loader = FILE_LOADERS[file_ext](file_path=file_path)
            docs = loader.load()

            # Split text based on file type (larger chunks for EPUB)
            chunk_size = 1500 if file_ext == "epub" else 1000
            chunk_overlap = 250 if file_ext == "epub" else 200

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            split_docs = splitter.split_documents(docs)
            all_docs.extend(split_docs)

    # Create collection if documents were found
    if all_docs:
        Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=subdir
        )
        print(f"  Created collection '{subdir}' with {len(all_docs)} chunks")
    else:
        print(f"  No supported files found in {subdir}")

print("Processing complete.")