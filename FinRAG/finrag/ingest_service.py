import os
import tempfile
from typing import IO
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finrag.astradb_vectorstore import FinRAGVectorStore

def ingest_file(uploaded_file: IO, filename: str, user_id: str):
    """
    Ingests a file EXACTLY like the FinRAG Reference Notebook:
    1. Load
    2. Split (RecursiveCharacterTextSplitter)
    3. Metadata tagging
    4. Store
    """
    print(f"Starting ingestion for {filename} (User: {user_id})")
    
    # 1. Load File
    suffix = ".pdf" if filename.endswith(".pdf") else ".txt" 
    # Handle others for completeness if useful, but focusing on PDF
    if filename.endswith(".csv"): suffix = ".csv"
    if filename.endswith(".xlsx"): suffix = ".xlsx"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
        
    try:
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif filename.lower().endswith(".csv"):
            loader = CSVLoader(tmp_path)
            docs = loader.load()
        elif filename.lower().endswith((".xls", ".xlsx")):
            loader = UnstructuredExcelLoader(tmp_path)
            docs = loader.load()
        else:
            loader = TextLoader(tmp_path)
            docs = loader.load()
    finally:
        os.remove(tmp_path)
        
    # 2. Chunking (Identical to Notebook)
    # chunk_size=1000, chunk_overlap=100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)
    
    # 3. Metadata (Identical to Notebook)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = filename
        chunk.metadata["user_id"] = user_id # Critical for AstraDB filtering
        
        # Ensure page exists
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = "Unknown"
            
    # 4. Storage
    vectorstore = FinRAGVectorStore()
    vectorstore.add_documents(chunks)
    
    print(f"Ingestion complete. Added {len(chunks)} chunks.")
    return len(chunks)
