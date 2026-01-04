import os
from typing import List, Optional, Dict
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from config import ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN, COLLECTION_NAME
from finrag.model_factory import get_huggingface_embeddings

class FinRAGVectorStore:
    def __init__(self):
        # IDENTICAL to Notebook: Use standard embedding model
        self.embeddings = get_huggingface_embeddings()
        
        # Initialize AstraDB (Cloud version of FAISS/Chroma from notebook)
        self.vectorstore = AstraDBVectorStore(
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
        )

    def add_documents(self, documents: List[Document]):
        """
        Standard add_documents, identical to notebook's behavior.
        """
        if not documents:
            return
        self.vectorstore.add_documents(documents)
        print(f"Stored {len(documents)} vectors in AstraDB.")

    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict] = None) -> List[Document]:
        """
        Standard similarity search with strict metadata filtering.
        """
        return self.vectorstore.similarity_search(query, k=k, filter=filter)
