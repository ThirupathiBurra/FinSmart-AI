from typing import List
from langchain_core.documents import Document
from finrag.astradb_vectorstore import FinRAGVectorStore

# IDENTICAL Logic to Reference Notebook
# No complex classes, just functional retrieval with intent detection.

class FinRAGRetriever:
    def __init__(self, vectorstore: FinRAGVectorStore):
        self.vectorstore = vectorstore

    def detect_intent(self, question: str) -> str:
        """
        Simple intent detection from Notebook.
        """
        question = question.lower()
        if any(x in question for x in ["summarize", "overview", "brief", "report"]):
            return "SUMMARY"
        return "SPECIFIC"

    def retrieve(self, query: str, user_id: str) -> List[Document]:
        """
        Retrieves documents based on intent.
        """
        intent = self.detect_intent(query)
        print(f"Query: {query} | Intent: {intent}")
        
        # Alignment with Notebook Logic:
        # Summary -> k=5
        # Specific -> k=3
        k = 5 if intent == "SUMMARY" else 3
        
        # Strict user_id filtering
        filter_dict = {"user_id": user_id}
        
        docs = self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        return docs
