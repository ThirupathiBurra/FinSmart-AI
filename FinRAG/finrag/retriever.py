from typing import List, Dict, Optional
from langchain_core.documents import Document
from finrag.astradb_vectorstore import FinRAGVectorStore

class FinRAGRetriever:
    def __init__(self, vectorstore: FinRAGVectorStore):
        self.vectorstore = vectorstore

    def detect_intent(self, question: str) -> str:
        """
        Step 4: Query Understanding.
        Analyzes intent to drive retrieval strategy.
        """
        question = question.lower()
        if any(x in question for x in ["summarize", "overview", "brief", "report"]):
            return "SUMMARY"
        return "SPECIFIC"

    def retrieve(self, query: str, user_id: str, session_id: str) -> List[Document]:
        """
        Step 5: Context Retrieval.
        Adapts depth based on intent.
        Strictly filters by user identity and session.
        """
        intent = self.detect_intent(query)
        print(f"Retrieval Request | Query: '{query}' | Intent: {intent} | User: {user_id}")
        
        # Adaptive Retrieval Depth
        # Summary: Needs broad context (but kept to 4 for speed optimization)
        # Specific: Needs focused context
        if intent == "SUMMARY":
            k = 4
            score_threshold = 0.35 
        else:
            k = 4
            score_threshold = 0.35
        
        # Metadata Filters (Step 3 Requirement)
        filter_dict = {
            "user_id": user_id,
            "session_id": session_id
        }
        
        # Step 6: Context Validation
        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k, filter=filter_dict)
        
        validated_docs = []
        rejected_count = 0
        
        for doc, score in results_with_scores:
            if score >= score_threshold:
                # Inject relevance score for transparency
                doc.metadata["relevance_score"] = score 
                validated_docs.append(doc)
            else:
                rejected_count += 1
                
        if rejected_count > 0:
             print(f"Validation: Rejected {rejected_count} chunks below threshold {score_threshold}")
        
        return validated_docs
