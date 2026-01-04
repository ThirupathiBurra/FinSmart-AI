import os
import time
from langchain_core.documents import Document
from finrag.astradb_vectorstore import FinRAGVectorStore

def test_db():
    print("--- 1. Initializing AstraDB Connection ---")
    try:
        vs = FinRAGVectorStore()
        print("✅ Connection Initialized.")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    print("\n--- 2. Performing Test Upsert ---")
    test_doc_content = "This is a FinRAG database verification test artifact. 12345."
    test_source = "TEST_VERIFICATION_SCRIPT"
    user_id = "test_user_inspector"
    
    doc = Document(
        page_content=test_doc_content,
        metadata={
            "source": test_source,
            "user_id": user_id,
            "page": "1"
        }
    )
    
    try:
        vs.add_documents([doc])
        print("✅ Upsert Successful.")
    except Exception as e:
        print(f"❌ Upsert Failed: {e}")
        return
        
    print("\n--- 3. Waiting for consistency (2s) ---")
    time.sleep(2)
    
    print("\n--- 4. Performing Verification Query ---")
    query = "FinRAG verification 12345"
    try:
        results = vs.similarity_search(query, k=1, filter={"user_id": user_id})
        
        if not results:
             print("❌ Query returned NO results. Vector search might be broken.")
        else:
            top_match = results[0]
            print(f"Found Doc: {top_match.page_content}")
            if top_match.metadata.get("source") == test_source:
                print("✅ Data Integrity Verified: Retrieved correct test document.")
            else:
                print("⚠️ Warning: Retrieved document does not match test source.")
                
    except Exception as e:
        print(f"❌ Query Failed: {e}")

if __name__ == "__main__":
    test_db()
