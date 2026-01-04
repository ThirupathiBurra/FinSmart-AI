import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from finrag.astradb_vectorstore import FinRAGVectorStore

class TestHierarchy(unittest.TestCase):
    
    @patch('finrag.astradb_vectorstore.AstraDBVectorStore')
    @patch('finrag.astradb_vectorstore.get_huggingface_embeddings')
    def test_upsert_hierarchy(self, mock_embed, mock_astra):
        # Setup
        vs = FinRAGVectorStore()
        
        docs = [
            Document(page_content="chunk1", metadata={"layer": "chunk", "user_id": "u1"}),
            Document(page_content="summary1", metadata={"layer": "document", "user_id": "u1"})
        ]
        
        # Action
        vs.upsert_hierarchy(docs)
        
        # Assert
        vs.vectorstore.add_documents.assert_called_once_with(docs)

    @patch('finrag.astradb_vectorstore.AstraDBVectorStore')
    @patch('finrag.astradb_vectorstore.get_huggingface_embeddings')
    def test_query_hierarchy(self, mock_embed, mock_astra):
        # Setup
        vs = FinRAGVectorStore()
        vs.vectorstore.similarity_search.return_value = [Document(page_content="res")]
        
        k_per_layer = {"document": 2, "chunk": 3}
        user_id = "test_user"
        
        # Action
        results = vs.query_hierarchy("query", user_id, k_per_layer)
        
        # Assert
        self.assertIn("document", results)
        self.assertIn("chunk", results)
        self.assertEqual(vs.vectorstore.similarity_search.call_count, 2)
        
        # Verify filtering
        # Check call args
        calls = vs.vectorstore.similarity_search.call_args_list
        # First call (order depends on dict iteration, usually insertion order in recent python)
        # We just check that filter was passed correctly in one of the calls
        filters_used = [c[1]['filter'] for c in calls]
        self.assertIn({"user_id": "test_user", "layer": "document"}, filters_used)
        self.assertIn({"user_id": "test_user", "layer": "chunk"}, filters_used)

if __name__ == '__main__':
    unittest.main()
