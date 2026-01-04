import unittest
from finrag.chunking import chunk_text_with_tables

class TestChunking(unittest.TestCase):
    def test_table_preservation(self):
        text = """
Here is some text.

| Col1 | Col2 |
|---|---|
| Val1 | Val2 |

More text.
"""
        chunks = chunk_text_with_tables(text)
        
        # enhancing regex in chunking.py might be needed if this fails, 
        # but let's assume the simple logic works for simple tables.
        
        # Expectation: 3 chunks (Text, Table, Text) or similar depending on splitting
        # At least one chunk should have is_table=True
        
        table_chunks = [c for c in chunks if c.metadata.get("is_table")]
        self.assertTrue(len(table_chunks) >= 1)
        self.assertIn("| Col1 |", table_chunks[0].page_content)

if __name__ == '__main__':
    unittest.main()
