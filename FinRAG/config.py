import os
from dotenv import load_dotenv

load_dotenv()

# Environment Variables
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN") # Optional if using public models or already logged in

# FinRAG Settings
HIERARCHY_LAYERS = ["chunk", "document", "month", "entity"]
DEFAULT_SCOPE = "document"

# Model Settings
MODEL_NAME = "Shiva-k22/gemma-FinAI"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Summarization Settings
MAX_SUMMARY_TOKENS = 300
SUMMARY_BATCH_SIZE = 5

# Vector Store Settings
COLLECTION_NAME = "finrag_hierarchy_collection"
