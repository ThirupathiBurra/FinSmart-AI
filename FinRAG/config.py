import os
from dotenv import load_dotenv

load_dotenv()

# Environment Variables
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# Model Settings
MODEL_NAME = "Shiva-k22/gemma-FinAI"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ingestion Settings - Aggressive Optimization for Performance
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Collection Name
COLLECTION_NAME = "finrag_clean_v1"
