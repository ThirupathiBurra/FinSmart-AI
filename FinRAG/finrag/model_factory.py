import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from config import MODEL_NAME, EMBEDDING_MODEL

@st.cache_resource
def get_huggingface_embeddings():
    """
    Loads the embedding model (Cached).
    User for Step 3 (Embedding).
    """
    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def get_llm():
    """
    Loads the Finetuned Finance Model.
    Step 8 (Answer Generation).
    """
    print(f"Loading LLM {MODEL_NAME}...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device set to: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Force float32 for stability on Mac/MPS to prevent empty outputs
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32  
    ).to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=768,    # Optimized for speed (was 1024)
        do_sample=True,
        temperature=0.1,       # Low temp for factual consistency
        top_p=0.95,
        repetition_penalty=1.15,
        return_full_text=True
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
