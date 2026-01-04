import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import streamlit as st
from config import MODEL_NAME

@st.cache_resource
def get_llm():
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print(f"Loading model on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,
        return_full_text=False, # Prevent prompting echoing
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def get_huggingface_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    from config import EMBEDDING_MODEL
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
