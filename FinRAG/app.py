import streamlit as st
import json
from finrag.astradb_vectorstore import FinRAGVectorStore
from finrag.retriever import FinRAGRetriever
from finrag.model_factory import get_llm
from finrag.ingest_service import ingest_file
from finrag.prompt_templates import FINRAG_PROMPT

# Setup
st.set_page_config(page_title="FinRAG Analyst", page_icon="📈", layout="wide")
st.title("📈 FinRAG: Financial Analyst System")

# Initialize components (Simplified)
@st.cache_resource
def get_system():
    vs = FinRAGVectorStore()
    retriever = FinRAGRetriever(vs)
    llm = get_llm()
    return retriever, llm

retriever_obj, llm = get_system()

# Sidebar
with st.sidebar:
    st.header("1. Document Ingestion")
    USER_ID = st.text_input("User ID", value="user_123_demo")
    
    uploaded_file = st.file_uploader("Upload Financial Doc", type=["pdf", "csv", "xlsx", "txt"])
    if uploaded_file and st.button("Ingest Document"):
        with st.spinner("Processing document..."):
            try:
                count = ingest_file(uploaded_file, uploaded_file.name, USER_ID)
                st.success(f"Ingested {count} chunks.")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.markdown("---")
    st.caption("FinRAG v2.0 | Production")

# Main Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello. I am the FinRAG Analyst. Upload your documents, then ask me strict financial questions."}]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            # 1. Retrieve (Identical to Notebook)
            retrieved_docs = retriever_obj.retrieve(prompt, USER_ID)
            
            # 2. Context Construction
            context_text = ""
            for d in retrieved_docs:
                meta = d.metadata
                source_str = f"[Source: {meta.get('source', 'Unknown')} | Page: {meta.get('page', 'N/A')}]"
                context_text += f"{source_str}\n{d.page_content}\n\n"
            
            if not context_text.strip():
                msg = "No relevant financial data found in the knowledge base."
                st.info(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            else:
                # 3. Generate (Using ChatPromptTemplate)
                try:
                    # FINRAG_PROMPT expects {"context": ..., "question": ...}
                    chain = FINRAG_PROMPT | llm
                    response = chain.invoke({"context": context_text, "question": prompt})
                    
                    # Handle response types (String or Object)
                    final_answer = response if isinstance(response, str) else response.content
                    
                    # Clean up any residual markdown block syntax if model adds it
                    final_answer = final_answer.replace("```markdown", "").replace("```", "").strip()

                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                        
                except Exception as e:
                    st.error(f"System Error: {e}")
