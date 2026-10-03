import streamlit as st
from dotenv import load_dotenv

import retrievers
from chains import build_qa_chain
from utils import format_answer_with_sources

load_dotenv()

st.title("PDF Question Answering System")
st.write("Ask any question from your document.")


@st.cache_resource
def get_vectorstore():
    return retrievers.load_vectorstore()


@st.cache_resource
def get_chunks():
    return retrievers.load_chunks()


@st.cache_resource
def get_ensemble_retriever(_chunks):
    vectorstore = get_vectorstore()
    return retrievers.build_ensemble_retriever(_chunks, vectorstore)


@st.cache_resource
def get_reranker():
    return retrievers.build_reranker()


vectorstore = get_vectorstore()
chunks = get_chunks()
ensemble_retriever = get_ensemble_retriever(chunks)
reranker = get_reranker()
compression_retriever = retrievers.build_compression_retriever(
    ensemble_retriever, reranker
)

llm = retrievers.get_llm()

qa_chain = build_qa_chain(llm, compression_retriever)

# ============================================
# Question interface
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous message
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# New question
if prompt := st.chat_input("Write your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"question": prompt})
        full_answer = format_answer_with_sources(result)
        st.write(full_answer)
        with st.expander("Source documents"):
            for doc in result["source_documents"]:
                page = doc.metadata.get("page", "?")
                st.write(f"Page {page}: {doc.page_content[:300]}...")
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
