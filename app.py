import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

# Page title
st.title("QA system from PDF")
st.write("Ask any question from your document")

# ============================================
# ১. Load vector database
# ============================================
@st.cache_resource
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

vectorstore = load_vectorstore()

# ============================================
# ২. LLM and retrieval chain
# ============================================
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

# ============================================
# ৩. Question interface
# ============================================
query = st.text_input("Write your question: ")

if query:
    with st.spinner("Generating answer..."):
        result = qa_chain.invoke({"query": query})
    
    st.subheader("Answer")
    st.write(result["result"])
    
    with st.expander("Comes from which documents?"):
        for doc in result["source_documents"]:
            page = doc.metadata.get("page", "?")
            st.write(f"Page {page}: {doc.page_content[:200]}...")