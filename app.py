# import streamlit as st
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_chroma import Chroma
# from langchain_classic.chains import RetrievalQA
# from langchain_classic.retrievers import ContextualCompressionRetriever
# from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain_community.retrievers import BM25Retriever
# from langchain_classic.retrievers import EnsembleRetriever
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
# import os

# load_dotenv()

# st.title("PDF Question Answering System")
# st.write("Ask any question from your document.")

# # ============================================
# # 1. Load vector database
# # ============================================
# @st.cache_resource
# def load_vectorstore():
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001",
#         google_api_key=os.getenv("GOOGLE_API_KEY")
#     )
#     return Chroma(
#         persist_directory="./chroma_db",
#         embedding_function=embeddings
#     )

# vectorstore = load_vectorstore()

# # ============================================
# # 2. Rebuild chunks for BM25 retriever
# # ============================================
# @st.cache_resource
# def load_chunks():
#     data_dir = "data"
#     pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
#     all_docs = []
#     for pdf_file in pdf_files:
#         loader = PyPDFLoader(os.path.join(data_dir, pdf_file))
#         all_docs.extend(loader.load())

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000, chunk_overlap=200
#     )
#     return splitter.split_documents(all_docs)

# chunks = load_chunks()

# # ============================================
# # 3. Hybrid retriever (vector + BM25)
# # ============================================
# @st.cache_resource
# def load_ensemble_retriever(_chunks):
#     vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
#     bm25_retriever = BM25Retriever.from_documents(_chunks)
#     bm25_retriever.k = 4

#     return EnsembleRetriever(
#         retrievers=[vector_retriever, bm25_retriever],
#         weights=[0.6, 0.4]
#     )

# ensemble_retriever = load_ensemble_retriever(chunks)

# # ============================================
# # 4. Reranker on top of hybrid retriever
# # ============================================
# @st.cache_resource
# def load_reranker():
#     cross_encoder = HuggingFaceCrossEncoder(
#         model_name="BAAI/bge-reranker-base"
#     )
#     return CrossEncoderReranker(model=cross_encoder, top_n=4)

# reranker = load_reranker()

# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=reranker,
#     base_retriever=ensemble_retriever
# )

# # ============================================
# # 5. LLM and QA chain
# # ============================================
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.1-flash-lite",
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0.3
# )

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=compression_retriever,
#     return_source_documents=True
# )

# # ============================================
# # 6. Question interface
# # ============================================
# query = st.text_input("Write your question:")

# if query:
#     with st.spinner("Generating answer..."):
#         result = qa_chain.invoke({"query": query})

#     st.subheader("Answer")
#     st.write(result["result"])

#     with st.expander("Source documents"):
#         for doc in result["source_documents"]:
#             page = doc.metadata.get("page", "?")
#             st.write(f"Page {page}: {doc.page_content[:200]}...")

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import os

load_dotenv()

st.title("PDF Question Answering System")
st.write("Ask any question from your document.")

# ============================================
# 1. Load vector database
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
# 2. Rebuild chunks for BM25 retriever
# ============================================
@st.cache_resource
def load_chunks():
    data_dir = "data"
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    all_docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(data_dir, pdf_file))
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    return splitter.split_documents(all_docs)

chunks = load_chunks()

# ============================================
# 3. Hybrid retriever (vector + BM25)
# ============================================
@st.cache_resource
def load_ensemble_retriever(_chunks):
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    bm25_retriever = BM25Retriever.from_documents(_chunks)
    bm25_retriever.k = 4

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

ensemble_retriever = load_ensemble_retriever(chunks)

# ============================================
# 4. Reranker on top of hybrid retriever
# ============================================
@st.cache_resource
def load_reranker():
    cross_encoder = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base"
    )
    return CrossEncoderReranker(model=cross_encoder, top_n=4)

reranker = load_reranker()

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever
)

# ============================================
# 5. LLM and QA chain
# ============================================
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# Memory - last 5 messages
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5,
    output_key="answer"
)

# Conversational Retrieval
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=compression_retriever,
    memory=memory,
    return_source_documents=True
)

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
        st.write(result["answer"])
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})