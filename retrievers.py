import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def get_llm():
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=config.TEMPERATURE,
    )


def load_vectorstore():
    return Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=get_embeddings(),
    )


def load_chunks():
    all_docs = []
    for pdf_file in os.listdir(config.DATA_DIR):
        if pdf_file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(config.DATA_DIR, pdf_file))
            all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(all_docs)


def build_ensemble_retriever(chunks, vectorstore):
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.RETRIEVAL_K}
    )
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = config.RETRIEVAL_K

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[config.VECTOR_WEIGHT, config.BM25_WEIGHT],
    )


def build_reranker():
    cross_encoder = HuggingFaceCrossEncoder(model_name=config.RERANK_MODEL)
    return CrossEncoderReranker(model=cross_encoder, top_n=config.RERANK_TOP_N)


def build_compression_retriever(ensemble_retriever, reranker):
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever,
    )


def build_multi_query_retriever(compression_retriever, llm):
    return MultiQueryRetriever.from_llm(
        retriever=compression_retriever,
        llm=llm,
    )
