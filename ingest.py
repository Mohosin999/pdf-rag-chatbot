import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

import config

load_dotenv()

# ============================================
# Load PDF
# ============================================
pdf_files = [f for f in os.listdir(config.DATA_DIR) if f.endswith(".pdf")]

all_documents = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(config.DATA_DIR, pdf_file)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    all_documents.extend(documents)
    print(f"Loaded PDF: {pdf_file} ({len(documents)} page)")

# ============================================
# Text Split (Chunking)
# ============================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(all_documents)
print(f"Total chunks: {len(chunks)}")

# ============================================
# Create embeddings and vectorstore
# ============================================
embeddings = GoogleGenerativeAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=config.CHROMA_DIR
)

print("Database successfully created!")
