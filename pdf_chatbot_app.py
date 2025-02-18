import os
import time
import shutil
import streamlit as st
import warnings
from typing import List
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA

# Suppress warnings
warnings.filterwarnings('ignore')

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBaFm3JJIsbyMl6JLA2Y2mUUuPMwxQc4cM"

def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file and return its documents."""
    try:
        return PyPDFLoader(file_path).load()
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

def split_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 80) -> List[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def create_vector_store(documents: List[Document]):
    """Create and return a fresh Chroma vector store."""
    try:
        persist_directory = "./chroma_db"
        
        # Ensure ChromaDB is closed before deleting
        if os.path.exists(persist_directory):
            try:
                vector_store = Chroma(persist_directory=persist_directory)
                vector_store.delete_collection()  # Remove previous data
                del vector_store  # Ensure it is deleted properly
                time.sleep(2)  # Give Windows time to release file locks
            except Exception as e:
                st.warning(f"Warning while closing ChromaDB: {e}")
            
            shutil.rmtree(persist_directory, ignore_errors=True)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def create_qa_chain(vector_store):
    """Create and return a question-answering chain."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

# Streamlit App
st.title("Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    pdf_path = os.path.join("./", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("PDF uploaded successfully!")
    
    # Load and process the PDF
    documents = load_pdf(pdf_path)
    if documents:
        split_docs = split_documents(documents)
        vector_store = create_vector_store(split_docs)
        
        if vector_store:
            qa_chain = create_qa_chain(vector_store)
            
            if qa_chain:
                st.subheader("Ask a question about the document")
                user_query = st.text_input("Your question:")
                
                if user_query:
                    try:
                        response = qa_chain.invoke({"query": user_query})
                        st.write("**Answer:**", response['result'])
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
