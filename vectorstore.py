# vectorstore.py
import os
import asyncio
import nest_asyncio
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GEMINI_API_KEY

# Enable nested event loops for Streamlit compatibility
nest_asyncio.apply()

def create_vectorstore(chunks, save_path=None, debug=False):
    """
    Create FAISS vector store from document chunks
    
    Args:
        chunks: List of document chunks
        save_path: Optional path to save the vector store
        debug: Print debug information about vectors
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY,
        )
        
        if debug:
            print(f"📊 Creating vectors for {len(chunks)} chunks...")
            print(f"📄 Sample chunk: {chunks[0].page_content[:100]}..." if chunks else "No chunks")
        
        # Create vector store with proper error handling
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        
        if debug:
            print(f"✅ Vector store created with {vectorstore.index.ntotal} vectors")
        
        if save_path:
            save_vectorstore(vectorstore, save_path)
            if debug:
                print(f"💾 Vector store saved to: {save_path}")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
        raise e

def save_vectorstore(vectorstore, save_path):
    """Save vector store to disk"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vectorstore.save_local(save_path)
    except Exception as e:
        print(f"❌ Error saving vector store: {str(e)}")
        raise e

def load_vectorstore(load_path, embeddings=None):
    """Load vector store from disk"""
    try:
        if embeddings is None:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GEMINI_API_KEY,
            )
        
        return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
        
    except Exception as e:
        print(f"❌ Error loading vector store: {str(e)}")
        raise e

def vectorstore_exists(path):
    """Check if vector store exists at path"""
    return os.path.exists(f"{path}/index.faiss") and os.path.exists(f"{path}/index.pkl")
