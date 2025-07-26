# loader.py
import hashlib
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf_docs(file_path, chunk_size=1000, chunk_overlap=200, debug=False):
    """
    Load and process PDF documents into chunks
    
    Args:
        file_path: Path to the PDF file
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        debug: Print debug information
    """
    if debug:
        print(f"📖 Loading PDF: {file_path}")
        print(f"📏 File size: {os.path.getsize(file_path)} bytes")
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    if debug:
        print(f"📄 Loaded {len(docs)} pages")
        total_chars = sum(len(doc.page_content) for doc in docs)
        print(f"📝 Total characters: {total_chars}")
    
    # Add file metadata to documents
    file_hash = get_file_hash(file_path)
    filename = os.path.basename(file_path)
    
    for doc in docs:
        doc.metadata.update({
            'source_file': filename,
            'file_hash': file_hash,
            'file_path': file_path
        })
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    
    if debug:
        print(f"✂️  Created {len(chunks)} chunks")
        avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
        print(f"📊 Average chunk size: {avg_chunk_size:.0f} characters")
    
    return chunks

def get_file_hash(file_path):
    """Generate hash for file to track changes"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]
