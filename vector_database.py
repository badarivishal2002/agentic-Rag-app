# vector_database.py
import os
import json
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import asyncio
import nest_asyncio
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from config import GEMINI_API_KEY

# Enable nested event loops for Streamlit compatibility
nest_asyncio.apply()

class EnhancedVectorDatabase:
    """Enhanced vector database with persistent storage and metadata management"""
    
    def __init__(self, db_dir: str = "vector_database"):
        self.db_dir = db_dir
        self.metadata_file = os.path.join(db_dir, "metadata.json")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY,
        )
        
        # Ensure database directory exists
        os.makedirs(db_dir, exist_ok=True)
        
        # Initialize metadata file
        self._init_metadata()
        
        # Active vector stores
        self.vector_stores = {}
        self.combined_store = None
    
    def _init_metadata(self):
        """Initialize metadata file"""
        if not os.path.exists(self.metadata_file):
            metadata = {
                "created_at": datetime.now().isoformat(),
                "documents": {},
                "collections": {},
                "stats": {
                    "total_documents": 0,
                    "total_vectors": 0,
                    "last_updated": None
                }
            }
            self._save_metadata(metadata)
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file"""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            self._init_metadata()
            return self._load_metadata()
    
    def _save_metadata(self, metadata: Dict):
        """Save metadata to file"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def add_document(self, file_path: str, chunks: List[Document], 
                    file_hash: str, metadata: Dict = None) -> str:
        """Add a document to the vector database"""
        try:
            # Create collection ID
            filename = os.path.basename(file_path)
            collection_id = f"{file_hash}_{filename.replace(' ', '_')}"
            collection_path = os.path.join(self.db_dir, collection_id)
            
            # Create vector store for this document
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            
            # Save vector store
            vector_store.save_local(collection_path)
            
            # Update metadata
            db_metadata = self._load_metadata()
            
            doc_metadata = {
                "filename": filename,
                "file_path": file_path,
                "file_hash": file_hash,
                "collection_id": collection_id,
                "collection_path": collection_path,
                "added_at": datetime.now().isoformat(),
                "chunk_count": len(chunks),
                "vector_count": vector_store.index.ntotal,
                "vector_dimension": vector_store.index.d,
                "metadata": metadata or {},
                "access_count": 0,
                "last_accessed": None
            }
            
            db_metadata["documents"][file_hash] = doc_metadata
            db_metadata["collections"][collection_id] = {
                "file_hash": file_hash,
                "path": collection_path,
                "created_at": datetime.now().isoformat()
            }
            
            # Update stats
            db_metadata["stats"]["total_documents"] += 1
            db_metadata["stats"]["total_vectors"] += vector_store.index.ntotal
            db_metadata["stats"]["last_updated"] = datetime.now().isoformat()
            
            self._save_metadata(db_metadata)
            
            # Store in memory
            self.vector_stores[collection_id] = vector_store
            
            return collection_id
            
        except Exception as e:
            print(f"❌ Error adding document to vector database: {str(e)}")
            raise e
    
    def load_document(self, file_hash: str) -> Optional[FAISS]:
        """Load a document's vector store"""
        try:
            metadata = self._load_metadata()
            
            if file_hash not in metadata["documents"]:
                return None
            
            doc_info = metadata["documents"][file_hash]
            collection_id = doc_info["collection_id"]
            collection_path = doc_info["collection_path"]
            
            # Check if already loaded
            if collection_id in self.vector_stores:
                return self.vector_stores[collection_id]
            
            # Load from disk
            if os.path.exists(f"{collection_path}/index.faiss"):
                vector_store = FAISS.load_local(
                    collection_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                self.vector_stores[collection_id] = vector_store
                
                # Update access stats
                self._update_access_stats(file_hash)
                
                return vector_store
            
            return None
            
        except Exception as e:
            print(f"❌ Error loading document from vector database: {str(e)}")
            return None
    
    def _update_access_stats(self, file_hash: str):
        """Update document access statistics"""
        metadata = self._load_metadata()
        
        if file_hash in metadata["documents"]:
            metadata["documents"][file_hash]["access_count"] += 1
            metadata["documents"][file_hash]["last_accessed"] = datetime.now().isoformat()
            self._save_metadata(metadata)
    
    def search_document(self, file_hash: str, query: str, k: int = 4) -> List[Document]:
        """Search within a specific document"""
        vector_store = self.load_document(file_hash)
        
        if vector_store is None:
            return []
        
        try:
            return vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"❌ Error searching document: {str(e)}")
            return []
    
    def search_all_documents(self, query: str, k: int = 4) -> List[Tuple[Document, str]]:
        """Search across all documents"""
        results = []
        metadata = self._load_metadata()
        
        for file_hash, doc_info in metadata["documents"].items():
            docs = self.search_document(file_hash, query, k=2)  # Get fewer from each doc
            for doc in docs:
                # Add source information
                doc.metadata["source_document"] = doc_info["filename"]
                doc.metadata["file_hash"] = file_hash
                results.append((doc, doc_info["filename"]))
        
        # Sort by relevance (this is a simple approach, could be improved)
        return results[:k]
    
    def create_combined_store(self, selected_hashes: List[str] = None) -> Optional[FAISS]:
        """Create a combined vector store from multiple documents"""
        try:
            metadata = self._load_metadata()
            documents_to_combine = selected_hashes or list(metadata["documents"].keys())
            
            if not documents_to_combine:
                return None
            
            # Load first document
            first_hash = documents_to_combine[0]
            combined_store = self.load_document(first_hash)
            
            if combined_store is None:
                return None
            
            # Merge other documents
            for file_hash in documents_to_combine[1:]:
                vector_store = self.load_document(file_hash)
                if vector_store:
                    combined_store.merge_from(vector_store)
            
            self.combined_store = combined_store
            return combined_store
            
        except Exception as e:
            print(f"❌ Error creating combined vector store: {str(e)}")
            return None
    
    def get_document_info(self, file_hash: str) -> Optional[Dict]:
        """Get document information"""
        metadata = self._load_metadata()
        return metadata["documents"].get(file_hash)
    
    def get_all_documents(self) -> Dict:
        """Get all documents in the database"""
        metadata = self._load_metadata()
        return metadata["documents"]
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        metadata = self._load_metadata()
        return metadata["stats"]
    
    def delete_document(self, file_hash: str) -> bool:
        """Delete a document from the database"""
        try:
            metadata = self._load_metadata()
            
            if file_hash not in metadata["documents"]:
                return False
            
            doc_info = metadata["documents"][file_hash]
            collection_id = doc_info["collection_id"]
            collection_path = doc_info["collection_path"]
            
            # Remove from disk
            if os.path.exists(collection_path):
                shutil.rmtree(collection_path)
            
            # Remove from memory
            if collection_id in self.vector_stores:
                del self.vector_stores[collection_id]
            
            # Update metadata
            vector_count = doc_info.get("vector_count", 0)
            
            del metadata["documents"][file_hash]
            del metadata["collections"][collection_id]
            
            metadata["stats"]["total_documents"] -= 1
            metadata["stats"]["total_vectors"] -= vector_count
            metadata["stats"]["last_updated"] = datetime.now().isoformat()
            
            self._save_metadata(metadata)
            
            return True
            
        except Exception as e:
            print(f"❌ Error deleting document: {str(e)}")
            return False
    
    def export_database(self) -> Dict:
        """Export database metadata"""
        metadata = self._load_metadata()
        return {
            "exported_at": datetime.now().isoformat(),
            "metadata": metadata,
            "active_collections": list(self.vector_stores.keys())
        }
    
    def cleanup_orphaned_collections(self):
        """Clean up orphaned collection files"""
        metadata = self._load_metadata()
        valid_collections = set(metadata["collections"].keys())
        
        # Check all directories in db_dir
        for item in os.listdir(self.db_dir):
            item_path = os.path.join(self.db_dir, item)
            if os.path.isdir(item_path) and item not in valid_collections and item != "metadata.json":
                print(f"🧹 Cleaning up orphaned collection: {item}")
                shutil.rmtree(item_path)
    
    def get_similar_documents(self, file_hash: str, top_k: int = 3) -> List[Dict]:
        """Find documents similar to the given document based on content"""
        # This is a placeholder for document similarity
        # In a real implementation, you'd compare document embeddings
        metadata = self._load_metadata()
        
        if file_hash not in metadata["documents"]:
            return []
        
        current_doc = metadata["documents"][file_hash]
        similar_docs = []
        
        for hash_key, doc_info in metadata["documents"].items():
            if hash_key != file_hash:
                # Simple similarity based on access patterns and metadata
                similarity_score = 0.0
                
                # Similar access counts
                if doc_info["access_count"] > 0 and current_doc["access_count"] > 0:
                    similarity_score += 0.3
                
                # Similar file sizes (vector counts)
                size_diff = abs(doc_info["vector_count"] - current_doc["vector_count"])
                if size_diff < 100:
                    similarity_score += 0.4
                
                # Similar upload times
                if abs(datetime.fromisoformat(doc_info["added_at"]).timestamp() - 
                       datetime.fromisoformat(current_doc["added_at"]).timestamp()) < 86400:
                    similarity_score += 0.3
                
                if similarity_score > 0.5:
                    similar_docs.append({
                        "file_hash": hash_key,
                        "filename": doc_info["filename"],
                        "similarity_score": similarity_score
                    })
        
        return sorted(similar_docs, key=lambda x: x["similarity_score"], reverse=True)[:top_k] 