# memory.py
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib

class ConversationMemory:
    """Manages conversation history and memory"""
    
    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = memory_dir
        self.conversation_file = os.path.join(memory_dir, "conversations.json")
        self.documents_file = os.path.join(memory_dir, "documents.json")
        self.analytics_file = os.path.join(memory_dir, "analytics.json")
        
        # Ensure memory directory exists
        os.makedirs(memory_dir, exist_ok=True)
        
        # Initialize memory files
        self._init_memory_files()
    
    def _init_memory_files(self):
        """Initialize memory files if they don't exist"""
        if not os.path.exists(self.conversation_file):
            self._save_json(self.conversation_file, {"sessions": {}})
        
        if not os.path.exists(self.documents_file):
            self._save_json(self.documents_file, {"documents": {}})
            
        if not os.path.exists(self.analytics_file):
            self._save_json(self.analytics_file, {
                "total_queries": 0,
                "total_documents": 0,
                "query_history": [],
                "popular_queries": {},
                "document_usage": {}
            })
    
    def _load_json(self, filepath: str) -> Dict:
        """Load JSON data from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def _save_json(self, filepath: str, data: Dict):
        """Save JSON data to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_session_id(self, user_identifier: str = "default") -> str:
        """Generate or get session ID"""
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"{user_identifier}_{timestamp}"
    
    def add_conversation(self, session_id: str, question: str, answer: str, 
                        document_name: str, metadata: Dict = None):
        """Add a conversation to memory"""
        conversations = self._load_json(self.conversation_file)
        
        if session_id not in conversations["sessions"]:
            conversations["sessions"][session_id] = {
                "created_at": datetime.now().isoformat(),
                "conversations": []
            }
        
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "document": document_name,
            "metadata": metadata or {}
        }
        
        conversations["sessions"][session_id]["conversations"].append(conversation_entry)
        self._save_json(self.conversation_file, conversations)
        
        # Update analytics
        self._update_analytics(question, document_name)
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for a session"""
        conversations = self._load_json(self.conversation_file)
        
        if session_id not in conversations["sessions"]:
            return []
        
        history = conversations["sessions"][session_id]["conversations"]
        return history[-limit:] if limit else history
    
    def add_document(self, file_path: str, file_hash: str, metadata: Dict = None):
        """Add document to memory"""
        documents = self._load_json(self.documents_file)
        
        filename = os.path.basename(file_path)
        
        documents["documents"][file_hash] = {
            "filename": filename,
            "file_path": file_path,
            "uploaded_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": None,
            "metadata": metadata or {}
        }
        
        self._save_json(self.documents_file, documents)
    
    def get_document_info(self, file_hash: str) -> Optional[Dict]:
        """Get document information"""
        documents = self._load_json(self.documents_file)
        return documents["documents"].get(file_hash)
    
    def update_document_access(self, file_hash: str):
        """Update document access statistics"""
        documents = self._load_json(self.documents_file)
        
        if file_hash in documents["documents"]:
            documents["documents"][file_hash]["access_count"] += 1
            documents["documents"][file_hash]["last_accessed"] = datetime.now().isoformat()
            self._save_json(self.documents_file, documents)
    
    def get_all_documents(self) -> Dict:
        """Get all documents in memory"""
        documents = self._load_json(self.documents_file)
        return documents["documents"]
    
    def _update_analytics(self, question: str, document_name: str):
        """Update analytics data"""
        analytics = self._load_json(self.analytics_file)
        
        # Update counters
        analytics["total_queries"] += 1
        
        # Add to query history
        query_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "document": document_name
        }
        analytics["query_history"].append(query_entry)
        
        # Keep only last 100 queries
        if len(analytics["query_history"]) > 100:
            analytics["query_history"] = analytics["query_history"][-100:]
        
        # Update popular queries
        question_lower = question.lower()
        if question_lower in analytics["popular_queries"]:
            analytics["popular_queries"][question_lower] += 1
        else:
            analytics["popular_queries"][question_lower] = 1
        
        # Update document usage
        if document_name in analytics["document_usage"]:
            analytics["document_usage"][document_name] += 1
        else:
            analytics["document_usage"][document_name] = 1
        
        self._save_json(self.analytics_file, analytics)
    
    def get_analytics(self) -> Dict:
        """Get analytics data"""
        return self._load_json(self.analytics_file)
    
    def get_conversation_context(self, session_id: str, current_question: str) -> str:
        """Get conversation context for better answers"""
        history = self.get_conversation_history(session_id, limit=3)
        
        if not history:
            return current_question
        
        context_parts = ["Previous conversation context:"]
        for conv in history[-2:]:  # Last 2 conversations
            context_parts.append(f"Q: {conv['question']}")
            context_parts.append(f"A: {conv['answer'][:200]}...")
        
        context_parts.append(f"\nCurrent question: {current_question}")
        return "\n".join(context_parts)
    
    def search_conversations(self, query: str, session_id: str = None) -> List[Dict]:
        """Search through conversation history"""
        conversations = self._load_json(self.conversation_file)
        results = []
        
        sessions_to_search = [session_id] if session_id else conversations["sessions"].keys()
        
        for sid in sessions_to_search:
            if sid in conversations["sessions"]:
                for conv in conversations["sessions"][sid]["conversations"]:
                    if (query.lower() in conv["question"].lower() or 
                        query.lower() in conv["answer"].lower()):
                        results.append({
                            "session_id": sid,
                            "timestamp": conv["timestamp"],
                            "question": conv["question"],
                            "answer": conv["answer"][:200] + "...",
                            "document": conv["document"]
                        })
        
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        conversations = self._load_json(self.conversation_file)
        if session_id in conversations["sessions"]:
            del conversations["sessions"][session_id]
            self._save_json(self.conversation_file, conversations)
    
    def export_memory(self, session_id: str = None) -> Dict:
        """Export memory data"""
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "conversations": self._load_json(self.conversation_file),
            "documents": self._load_json(self.documents_file),
            "analytics": self._load_json(self.analytics_file)
        }
        
        if session_id:
            # Export only specific session
            all_conversations = export_data["conversations"]
            if session_id in all_conversations["sessions"]:
                export_data["conversations"] = {
                    "sessions": {session_id: all_conversations["sessions"][session_id]}
                }
            else:
                export_data["conversations"] = {"sessions": {}}
        
        return export_data 