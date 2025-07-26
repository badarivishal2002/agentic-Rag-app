# 🧠 Advanced RAG PDF Reader App with Vector Database & Memory

A sophisticated Retrieval Augmented Generation (RAG) application with **persistent vector database**, **conversation memory**, and **analytics**. Built with Streamlit, LangChain, and Google's Gemini AI.

## 🚀 Advanced Features

### 📄 **Document Processing**
- **PDF Upload & Processing** - Upload any PDF with intelligent chunking
- **Vector Database Storage** - Persistent FAISS vector storage with metadata
- **Smart Caching** - Instant loading of previously processed documents
- **Multi-Document Support** - Manage multiple documents simultaneously

### 🧠 **Memory & Analytics**
- **Conversation Memory** - Persistent chat history across sessions
- **Context Awareness** - AI remembers previous conversations for better answers
- **Usage Analytics** - Track popular queries and document usage
- **Search History** - Full-text search through conversation history

### 🗃️ **Vector Database Management**
- **Persistent Storage** - All vectors saved with rich metadata
- **Document Library** - Browse and manage all uploaded documents
- **Cross-Document Search** - Search across entire document collection
- **Database Analytics** - Track vectors, documents, and access patterns

### 📊 **Rich Dashboard**
- **4 Main Modes**: Document Chat, Vector Database, Memory & Analytics, Settings
- **Real-time Statistics** - Live metrics and usage data
- **Export Functionality** - Export conversations and database metadata
- **Debug Mode** - Detailed processing information

## 📋 Prerequisites

- Python 3.8+
- Google Generative AI API key (from Google AI Studio)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd agentic-rag-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   
   Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## 🎯 Usage

1. **Start the application**
   ```bash
   python -m streamlit run main.py
   ```

2. **Choose Your Mode**
   - **📄 Document Chat** - Upload PDFs and chat with AI
   - **🗃️ Vector Database** - Manage your document collection
   - **🧠 Memory & Analytics** - View conversation history and analytics
   - **⚙️ Settings** - Configure debug and cache options

3. **Upload & Chat**
   - Upload a PDF document
   - Ask questions in natural language
   - View conversation history and analytics

## 📁 Project Structure

```
agentic-rag-app/
├── main.py                 # Advanced Streamlit interface with multiple modes
├── config.py               # Configuration and API key setup
├── loader.py               # Enhanced PDF loading with metadata
├── vector_database.py      # Advanced vector database with persistence
├── memory.py               # Conversation memory and analytics system
├── rag_chain.py           # RAG chain with Gemini integration
├── vectorstore.py         # Legacy vector store (for compatibility)
├── agent_tools.py         # Agent tools setup
├── requirements.txt       # Python dependencies
├── docs/                  # Uploaded PDFs storage
├── vector_database/       # Persistent vector storage
├── memory/               # Conversation memory storage
├── vectorstore_cache/    # Legacy cache directory
└── README.md             # This file
```

## 🔧 Advanced Configuration

### **Vector Database Settings**
- **Storage Location**: `vector_database/` directory
- **Metadata Tracking**: Full document metadata with access stats
- **Search Capabilities**: Single document or cross-document search

### **Memory System**
- **Session Management**: Daily sessions with persistent history
- **Analytics Tracking**: Query patterns and document usage
- **Export Options**: JSON export of conversations and analytics

### **Model Configuration**
- **LLM Model**: `gemini-1.5-flash` (configurable in `rag_chain.py`)
- **Embedding Model**: `models/embedding-001`
- **Temperature**: 0.7 (adjustable for response creativity)

## 🎨 User Interface Modes

### 📄 **Document Chat Mode**
- Upload and process PDFs
- Real-time chat with AI assistant
- Document information panel
- Conversation history viewer

### 🗃️ **Vector Database Mode**
- View all documents in database
- Database statistics and metrics
- Cross-document search functionality
- Database management tools

### 🧠 **Memory & Analytics Mode**
- Conversation analytics dashboard
- Popular queries tracking
- Document usage statistics
- Memory management tools

### ⚙️ **Settings Mode**
- Debug mode toggle
- Cache configuration
- Advanced processing options

## 📊 Analytics Features

- **Total Queries Tracking** - Monitor AI usage
- **Document Access Patterns** - See which documents are most used
- **Popular Queries** - Track frequently asked questions
- **Session Statistics** - Daily and historical usage data

## 🐛 Troubleshooting

- **API Key Error**: Ensure your `.env` file contains a valid GEMINI_API_KEY
- **Import Errors**: Run `pip install -r requirements.txt` to install all dependencies
- **Memory Issues**: Large PDFs may require more processing time
- **Database Corruption**: Use the "Cleanup Database" feature in Vector Database mode

## 🔧 Technical Details

### **Enhanced Vector Database**
- **Technology**: FAISS with persistent storage
- **Features**: Metadata tracking, access statistics, multi-document support
- **Performance**: Optimized loading with in-memory caching

### **Memory System**
- **Storage**: JSON-based persistent memory
- **Features**: Session management, analytics, search functionality
- **Scalability**: Designed for thousands of conversations

### **AI Integration**
- **Primary LLM**: Google Gemini 1.5 Flash
- **Embeddings**: Google Generative AI Embeddings
- **Context**: Conversation-aware responses with memory integration

## 📝 License

This project is open source and available under the MIT License.

---

## 🚀 Recent Updates

- ✅ **Enhanced Vector Database** - Persistent storage with rich metadata
- ✅ **Conversation Memory** - Full conversation history with analytics  
- ✅ **Multi-Document Support** - Manage multiple PDFs simultaneously
- ✅ **Advanced Analytics** - Usage tracking and query analysis
- ✅ **Rich Dashboard** - Multiple interface modes for different use cases
- ✅ **Export Functionality** - Export conversations and database metadata
- ✅ **Performance Optimization** - Smart caching and async processing 