# main.py
import streamlit as st
import os
import asyncio
import nest_asyncio
from datetime import datetime
from loader import load_pdf_docs, get_file_hash
from vector_database import EnhancedVectorDatabase
from memory import ConversationMemory
from rag_chain import create_rag_chain
from config import GEMINI_API_KEY

# Enable nested event loops for Streamlit compatibility
nest_asyncio.apply()

# Page configuration
st.set_page_config(
    page_title="🧠 Advanced RAG - AI Document Assistant", 
    page_icon="🧠",
    layout="wide"
)

# Initialize systems
@st.cache_resource
def initialize_systems():
    """Initialize vector database and memory systems"""
    vector_db = EnhancedVectorDatabase()
    memory = ConversationMemory()
    return vector_db, memory

# Header
st.title("🧠 Advanced RAG with Vector Database & Memory")
st.markdown("*AI Document Assistant with Persistent Memory & Analytics*")

# Check API key
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found! Please set your API key in the .env file.")
    st.info("1. Create a .env file in your project folder\n2. Add: GEMINI_API_KEY=your_actual_api_key_here\n3. Get your key from: https://makersuite.google.com/app/apikey\n4. Restart the app")
    st.stop()

# Initialize systems
vector_db, memory = initialize_systems()

# Sidebar navigation
st.sidebar.title("🛠️ Control Panel")

# Main tabs
main_tab = st.sidebar.selectbox(
    "Choose Mode:",
    ["📄 Document Chat", "🗃️ Vector Database", "🧠 Memory & Analytics", "⚙️ Settings"]
)

# Settings section
if main_tab == "⚙️ Settings":
    st.sidebar.header("Debug Options")
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", help="Show detailed processing information")
    use_cache = st.sidebar.checkbox("💾 Use Vector Cache", value=True, help="Enable vector database caching")
else:
    debug_mode = False
    use_cache = True

# Document Chat Mode
if main_tab == "📄 Document Chat":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Document Upload & Chat")
        
        # File upload
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        
        if uploaded_file:
            try:
                # Save uploaded file
                file_path = f"docs/{uploaded_file.name}"
                os.makedirs("docs", exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Generate file hash
                file_hash = get_file_hash(file_path)
                
                # Check if document exists in database
                doc_info = vector_db.get_document_info(file_hash)
                
                if doc_info:
                    st.success(f"📦 Found existing document in database: {doc_info['filename']}")
                    vector_store = vector_db.load_document(file_hash)
                    memory.update_document_access(file_hash)
                else:
                    st.success("PDF uploaded successfully! Processing...")
                    
                    # Process the PDF
                    with st.spinner("Loading and processing PDF..."):
                        chunks = load_pdf_docs(file_path, debug=debug_mode)
                        if debug_mode:
                            st.info(f"📄 Created {len(chunks)} text chunks")
                    
                    with st.spinner("Adding to vector database..."):
                        collection_id = vector_db.add_document(
                            file_path, chunks, file_hash,
                            metadata={"upload_source": "streamlit", "user": "default"}
                        )
                        memory.add_document(file_path, file_hash, {"collection_id": collection_id})
                        vector_store = vector_db.load_document(file_hash)
                        
                        if debug_mode:
                            st.info(f"🎯 Added to database with collection ID: {collection_id}")
                
                # Create RAG chain
                with st.spinner("Setting up AI assistant..."):
                    rag_chain = create_rag_chain(vector_store)
                
                st.success("✅ Document ready! Ask your questions below 👇")
                
                # Store in session state
                st.session_state.rag_chain = rag_chain
                st.session_state.current_file_hash = file_hash
                st.session_state.current_file_name = uploaded_file.name
                st.session_state.pdf_processed = True
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                if debug_mode:
                    st.exception(e)
                st.stop()
    
    with col2:
        st.header("📊 Document Info")
        if hasattr(st.session_state, 'current_file_hash'):
            doc_info = vector_db.get_document_info(st.session_state.current_file_hash)
            if doc_info:
                st.write(f"**📁 File:** {doc_info['filename']}")
                st.write(f"**📊 Vectors:** {doc_info['vector_count']:,}")
                st.write(f"**📄 Chunks:** {doc_info['chunk_count']}")
                st.write(f"**👆 Access Count:** {doc_info['access_count']}")
                st.write(f"**📅 Added:** {doc_info['added_at'][:10]}")
                
                # Similar documents
                similar_docs = vector_db.get_similar_documents(st.session_state.current_file_hash)
                if similar_docs:
                    st.write("**🔗 Similar Documents:**")
                    for doc in similar_docs:
                        st.write(f"• {doc['filename']} ({doc['similarity_score']:.2f})")
    
    # Chat interface
    if hasattr(st.session_state, 'pdf_processed') and st.session_state.pdf_processed:
        st.header("💬 Chat with Your Document")
        
        # Generate session ID
        session_id = memory.get_session_id()
        
        # Show conversation history
        with st.expander("📜 Conversation History", expanded=False):
            history = memory.get_conversation_history(session_id, limit=5)
            if history:
                for conv in history:
                    st.write(f"**Q:** {conv['question']}")
                    st.write(f"**A:** {conv['answer'][:300]}...")
                    st.write(f"*{conv['timestamp'][:19]}*")
                    st.divider()
            else:
                st.write("No previous conversations found.")
        
        # Question input
        query = st.text_input("Enter your question about the PDF:")
        
        if query:
            try:
                with st.spinner("🔍 Searching and generating answer..."):
                    # Get conversation context for better answers
                    contextualized_query = memory.get_conversation_context(session_id, query)
                    
                    # Generate answer
                    result = st.session_state.rag_chain.run(contextualized_query)
                    
                    # Save to memory
                    memory.add_conversation(
                        session_id, query, result, 
                        st.session_state.current_file_name,
                        metadata={"file_hash": st.session_state.current_file_hash}
                    )
                
                st.write("### 🤖 Answer:")
                st.write(result)
                
                if debug_mode:
                    with st.expander("🔍 Debug Information"):
                        retriever = st.session_state.rag_chain.retriever
                        docs = retriever.get_relevant_documents(query)
                        st.write(f"**Retrieved {len(docs)} relevant chunks:**")
                        for i, doc in enumerate(docs):
                            st.write(f"**Chunk {i+1}:** {doc.page_content[:200]}...")
                            st.write(f"**Source:** Page {doc.metadata.get('page', 'N/A')}")
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
                if debug_mode:
                    st.exception(e)
    else:
        st.info("👆 Please upload a PDF to get started")

# Vector Database Mode
elif main_tab == "🗃️ Vector Database":
    st.header("🗃️ Vector Database Management")
    
    # Database stats
    stats = vector_db.get_database_stats()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📄 Total Documents", stats.get("total_documents", 0))
    with col2:
        st.metric("🎯 Total Vectors", f"{stats.get('total_vectors', 0):,}")
    with col3:
        last_updated = stats.get("last_updated", "Never")
        if last_updated != "Never":
            last_updated = last_updated[:19].replace("T", " ")
        st.metric("🕐 Last Updated", last_updated)
    
    # Documents table
    st.subheader("📚 Document Library")
    all_docs = vector_db.get_all_documents()
    
    if all_docs:
        docs_data = []
        for file_hash, doc_info in all_docs.items():
            docs_data.append({
                "📁 Filename": doc_info["filename"],
                "📊 Vectors": f"{doc_info['vector_count']:,}",
                "📄 Chunks": doc_info["chunk_count"],
                "👆 Access": doc_info["access_count"],
                "📅 Added": doc_info["added_at"][:10],
                "🆔 Hash": file_hash[:8] + "..."
            })
        
        st.dataframe(docs_data, use_container_width=True)
        
        # Document management
        st.subheader("🛠️ Document Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧹 Cleanup Database"):
                vector_db.cleanup_orphaned_collections()
                st.success("Database cleanup completed!")
                st.rerun()
        
        with col2:
            if st.button("📤 Export Database"):
                export_data = vector_db.export_database()
                st.download_button(
                    "💾 Download Export",
                    data=str(export_data),
                    file_name=f"vector_db_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            # Search across all documents
            search_query = st.text_input("🔍 Search across all documents:")
            if search_query:
                results = vector_db.search_all_documents(search_query, k=6)
                if results:
                    st.write(f"**Found {len(results)} results:**")
                    for doc, filename in results:
                        st.write(f"**{filename}:**")
                        st.write(f"*{doc.page_content[:200]}...*")
                        st.divider()
                else:
                    st.write("No results found.")
    else:
        st.info("No documents in database yet. Upload some PDFs to get started!")

# Memory & Analytics Mode
elif main_tab == "🧠 Memory & Analytics":
    st.header("🧠 Memory & Analytics Dashboard")
    
    # Analytics overview
    analytics = memory.get_analytics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💬 Total Queries", analytics.get("total_queries", 0))
    with col2:
        st.metric("📄 Documents Used", len(analytics.get("document_usage", {})))
    with col3:
        st.metric("📈 Sessions Today", len([s for s in memory._load_json(memory.conversation_file).get("sessions", {}).keys() if datetime.now().strftime("%Y%m%d") in s]))
    
    # Popular queries
    st.subheader("🔥 Popular Queries")
    popular_queries = analytics.get("popular_queries", {})
    if popular_queries:
        sorted_queries = sorted(popular_queries.items(), key=lambda x: x[1], reverse=True)[:10]
        for query, count in sorted_queries:
            st.write(f"**{count}x** - {query}")
    else:
        st.info("No queries recorded yet.")
    
    # Document usage
    st.subheader("📊 Document Usage")
    doc_usage = analytics.get("document_usage", {})
    if doc_usage:
        sorted_docs = sorted(doc_usage.items(), key=lambda x: x[1], reverse=True)
        for doc, count in sorted_docs:
            st.write(f"**{count}x** - {doc}")
    else:
        st.info("No document usage recorded yet.")
    
    # Recent queries
    st.subheader("🕐 Recent Query History")
    query_history = analytics.get("query_history", [])
    if query_history:
        recent_queries = query_history[-10:][::-1]  # Last 10, reversed
        for query_info in recent_queries:
            st.write(f"**{query_info['timestamp'][:19]}** - {query_info['question']}")
            st.write(f"*Document: {query_info['document']}*")
            st.divider()
    else:
        st.info("No query history available.")
    
    # Memory management
    st.subheader("🗂️ Memory Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Search Conversations"):
            search_term = st.text_input("Enter search term:")
            if search_term:
                results = memory.search_conversations(search_term)
                if results:
                    for result in results[:5]:
                        st.write(f"**Q:** {result['question']}")
                        st.write(f"**A:** {result['answer']}")
                        st.write(f"*{result['timestamp'][:19]} - {result['document']}*")
                        st.divider()
    
    with col2:
        session_id = memory.get_session_id()
        if st.button("🗑️ Clear Today's Session"):
            memory.clear_session(session_id)
            st.success("Session cleared!")
            st.rerun()
    
    with col3:
        if st.button("📤 Export Memory"):
            export_data = memory.export_memory()
            st.download_button(
                "💾 Download Memory Export",
                data=str(export_data),
                file_name=f"memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🧠 Advanced RAG System with Vector Database & Memory | 
        Built with Streamlit, LangChain & Google Gemini
    </div>
    """, 
    unsafe_allow_html=True
)
