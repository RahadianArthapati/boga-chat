"""
Boga Chat - Streamlit Frontend
"""
import json
import uuid
from typing import List, Dict

import streamlit as st
from streamlit_chat import message

from utils.api import ChatAPI

# Page configuration
st.set_page_config(
    page_title="Boga Chat",
    page_icon="💬",
    layout="centered",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "use_rag" not in st.session_state:
    st.session_state.use_rag = False

if "documents" not in st.session_state:
    st.session_state.documents = []


def display_messages():
    """Display all messages in the chat."""
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"user_{i}")
        else:
            message(msg["content"], is_user=False, key=f"assistant_{i}")


def display_documents():
    """Display retrieved documents if available."""
    if st.session_state.documents and len(st.session_state.documents) > 0:
        with st.expander("📚 Retrieved Documents", expanded=False):
            for i, doc in enumerate(st.session_state.documents):
                metadata = doc.get("metadata", {})
                title = metadata.get("title", "Untitled Document")
                source = metadata.get("source", "Unknown Source")
                similarity = doc.get("similarity", 0)
                
                st.markdown(f"**Document {i+1}**: {title} (Source: {source}) - Similarity: {similarity:.2f}")
                st.text_area(f"Content {i+1}", doc["chunk_text"], height=150, key=f"doc_{i}")
                st.markdown("---")


def handle_submit():
    """Handle user message submission."""
    user_message = st.session_state.user_input
    
    if not user_message:
        return
    
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    # Clear input
    st.session_state.user_input = ""
    
    # Call API
    api = ChatAPI(base_url="http://localhost:8000")
    response = api.send_message(
        st.session_state.messages,
        st.session_state.conversation_id,
        use_rag=st.session_state.use_rag
    )
    
    # Update conversation ID
    st.session_state.conversation_id = response.get("conversation_id")
    
    # Update documents if available
    if "documents" in response and response["documents"]:
        st.session_state.documents = response["documents"]
    else:
        st.session_state.documents = []
    
    # Add assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response.get("response", "")})


def toggle_rag():
    """Toggle RAG functionality."""
    st.session_state.use_rag = not st.session_state.use_rag


def main():
    """Main application function."""
    # Header
    st.title("Boga Chat 💬")
    st.markdown("""
    Welcome to Boga Chat! A chatbot powered by LangChain, LangSmith, and LangGraph.
    """)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        display_messages()
        
        # Display documents if RAG is enabled and documents are available
        if st.session_state.use_rag:
            display_documents()
    
    # User input
    st.text_input(
        "Your message",
        key="user_input",
        on_change=handle_submit,
        placeholder="Type your message here...",
    )
    
    # Sidebar
    with st.sidebar:
        st.title("About")
        st.markdown("""
        **Boga Chat** is a chatbot application built with:
        - 🔗 LangChain
        - 📊 LangSmith
        - 📈 LangGraph
        - 🚀 FastAPI
        - 📱 Streamlit
        - 🗄️ Supabase
        """)
        
        # RAG toggle
        st.markdown("### Settings")
        rag_enabled = st.toggle("Enable Document Retrieval (RAG)", value=st.session_state.use_rag, key="rag_toggle", on_change=toggle_rag)
        
        if rag_enabled:
            st.info("Document retrieval is enabled. The chatbot will search for relevant documents to answer your questions.")
        
        st.markdown("---")
        
        # New conversation button
        if st.button("New Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.session_state.documents = []
            st.rerun()
        
        # Show conversation ID
        if st.session_state.conversation_id:
            st.text_input(
                "Conversation ID",
                value=st.session_state.conversation_id,
                disabled=True
            )
            
        st.markdown("---")
        
        # Link to Documents page
        if st.button("Go to Documents"):
            st.switch_page("pages/documents.py")


if __name__ == "__main__":
    main() 