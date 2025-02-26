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


def display_messages():
    """Display all messages in the chat."""
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"user_{i}")
        else:
            message(msg["content"], is_user=False, key=f"assistant_{i}")


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
    api = ChatAPI(base_url="http://localhost:8080")
    response = api.send_message(
        st.session_state.messages,
        st.session_state.conversation_id
    )
    
    # Update conversation ID
    st.session_state.conversation_id = response.get("conversation_id")
    
    # Add assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response.get("response", "")})


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
        
        # New conversation button
        if st.button("New Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
        
        # Show conversation ID
        if st.session_state.conversation_id:
            st.text_input(
                "Conversation ID",
                value=st.session_state.conversation_id,
                disabled=True
            )


if __name__ == "__main__":
    main() 