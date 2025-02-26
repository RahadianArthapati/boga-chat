"""
Boga Chat - Document Management
"""
import os
import tempfile
from typing import List, Dict, Any

import streamlit as st
from utils.api import DocumentAPI

# Page configuration
st.set_page_config(
    page_title="Boga Documents",
    page_icon="📄",
    layout="centered",
)

# Initialize API client
api = DocumentAPI(base_url="http://localhost:8000")


def display_document_chunk(chunk: Dict[str, Any], index: int):
    """Display a document chunk with metadata."""
    with st.expander(f"Chunk {index + 1} - Similarity: {chunk.get('similarity', 'N/A')}", expanded=index == 0):
        st.markdown(f"**Document ID:** {chunk['document_id']}")
        st.markdown(f"**Chunk ID:** {chunk['chunk_id']}")
        
        # Display metadata if available
        if chunk.get('metadata'):
            with st.expander("Metadata"):
                metadata = chunk['metadata']
                if metadata.get('title'):
                    st.markdown(f"**Title:** {metadata['title']}")
                if metadata.get('author'):
                    st.markdown(f"**Author:** {metadata['author']}")
                if metadata.get('source'):
                    st.markdown(f"**Source:** {metadata['source']}")
                if metadata.get('date'):
                    st.markdown(f"**Date:** {metadata['date']}")
                if metadata.get('tags'):
                    st.markdown(f"**Tags:** {', '.join(metadata['tags'])}")
        
        # Display the text content
        st.markdown("### Content")
        st.markdown(chunk['chunk_text'])


def upload_section():
    """Document upload section."""
    st.header("Upload Document")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a text file", type=["txt", "md", "csv", "json"])
    
    # Metadata form
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Title")
        author = st.text_input("Author")
    
    with col2:
        source = st.text_input("Source")
        tags = st.text_input("Tags (comma-separated)")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                # Save uploaded file to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # Process tags
                tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
                
                # Upload document
                result = api.upload_document(
                    file_path=tmp_path,
                    title=title or uploaded_file.name,
                    author=author,
                    source=source,
                    tags=tag_list
                )
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"Document uploaded successfully! Document ID: {result['document_id']}")
                    st.session_state.last_document_id = result['document_id']
                    st.markdown(f"Document has been split into {result['chunk_count']} chunks.")


def search_section():
    """Document search section."""
    st.header("Search Documents")
    
    query = st.text_input("Search Query")
    
    col1, col2 = st.columns(2)
    with col1:
        limit = st.slider("Result Limit", min_value=1, max_value=20, value=5)
    
    with col2:
        threshold = st.slider("Similarity Threshold", min_value=0.1, max_value=1.0, value=0.7, step=0.05)
    
    if query:
        if st.button("Search"):
            with st.spinner("Searching..."):
                results = api.search_documents(
                    query=query,
                    limit=limit,
                    similarity_threshold=threshold
                )
                
                if "error" in results:
                    st.error(results["error"])
                elif not results.get("results"):
                    st.info("No results found. Try adjusting your search query or lowering the similarity threshold.")
                else:
                    st.success(f"Found {len(results['results'])} results")
                    
                    # Display results
                    for i, chunk in enumerate(results["results"]):
                        display_document_chunk(chunk, i)


def view_document_section():
    """View document by ID section."""
    st.header("View Document")
    
    # Use the last document ID if available
    default_id = st.session_state.get("last_document_id", "")
    document_id = st.text_input("Document ID", value=default_id)
    
    if document_id:
        if st.button("View Document"):
            with st.spinner("Loading document..."):
                chunks = api.get_document(document_id)
                
                if not chunks:
                    st.error("Document not found or error retrieving document.")
                else:
                    st.success(f"Document loaded with {len(chunks)} chunks")
                    
                    # Display chunks
                    for i, chunk in enumerate(chunks):
                        display_document_chunk(chunk, i)
                    
                    # Delete option
                    if st.button("Delete Document"):
                        if st.checkbox("Confirm deletion"):
                            result = api.delete_document(document_id)
                            if "error" in result:
                                st.error(result["error"])
                            else:
                                st.success(result["message"])
                                if "last_document_id" in st.session_state and st.session_state.last_document_id == document_id:
                                    del st.session_state.last_document_id


def main():
    """Main application function."""
    st.title("Boga Documents 📄")
    st.markdown("""
    Manage and search documents using vector embeddings.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Upload", "Search", "View Document"])
    
    with tab1:
        upload_section()
    
    with tab2:
        search_section()
    
    with tab3:
        view_document_section()
    
    # Sidebar
    with st.sidebar:
        st.title("About")
        st.markdown("""
        **Boga Documents** is a document management system built with:
        - 🔗 LangChain
        - 📊 OpenAI Embeddings
        - 🗄️ Supabase pgvector
        - 📱 Streamlit
        """)
        
        st.markdown("---")
        
        if st.button("Go to Chat"):
            st.switch_page("app.py")


if __name__ == "__main__":
    main() 