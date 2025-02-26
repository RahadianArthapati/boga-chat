"""
API client for communicating with the Boga Chat backend.
"""
import json
from typing import Dict, List, Any, Optional, Generator, Callable
import os

import requests
import streamlit as st
import sseclient


class ChatAPI:
    """Client for interacting with the Boga Chat API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: The base URL of the API
        """
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def send_message(
        self, 
        messages: List[Dict[str, str]], 
        conversation_id: Optional[str] = None,
        stream: bool = False,
        use_rag: bool = False
    ) -> Dict[str, Any]:
        """
        Send a message to the chat API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            conversation_id: Optional conversation ID
            stream: Whether to stream the response
            use_rag: Whether to use RAG (Retrieval Augmented Generation)
            
        Returns:
            Dict with response, conversation_id, and optional documents
        """
        try:
            # Use the RAG endpoint if requested
            url = f"{self.base_url}/api/chat/rag" if use_rag else f"{self.base_url}/api/chat/"
            
            payload = {
                "messages": messages,
                "conversation_id": conversation_id,
                "stream": stream,
                "use_rag": use_rag
            }
            
            response = requests.post(
                url, 
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                return {
                    "response": "Sorry, I encountered an error. Please try again.",
                    "conversation_id": conversation_id
                }
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return {
                "response": "Sorry, I couldn't connect to the server. Please try again later.",
                "conversation_id": conversation_id
            }
    
    def stream_message(
        self,
        messages: List[Dict[str, str]],
        conversation_id: Optional[str] = None,
        on_chunk: Callable[[str, str], None] = None,
        use_rag: bool = False
    ) -> Dict[str, Any]:
        """
        Stream a message from the chat API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            conversation_id: Optional conversation ID
            on_chunk: Callback function to handle each chunk
            use_rag: Whether to use RAG (Retrieval Augmented Generation)
            
        Returns:
            Dict with full_response, conversation_id, and optional documents
        """
        try:
            url = f"{self.base_url}/api/chat/stream"
            
            payload = {
                "messages": messages,
                "conversation_id": conversation_id,
                "use_rag": use_rag
            }
            
            # Make a streaming request
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                stream=True
            )
            
            if response.status_code != 200:
                st.error(f"Error: {response.status_code} - {response.text}")
                return {
                    "full_response": "Sorry, I encountered an error. Please try again.",
                    "conversation_id": conversation_id
                }
            
            # Create SSE client
            client = sseclient.SSEClient(response)
            
            # Process events
            full_response = ""
            result_conversation_id = conversation_id
            documents = None
            
            for event in client.events():
                if event.event == "chunk":
                    data = json.loads(event.data)
                    chunk = data.get("chunk", "")
                    result_conversation_id = data.get("conversation_id", conversation_id)
                    documents = data.get("documents")
                    
                    # Update full response
                    full_response += chunk
                    
                    # Call callback if provided
                    if on_chunk:
                        on_chunk(chunk, result_conversation_id)
                
                elif event.event == "done":
                    data = json.loads(event.data)
                    full_response = data.get("full_response", full_response)
                    result_conversation_id = data.get("conversation_id", result_conversation_id)
                    documents = data.get("documents", documents)
                    break
            
            return {
                "full_response": full_response,
                "conversation_id": result_conversation_id,
                "documents": documents
            }
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return {
                "full_response": "Sorry, I couldn't connect to the server. Please try again later.",
                "conversation_id": conversation_id
            }
    
    def get_conversation(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Retrieve a conversation by ID.
        
        Args:
            conversation_id: The ID of the conversation to retrieve
            
        Returns:
            List of message dictionaries
        """
        try:
            url = f"{self.base_url}/api/chat/conversations/{conversation_id}"
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return []


class DocumentAPI:
    """Client for interacting with the Document API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: The base URL of the API
        """
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def upload_document(
        self, 
        file_path: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Upload a document to the API.
        
        Args:
            file_path: Path to the file to upload
            title: Optional document title
            author: Optional document author
            source: Optional document source
            tags: Optional list of tags
            
        Returns:
            Dict with document_id and chunk_count
        """
        try:
            url = f"{self.base_url}/api/documents/upload"
            
            # Prepare form data
            files = {"file": open(file_path, "rb")}
            
            data = {}
            if title:
                data["title"] = title
            if author:
                data["author"] = author
            if source:
                data["source"] = source
            if tags:
                data["tags"] = ",".join(tags)
            
            # Make request
            response = requests.post(
                url,
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                return {
                    "error": f"Error uploading document: {response.text}"
                }
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return {
                "error": f"Error uploading document: {str(e)}"
            }
    
    def search_documents(
        self,
        query: str,
        limit: int = 5,
        similarity_threshold: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for documents using vector similarity.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            metadata_filter: Optional filter for metadata fields
            
        Returns:
            Dict with search results
        """
        try:
            url = f"{self.base_url}/api/documents/search"
            
            payload = {
                "query": query,
                "limit": limit,
                "similarity_threshold": similarity_threshold
            }
            
            if metadata_filter:
                payload["metadata_filter"] = metadata_filter
            
            response = requests.post(
                url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                return {
                    "results": [],
                    "error": f"Error searching documents: {response.text}"
                }
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return {
                "results": [],
                "error": f"Error searching documents: {str(e)}"
            }
    
    def get_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            List of document chunks
        """
        try:
            url = f"{self.base_url}/api/documents/{document_id}"
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document by ID.
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            Dict with success message
        """
        try:
            url = f"{self.base_url}/api/documents/{document_id}"
            
            response = requests.delete(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                return {
                    "error": f"Error deleting document: {response.text}"
                }
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return {
                "error": f"Error deleting document: {str(e)}"
            } 