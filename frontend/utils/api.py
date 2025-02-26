"""
API client for communicating with the Boga Chat backend.
"""
import json
from typing import Dict, List, Any, Optional

import requests
import streamlit as st


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
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a message to the chat API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            conversation_id: Optional conversation ID
            
        Returns:
            Dict with response and conversation_id
        """
        try:
            url = f"{self.base_url}/api/chat/"
            
            payload = {
                "messages": messages,
                "conversation_id": conversation_id
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