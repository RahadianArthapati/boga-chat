"""
LangGraph workflows for Boga Chat.
"""
from typing import Dict, List, Any, Annotated, TypedDict
import uuid
import json

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.config import settings
from app.db.supabase import get_supabase_client


# Define the state for our graph
class ChatState(TypedDict):
    """State for the chat graph."""
    messages: List[Dict[str, str]]
    conversation_id: str
    metadata: Dict[str, Any]


# Node functions
def generate_response(state: ChatState) -> ChatState:
    """
    Generate a response using the LLM.
    
    Args:
        state: The current chat state
        
    Returns:
        Updated state with the assistant's response
    """
    # Get the chat model
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY
    )
    
    # Format messages for the model
    formatted_messages = []
    for msg in state["messages"]:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_messages.append(AIMessage(content=msg["content"]))
    
    # Generate response
    response = model.invoke(formatted_messages)
    
    # Update state with the response
    state["messages"].append({
        "role": "assistant",
        "content": response.content
    })
    
    return state


def save_to_database(state: ChatState) -> ChatState:
    """
    Save the conversation to the database.
    
    Args:
        state: The current chat state
        
    Returns:
        The unchanged state
    """
    supabase = get_supabase_client()
    
    # Save to Supabase
    supabase.table("conversations").upsert({
        "id": state["conversation_id"],
        "messages": state["messages"]
    }).execute()
    
    return state


# Build the graph
def build_chat_graph():
    """
    Build and return the chat workflow graph.
    
    Returns:
        StateGraph: The configured chat workflow graph
    """
    # Create a new graph
    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("generate_response", generate_response)
    graph.add_node("save_to_database", save_to_database)
    
    # Define the edges
    graph.add_edge("generate_response", "save_to_database")
    graph.add_edge("save_to_database", END)
    
    # Set the entry point
    graph.set_entry_point("generate_response")
    
    # Compile the graph
    return graph.compile()


def get_chat_graph():
    """
    Get the compiled chat graph for processing chat requests.
    
    Returns:
        The compiled chat graph
    """
    return build_chat_graph()


def process_with_graph(messages: List[Dict[str, str]], conversation_id: str = None) -> Dict[str, Any]:
    """
    Process a chat request using the LangGraph workflow.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        conversation_id: Optional conversation ID
        
    Returns:
        Dict with response and conversation_id
    """
    # Get the graph
    graph = get_chat_graph()
    
    # If no conversation_id, create a new one
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Initialize state
    state = ChatState(
        messages=messages,
        conversation_id=conversation_id,
        metadata={}
    )
    
    # Run the graph
    result = graph.invoke(state)
    
    # Extract the response
    response = result["messages"][-1]["content"] if result["messages"] else ""
    
    return {
        "response": response,
        "conversation_id": result["conversation_id"]
    } 