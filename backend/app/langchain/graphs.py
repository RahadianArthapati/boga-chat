"""
LangGraph workflows for Boga Chat.
"""
from typing import Dict, List, Any, TypedDict
import uuid
import json
import logging

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.config import settings
from app.db.supabase import get_supabase_client
from app.langchain.router import should_use_rag
from app.langchain.rag import get_relevant_documents, format_documents_for_prompt

logger = logging.getLogger(__name__)

# Define the state for our chat graph
class ChatState(TypedDict):
    """State for the chat graph."""
    messages: List[Dict[str, str]]
    conversation_id: str
    metadata: Dict[str, Any]


# Enhanced state with routing decision and document retrieval
class RoutedChatState(TypedDict):
    """State for the routed chat graph with RAG support."""
    messages: List[Dict[str, str]]
    conversation_id: str
    metadata: Dict[str, Any]
    documents: List[Dict[str, Any]]
    use_rag: bool
    routing_decision: Dict[str, Any]
    last_user_query: str


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
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY,
        streaming=True
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
    
    # Create a copy of the messages and append the new message
    new_messages = state["messages"].copy()
    new_messages.append({
        "role": "assistant",
        "content": response.content
    })
    
    # Create a completely new state object with all fields explicitly copied
    return {
        "messages": new_messages,
        "conversation_id": state["conversation_id"],
        "metadata": state["metadata"].copy()
    }


def save_to_database(state: ChatState) -> ChatState:
    """
    This function is a no-op as the conversations table is not needed.
    In a production environment, this would save the conversation to a database.
    
    Args:
        state: The current chat state
        
    Returns:
        A new copy of the state
    """
    # Log that we're skipping database save
    logger.debug(f"Skipping save to conversations table for conversation {state['conversation_id']}")
    
    # Create a completely new state object with all fields explicitly copied
    return {
        "messages": state["messages"].copy(),
        "conversation_id": state["conversation_id"],
        "metadata": state["metadata"].copy()
    }


# New functions for the routed graph

async def route_query(state: RoutedChatState) -> RoutedChatState:
    """
    Determine whether to use RAG based on the last user query.
    
    Args:
        state: The current chat state
        
    Returns:
        Updated state with routing decision
    """
    # Extract the last user message
    last_user_message = next(
        (msg["content"] for msg in reversed(state["messages"]) if msg["role"] == "user"),
        ""
    )
    
    # Use the router to decide
    routing_decision = await should_use_rag(last_user_message)
    
    # Create a completely new state object with all fields explicitly copied
    return {
        "messages": state["messages"].copy(),
        "conversation_id": state["conversation_id"],
        "metadata": state["metadata"].copy(),
        "documents": state["documents"].copy(),
        "use_rag": routing_decision.get("use_rag", False),
        "routing_decision": routing_decision,
        "last_user_query": last_user_message
    }


async def retrieve_documents(state: RoutedChatState) -> RoutedChatState:
    """
    Retrieve relevant documents if RAG is enabled.
    
    Args:
        state: The current chat state
        
    Returns:
        Updated state with retrieved documents
    """
    documents = []
    
    if state["use_rag"]:
        # Retrieve documents using the last user query
        logger.info(f"Retrieving documents for query: {state['last_user_query']}")
        documents = await get_relevant_documents(
            query=state["last_user_query"],
            limit=5,
            threshold=0.3
        )
        logger.info(f"Retrieved {len(documents)} documents")
        logger.debug(f"Retrieved documents: {json.dumps(documents, indent=2)}")
    else:
        logger.debug("Skipping document retrieval as RAG is disabled")
    
    # Create a completely new state object with all fields explicitly copied
    return {
        "messages": state["messages"].copy(),
        "conversation_id": state["conversation_id"],
        "metadata": state["metadata"].copy(),
        "documents": documents,
        "use_rag": state["use_rag"],
        "routing_decision": state["routing_decision"].copy(),
        "last_user_query": state["last_user_query"]
    }


def generate_rag_response(state: RoutedChatState) -> RoutedChatState:
    """
    Generate a response using RAG.
    
    Args:
        state: The current chat state with documents
        
    Returns:
        Updated state with the assistant's response
    """
    # Get the chat model
    model = ChatOpenAI(
        model="gpt-4o-mini",
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
    
    # Format chat history for the prompt
    chat_history = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in state["messages"][:-1]  # Exclude the last message
    ])
    
    # Format documents for the prompt
    documents_text = format_documents_for_prompt(state["documents"])
    
    # Create RAG prompt
    rag_template = """
    You are a helpful and friendly AI assistant with access to a knowledge base of documents.
    
    Current conversation:
    {chat_history}
    
    Relevant documents for the query:
    {documents}
    
    Human: {input}
    
    Based on the conversation history and the relevant documents provided, please respond to the human's query.
    If the documents don't contain relevant information, just respond based on your general knowledge.
    AI: 
    """
    
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    
    # Prepare inputs
    inputs = {
        "chat_history": chat_history,
        "documents": documents_text,
        "input": state["last_user_query"]
    }
    
    # Generate response
    prompt = rag_prompt.format(**inputs)
    response = model.invoke(prompt)
    
    # Create a copy of the messages and append the new message
    new_messages = state["messages"].copy()
    new_messages.append({"role": "assistant", "content": response.content})
    
    # Create a completely new state object with all fields explicitly copied
    return {
        "messages": new_messages,
        "conversation_id": state["conversation_id"],
        "metadata": state["metadata"].copy(),
        "documents": state["documents"].copy(),
        "use_rag": state["use_rag"],
        "routing_decision": state["routing_decision"].copy(),
        "last_user_query": state["last_user_query"]
    }


def generate_standard_response(state: RoutedChatState) -> RoutedChatState:
    """
    Generate a standard response without RAG.
    
    Args:
        state: The current chat state
        
    Returns:
        Updated state with the assistant's response
    """
    # Get the chat model
    model = ChatOpenAI(
        model="gpt-4o-mini",
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
    
    # Create a copy of the messages and append the new message
    new_messages = state["messages"].copy()
    new_messages.append({"role": "assistant", "content": response.content})
    
    # Create a completely new state object with all fields explicitly copied
    return {
        "messages": new_messages,
        "conversation_id": state["conversation_id"],
        "metadata": state["metadata"].copy(),
        "documents": state["documents"].copy(),
        "use_rag": state["use_rag"],
        "routing_decision": state["routing_decision"].copy(),
        "last_user_query": state["last_user_query"]
    }


def save_routed_to_database(state: RoutedChatState) -> RoutedChatState:
    """
    This function is a no-op as the conversations table is not needed.
    In a production environment, this would save the conversation to a database.
    
    Args:
        state: The current chat state
        
    Returns:
        A new copy of the state
    """
    # Log that we're skipping database save
    logger.debug(f"Skipping save to conversations table for conversation {state['conversation_id']}")
    
    # Create a completely new state object with all fields explicitly copied
    return {
        "messages": state["messages"].copy(),
        "conversation_id": state["conversation_id"],
        "metadata": state["metadata"].copy(),
        "documents": state["documents"].copy(),
        "use_rag": state["use_rag"],
        "routing_decision": state["routing_decision"].copy(),
        "last_user_query": state["last_user_query"]
    }


# Build the original graph
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


# Build the new routed graph
def build_routed_chat_graph():
    """
    Build and return the routed chat workflow graph with RAG support.
    
    Returns:
        StateGraph: The configured routed chat workflow graph
    """
    # Create a new graph
    graph = StateGraph(RoutedChatState)
    
    # Add nodes
    graph.add_node("route_query", route_query)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("generate_rag_response", generate_rag_response)
    graph.add_node("generate_standard_response", generate_standard_response)
    graph.add_node("save_to_database", save_routed_to_database)
    
    # Define the edges
    graph.add_edge("route_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", "generate_rag_response")
    graph.add_edge("retrieve_documents", "generate_standard_response")
    graph.add_edge("generate_rag_response", "save_to_database")
    graph.add_edge("generate_standard_response", "save_to_database")
    graph.add_edge("save_to_database", END)
    
    # Set the conditional logic for routing
    graph.add_conditional_edges(
        "retrieve_documents",
        lambda state: "generate_rag_response" if state["use_rag"] and state["documents"] else "generate_standard_response"
    )
    
    # Set the entry point
    graph.set_entry_point("route_query")
    
    # Compile the graph
    return graph.compile()


def get_chat_graph():
    """
    Get the compiled chat graph for processing chat requests.
    
    Returns:
        The compiled chat graph
    """
    return build_chat_graph()


def get_routed_chat_graph():
    """
    Get the compiled routed chat graph for processing chat requests with intelligent routing.
    
    Returns:
        The compiled routed chat graph
    """
    return build_routed_chat_graph()


def process_with_graph(messages: List[Dict[str, str]], conversation_id: str = None) -> Dict[str, Any]:
    """
    Process a chat request using a sequential approach instead of the graph.
    This avoids LangGraph state management issues.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        conversation_id: Optional conversation ID
        
    Returns:
        Dict with response and conversation_id
    """
    # If no conversation_id, create a new one
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Initialize state as a plain dictionary
    state = {
        "messages": messages.copy(),
        "conversation_id": conversation_id,
        "metadata": {}
    }
    
    # Step 1: Generate response
    state = generate_response(state)
    
    # Step 2: Save to database
    state = save_to_database(state)
    
    # Return the result
    return {
        "response": state["messages"][-1]["content"] if state["messages"] else "",
        "conversation_id": state["conversation_id"]
    }


async def process_with_routed_graph(messages: List[Dict[str, str]], conversation_id: str = None) -> Dict[str, Any]:
    """
    Process a chat request using a sequential approach instead of the graph.
    This avoids LangGraph state management issues.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        conversation_id: Optional conversation ID
        
    Returns:
        Dict with response, conversation_id, and documents
    """
    # If no conversation_id, create a new one
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Initialize state as a plain dictionary
    state = {
        "messages": messages.copy(),
        "conversation_id": conversation_id,
        "metadata": {},
        "documents": [],
        "use_rag": False,
        "routing_decision": {},
        "last_user_query": ""
    }
    
    # Step 1: Route the query
    state = await route_query(state)
    
    # Step 2: Retrieve documents
    state = await retrieve_documents(state)
    
    # Step 3: Generate response based on routing decision
    if state["use_rag"] and state["documents"]:
        state = generate_rag_response(state)
    else:
        state = generate_standard_response(state)
    
    # Step 4: Save to database
    state = save_routed_to_database(state)
    
    # Return the result
    return {
        "response": state["messages"][-1]["content"] if state["messages"] else "",
        "conversation_id": state["conversation_id"],
        "documents": state["documents"],
        "use_rag": state["use_rag"],
        "routing_decision": state["routing_decision"]
    } 