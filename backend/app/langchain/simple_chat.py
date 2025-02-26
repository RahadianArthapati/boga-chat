"""
Simplified chat processing without LangGraph.
This module provides a simpler implementation of chat processing
to avoid state management issues with LangGraph.
"""
from typing import Dict, List, Any
import uuid
import logging

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.db.supabase import get_supabase_client
from app.langchain.router import should_use_rag
from app.langchain.rag import get_relevant_documents, format_documents_for_prompt

logger = logging.getLogger(__name__)


async def process_chat(
    messages: List[Dict[str, str]], 
    conversation_id: str = None,
    use_rag: bool = None
) -> Dict[str, Any]:
    """
    Process a chat request with optional RAG support.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        conversation_id: Optional conversation ID
        use_rag: Whether to use RAG. If None, will be determined by router
        
    Returns:
        Dict with response, conversation_id, and optional documents
    """
    # If no conversation_id, create a new one
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Extract the last user message
    last_user_message = next(
        (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
        ""
    )
    
    # Determine whether to use RAG
    routing_decision = {}
    documents = []
    
    if use_rag is None:
        # Use the router to decide
        routing_decision = await should_use_rag(last_user_message)
        use_rag = routing_decision.get("use_rag", False)
        logger.info(f"Router decision for query '{last_user_message[:30]}...': {routing_decision}")
    else:
        routing_decision = {
            "use_rag": use_rag,
            "reasoning": "Manually set by user"
        }
    
    # Retrieve documents if using RAG
    if use_rag:
        documents = await get_relevant_documents(
            query=last_user_message,
            limit=3,
            threshold=0.45
        )
    
    # Generate response
    response_content = ""
    
    if use_rag and documents:
        # Generate RAG response
        response_content = await generate_rag_response(
            messages=messages,
            documents=documents,
            last_user_query=last_user_message
        )
    else:
        # Generate standard response
        response_content = await generate_standard_response(messages)
    
    # Create a copy of messages with the new response
    updated_messages = messages.copy()
    updated_messages.append({
        "role": "assistant",
        "content": response_content
    })
    
    # Save to database
    save_to_database(
        conversation_id=conversation_id,
        messages=updated_messages,
        use_rag=use_rag,
        routing_decision=routing_decision,
        has_documents=len(documents) > 0
    )
    
    # Return the result
    return {
        "response": response_content,
        "conversation_id": conversation_id,
        "documents": documents,
        "use_rag": use_rag,
        "routing_decision": routing_decision
    }


async def generate_rag_response(
    messages: List[Dict[str, str]],
    documents: List[Dict[str, Any]],
    last_user_query: str
) -> str:
    """
    Generate a response using RAG.
    
    Args:
        messages: List of message dictionaries
        documents: List of retrieved documents
        last_user_query: The last user query
        
    Returns:
        The generated response text
    """
    # Get the chat model
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY
    )
    
    # Format chat history for the prompt
    chat_history = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in messages[:-1]  # Exclude the last message
    ])
    
    # Format documents for the prompt
    documents_text = format_documents_for_prompt(documents)
    
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
        "input": last_user_query
    }
    
    # Generate response
    prompt = rag_prompt.format(**inputs)
    response = model.invoke(prompt)
    
    return response.content


async def generate_standard_response(messages: List[Dict[str, str]]) -> str:
    """
    Generate a standard response without RAG.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        The generated response text
    """
    # Get the chat model
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY
    )
    
    # Format messages for the model
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_messages.append(AIMessage(content=msg["content"]))
    
    # Generate response
    response = model.invoke(formatted_messages)
    
    return response.content


def save_to_database(
    conversation_id: str,
    messages: List[Dict[str, str]],
    use_rag: bool = False,
    routing_decision: Dict[str, Any] = None,
    has_documents: bool = False
) -> None:
    """
    This function is a no-op as the conversations table is not needed.
    In a production environment, this would save the conversation to a database.
    
    Args:
        conversation_id: The conversation ID
        messages: List of message dictionaries
        use_rag: Whether RAG was used
        routing_decision: The routing decision
        has_documents: Whether documents were retrieved
    """
    # Log that we're skipping database save
    logger.debug(f"Skipping save to conversations table for conversation {conversation_id}")
    # No database operation is performed


async def stream_chat(
    messages: List[Dict[str, str]], 
    conversation_id: str = None,
    use_rag: bool = None
) -> Dict[str, Any]:
    """
    Process a chat request with streaming response.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        conversation_id: Optional conversation ID
        use_rag: Whether to use RAG. If None, will be determined by router
        
    Returns:
        An async generator that yields response chunks
    """
    # If no conversation_id, create a new one
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Extract the last user message
    last_user_message = next(
        (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
        ""
    )
    
    # Determine whether to use RAG
    routing_decision = {}
    documents = []
    
    if use_rag is None:
        # Use the router to decide
        routing_decision = await should_use_rag(last_user_message)
        use_rag = routing_decision.get("use_rag", False)
        logger.info(f"Router decision for query '{last_user_message[:30]}...': {routing_decision}")
    else:
        routing_decision = {
            "use_rag": use_rag,
            "reasoning": "Manually set by user"
        }
    
    # Retrieve documents if using RAG
    if use_rag:
        documents = await get_relevant_documents(
            query=last_user_message,
            limit=3,
            threshold=0.45
        )
    
    # Get the streaming model
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY,
        streaming=True
    )
    
    # Prepare the prompt based on whether we're using RAG
    if use_rag and documents:
        # Format chat history for the prompt
        chat_history = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in messages[:-1]  # Exclude the last message
        ])
        
        # Format documents for the prompt
        documents_text = format_documents_for_prompt(documents)
        
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
            "input": last_user_message
        }
        
        # Format the prompt
        prompt = rag_prompt.format(**inputs)
        
        # Create a message for the model
        messages_for_model = [HumanMessage(content=prompt)]
    else:
        # Format messages for the model
        messages_for_model = []
        for msg in messages:
            if msg["role"] == "user":
                messages_for_model.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages_for_model.append(AIMessage(content=msg["content"]))
    
    # Create a copy of messages for saving later
    updated_messages = messages.copy()
    
    # Stream the response
    full_response = ""
    async for chunk in model.astream(messages_for_model):
        chunk_content = chunk.content
        full_response += chunk_content
        
        # Yield the chunk
        yield {
            "event": "chunk",
            "data": {
                "chunk": chunk_content,
                "conversation_id": conversation_id,
                "documents": documents,
                "routing_decision": routing_decision
            }
        }
    
    # Add the full response to messages
    updated_messages.append({
        "role": "assistant",
        "content": full_response
    })
    
    # Save to database (with error handling)
    try:
        save_to_database(
            conversation_id=conversation_id,
            messages=updated_messages,
            use_rag=use_rag,
            routing_decision=routing_decision,
            has_documents=len(documents) > 0
        )
    except Exception as e:
        logger.warning(f"Failed to save streamed conversation to database: {str(e)}")
    
    # Yield the final event
    yield {
        "event": "done",
        "data": {
            "full_response": full_response,
            "conversation_id": conversation_id,
            "documents": documents,
            "routing_decision": routing_decision
        }
    } 