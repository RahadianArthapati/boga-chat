"""
Chat API endpoints for Boga Chat.
"""
from typing import Dict, List, Optional, Any
import logging
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.langchain.chains import get_chat_chain, get_streaming_chat_chain
from app.langchain.rag import process_with_rag, stream_with_rag
from app.langchain.simple_chat import process_chat, stream_chat as simple_stream_chat
from app.db.supabase import get_supabase_client
from app.langchain.router import should_use_rag

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    """Chat request model."""
    messages: List[ChatMessage]
    conversation_id: Optional[str] = None
    stream: bool = False
    use_rag: Optional[bool] = None  # Now optional, will be determined by LLM if not provided


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    conversation_id: str
    documents: Optional[List[Dict[str, Any]]] = None
    use_rag: Optional[bool] = None  # Added to show if RAG was used
    routing_decision: Optional[Dict[str, Any]] = None  # Added to show routing logic


class StreamingChatResponse(BaseModel):
    """Streaming chat response model."""
    chunk: str
    full_response: Optional[str] = None
    conversation_id: str
    documents: Optional[List[Dict[str, Any]]] = None
    done: bool = False


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    supabase = Depends(get_supabase_client)
):
    """
    Process a chat request and return a response.
    
    Args:
        request: The chat request containing messages and optional conversation_id
        
    Returns:
        ChatResponse: The assistant's response and conversation ID
    """
    # If streaming is requested, use the streaming endpoint
    if request.stream:
        return await stream_chat(request, supabase)
    
    try:
        # Format messages for the chain
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        # Use the new simplified chat processing
        logger.info(f"Processing chat request with {len(formatted_messages)} messages")
        result = await process_chat(
            messages=formatted_messages,
            conversation_id=request.conversation_id,
            use_rag=request.use_rag
        )
        
        logger.info(f"Chat result: {result}")
        
        # Return the response with routing information
        return ChatResponse(
            response=result.get("response", ""),
            conversation_id=result.get("conversation_id"),
            documents=result.get("documents"),
            use_rag=result.get("use_rag"),
            routing_decision=result.get("routing_decision")
        )
    
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    supabase = Depends(get_supabase_client)
):
    """
    Process a chat request and stream the response.
    
    Args:
        request: The chat request containing messages and optional conversation_id
        
    Returns:
        StreamingResponse: A streaming response with the assistant's response
    """
    try:
        # Format messages for the chain
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        logger.info(f"Streaming chat request with {len(formatted_messages)} messages")
        
        # Use the streaming function from simple_chat
        async def event_generator():
            async for event in simple_stream_chat(
                messages=formatted_messages,
                conversation_id=request.conversation_id,
                use_rag=request.use_rag
            ):
                yield event
        
        return EventSourceResponse(event_generator())
    
    except Exception as e:
        logger.error(f"Error streaming chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag", response_model=ChatResponse)
async def rag_chat(
    request: ChatRequest,
    supabase = Depends(get_supabase_client)
):
    """
    Process a chat request with RAG and return a response.
    
    Args:
        request: The chat request containing messages and optional conversation_id
        
    Returns:
        ChatResponse: The assistant's response, conversation ID, and relevant documents
    """
    # Set use_rag to True and delegate to the main chat endpoint
    request.use_rag = True
    return await chat(request, supabase)


@router.get("/conversations/{conversation_id}", response_model=List[ChatMessage])
async def get_conversation(
    conversation_id: str,
    supabase = Depends(get_supabase_client)
):
    """
    Get a conversation by ID.
    
    Note: Conversation history is not currently being stored in the database.
    This endpoint is a placeholder for future functionality.
    
    Args:
        conversation_id: The ID of the conversation to retrieve
        
    Returns:
        List[ChatMessage]: The messages in the conversation
    """
    # Return a clear message that conversations are not being stored
    raise HTTPException(
        status_code=404, 
        detail="Conversation history is not being stored. The application is currently configured to use only document embeddings in Supabase."
    ) 