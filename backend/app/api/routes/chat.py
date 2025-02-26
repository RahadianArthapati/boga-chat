"""
Chat API endpoints for Boga Chat.
"""
from typing import Dict, List, Optional, Any
import logging
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.langchain.chains import get_chat_chain, get_streaming_chat_chain
from app.langchain.rag import process_with_rag, stream_with_rag
from app.db.supabase import get_supabase_client

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
    use_rag: bool = False


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    conversation_id: str
    documents: Optional[List[Dict[str, Any]]] = None


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
        
        # Process the chat request with or without RAG
        if request.use_rag:
            logger.info(f"Processing RAG chat request with {len(formatted_messages)} messages")
            result = await process_with_rag(
                messages=formatted_messages,
                conversation_id=request.conversation_id
            )
            documents = result.get("documents", [])
        else:
            # Get the chat chain
            chat_chain = get_chat_chain()
            
            logger.info(f"Processing standard chat request with {len(formatted_messages)} messages")
            result = chat_chain({
                "messages": formatted_messages,
                "conversation_id": request.conversation_id
            })
            documents = None
            
        logger.info(f"Chat result: {result}")
        
        # Get conversation ID from request or result
        conversation_id = request.conversation_id or result.get("conversation_id")
        
        # No need to save to Supabase conversations table
        
        return ChatResponse(
            response=result["response"],
            conversation_id=conversation_id,
            documents=documents
        )
    
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


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
        
        logger.info(f"Processing streaming chat request with {len(formatted_messages)} messages")
        
        async def event_generator():
            full_response = ""
            conversation_id = request.conversation_id
            documents = None
            
            try:
                if request.use_rag:
                    # Use RAG streaming
                    async for chunk_data in stream_with_rag(
                        messages=formatted_messages,
                        conversation_id=conversation_id
                    ):
                        chunk = chunk_data["chunk"]
                        full_response = chunk_data["full_response"]
                        conversation_id = chunk_data["conversation_id"]
                        documents = chunk_data.get("documents")
                        
                        # Yield the chunk as a server-sent event
                        yield {
                            "event": "chunk",
                            "data": json.dumps({
                                "chunk": chunk,
                                "conversation_id": conversation_id,
                                "documents": documents,
                                "done": False
                            })
                        }
                else:
                    # Use standard streaming
                    async for chunk_data in get_streaming_chat_chain(
                        messages=formatted_messages,
                        conversation_id=conversation_id
                    ):
                        chunk = chunk_data["chunk"]
                        full_response = chunk_data["full_response"]
                        conversation_id = chunk_data["conversation_id"]
                        
                        # Yield the chunk as a server-sent event
                        yield {
                            "event": "chunk",
                            "data": json.dumps({
                                "chunk": chunk,
                                "conversation_id": conversation_id,
                                "documents": None,
                                "done": False
                            })
                        }
                
                # Final chunk with done=True
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "full_response": full_response,
                        "conversation_id": conversation_id,
                        "documents": documents,
                        "done": True
                    })
                }
                
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}", exc_info=True)
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "full_response": full_response + f"\n\nError: {str(e)}",
                        "conversation_id": conversation_id,
                        "documents": documents,
                        "done": True
                    })
                }
        
        return EventSourceResponse(event_generator())
    
    except Exception as e:
        logger.error(f"Streaming chat processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Streaming chat processing error: {str(e)}")


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
    
    Args:
        conversation_id: The ID of the conversation to retrieve
        
    Returns:
        List[ChatMessage]: The messages in the conversation
    """
    try:
        # Since we're not using the conversations table, return an empty list or error
        # This endpoint can be modified later if conversation persistence is needed
        raise HTTPException(status_code=404, detail="Conversation history not available")
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}") 