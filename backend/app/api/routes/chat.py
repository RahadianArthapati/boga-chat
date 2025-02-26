"""
Chat API endpoints for Boga Chat.
"""
from typing import Dict, List, Optional
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.langchain.chains import get_chat_chain
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


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    conversation_id: str


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
    try:
        # Get the chat chain
        chat_chain = get_chat_chain()
        
        # Format messages for the chain
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        # Process the chat request
        logger.info(f"Processing chat request with {len(formatted_messages)} messages")
        result = chat_chain({
            "messages": formatted_messages,
            "conversation_id": request.conversation_id
        })
        logger.info(f"Chat chain result: {result}")
        
        # Store conversation in Supabase if needed
        conversation_id = request.conversation_id or result.get("conversation_id")
        
        # Try to save to Supabase, but continue even if it fails
        if conversation_id:
            try:
                supabase.table("conversations").upsert({
                    "id": conversation_id,
                    "messages": formatted_messages + [{"role": "assistant", "content": result["response"]}]
                }).execute()
                logger.info(f"Saved conversation {conversation_id} to Supabase")
            except Exception as e:
                logger.warning(f"Failed to save conversation to Supabase: {str(e)}")
                # Continue without saving to Supabase
        
        return ChatResponse(
            response=result["response"],
            conversation_id=conversation_id
        )
    
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@router.get("/conversations/{conversation_id}", response_model=List[ChatMessage])
async def get_conversation(
    conversation_id: str,
    supabase = Depends(get_supabase_client)
):
    """
    Retrieve a conversation by ID.
    
    Args:
        conversation_id: The ID of the conversation to retrieve
        
    Returns:
        List[ChatMessage]: The messages in the conversation
    """
    try:
        try:
            response = supabase.table("conversations").select("messages").eq("id", conversation_id).execute()
            
            if not response.data:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            return [ChatMessage(**msg) for msg in response.data[0]["messages"]]
        except Exception as e:
            logger.warning(f"Failed to retrieve conversation from Supabase: {str(e)}")
            raise HTTPException(status_code=404, detail="Conversation not found or database error")
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}") 