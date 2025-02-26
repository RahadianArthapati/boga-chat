"""
LangChain components for Boga Chat.
"""
import uuid
from typing import Dict, List, Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.db.supabase import get_supabase_client


def get_chat_model():
    """
    Get the chat model for the application.
    
    Returns:
        ChatOpenAI: A configured chat model
    """
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY
    )


def get_chat_prompt():
    """
    Get the chat prompt template.
    
    Returns:
        ChatPromptTemplate: A configured chat prompt template
    """
    template = """
    You are Boga, a helpful and friendly AI assistant. 
    
    Current conversation:
    {chat_history}
    
    Human: {input}
    AI: 
    """
    return ChatPromptTemplate.from_template(template)


def format_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Format messages for the chat model.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        
    Returns:
        List of formatted messages
    """
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_messages.append(AIMessage(content=msg["content"]))
    return formatted_messages


def get_chat_chain():
    """
    Get the chat chain for processing chat requests.
    
    Returns:
        A function that processes chat requests
    """
    chat_model = get_chat_model()
    prompt = get_chat_prompt()
    
    def process_chat(inputs: Dict[str, Any]) -> Dict[str, Any]:
        messages = inputs.get("messages", [])
        conversation_id = inputs.get("conversation_id")
        
        # If no conversation_id, create a new one
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Extract the last user message
        last_user_message = next((m["content"] for m in reversed(messages) 
                                if m["role"] == "user"), "")
        
        # Format previous messages for chat history
        previous_messages = messages[:-1] if messages else []
        chat_history = "\n".join([
            f"{'Human' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in previous_messages
        ])
        
        # Process with the chain
        chain = prompt | chat_model
        response = chain.invoke({
            "chat_history": chat_history,
            "input": last_user_message
        })
        
        # Handle different response formats
        response_content = ""
        if hasattr(response, 'content'):
            response_content = response.content
        elif isinstance(response, dict) and 'text' in response:
            response_content = response['text']
        elif isinstance(response, str):
            response_content = response
        else:
            # Try to convert to string as a fallback
            response_content = str(response)
        
        return {
            "response": response_content,
            "conversation_id": conversation_id
        }
    
    return process_chat 