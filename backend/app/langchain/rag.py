"""
RAG (Retrieval Augmented Generation) module for Boga Chat.
This module connects the chat functionality with document embeddings
to provide document-augmented responses.
"""
import logging
from typing import Dict, List, Any, Optional, AsyncIterator

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.langchain.embeddings import search_documents

logger = logging.getLogger(__name__)


async def get_relevant_documents(query: str, limit: int = 3, threshold: float = 0.45) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents for a query.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        threshold: Minimum similarity threshold (default lowered to 0.45 based on testing)
        
    Returns:
        List of relevant document chunks
    """
    try:
        results = await search_documents(
            query=query,
            limit=limit,
            similarity_threshold=threshold
        )
        
        return results
    except Exception as e:
        logger.error(f"Error retrieving relevant documents: {str(e)}", exc_info=True)
        return []


def format_documents_for_prompt(documents: List[Dict[str, Any]]) -> str:
    """
    Format document chunks for inclusion in the prompt.
    
    Args:
        documents: List of document chunks
        
    Returns:
        Formatted string of document content
    """
    if not documents:
        return "No relevant documents found."
    
    formatted_docs = []
    for i, doc in enumerate(documents):
        metadata = doc.get("metadata", {})
        title = metadata.get("title", "Untitled Document")
        source = metadata.get("source", "Unknown Source")
        
        formatted_doc = f"Document {i+1}: {title} (Source: {source})\n"
        formatted_doc += f"Content: {doc['chunk_text']}\n"
        formatted_docs.append(formatted_doc)
    
    return "\n\n".join(formatted_docs)


def get_rag_prompt():
    """
    Get the RAG prompt template.
    
    Returns:
        ChatPromptTemplate: A configured RAG prompt template
    """
    template = """
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
    return ChatPromptTemplate.from_template(template)


async def get_rag_chain(streaming: bool = False):
    """
    Get the RAG chain for processing chat requests with document retrieval.
    
    Args:
        streaming: Whether to enable streaming for the model
        
    Returns:
        A function that processes chat requests with document retrieval
    """
    chat_model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY,
        streaming=streaming
    )
    
    prompt = get_rag_prompt()
    
    # Create the chain
    chain = prompt | chat_model | StrOutputParser()
    
    return chain


async def process_with_rag(
    messages: List[Dict[str, str]],
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a chat request using RAG.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        conversation_id: Optional conversation ID
        
    Returns:
        Dict with response and conversation_id
    """
    try:
        # Extract the last user message
        last_user_message = next((m["content"] for m in reversed(messages) 
                                if m["role"] == "user"), "")
        
        # Format previous messages for chat history
        previous_messages = messages[:-1] if messages else []
        chat_history = "\n".join([
            f"{'Human' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in previous_messages
        ])
        
        # Retrieve relevant documents
        relevant_docs = await get_relevant_documents(last_user_message)
        formatted_docs = format_documents_for_prompt(relevant_docs)
        
        # Get the RAG chain
        chain = await get_rag_chain()
        
        # Process with the chain
        response = await chain.ainvoke({
            "chat_history": chat_history,
            "documents": formatted_docs,
            "input": last_user_message
        })
        
        return {
            "response": response,
            "conversation_id": conversation_id or "new_conversation",
            "documents": relevant_docs
        }
        
    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}", exc_info=True)
        raise


async def stream_with_rag(
    messages: List[Dict[str, str]],
    conversation_id: Optional[str] = None
) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream a chat response using RAG.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        conversation_id: Optional conversation ID
        
    Yields:
        Dict with response chunk and conversation_id
    """
    try:
        # Extract the last user message
        last_user_message = next((m["content"] for m in reversed(messages) 
                                if m["role"] == "user"), "")
        
        # Format previous messages for chat history
        previous_messages = messages[:-1] if messages else []
        chat_history = "\n".join([
            f"{'Human' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in previous_messages
        ])
        
        # Retrieve relevant documents
        relevant_docs = await get_relevant_documents(last_user_message)
        formatted_docs = format_documents_for_prompt(relevant_docs)
        
        # Get the streaming RAG chain
        chain = await get_rag_chain(streaming=True)
        
        # Process with streaming
        full_response = ""
        async for chunk in chain.astream({
            "chat_history": chat_history,
            "documents": formatted_docs,
            "input": last_user_message
        }):
            full_response += chunk
            yield {
                "chunk": chunk,
                "full_response": full_response,
                "conversation_id": conversation_id or "new_conversation",
                "documents": relevant_docs
            }
            
    except Exception as e:
        logger.error(f"Error streaming with RAG: {str(e)}", exc_info=True)
        raise 