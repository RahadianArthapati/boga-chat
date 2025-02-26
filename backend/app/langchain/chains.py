"""
LangChain components for Boga Chat.
"""
import uuid
import logging
from typing import Dict, List, Any, AsyncIterator, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.memory import BaseMemory
from langchain.chains import ConversationChain, LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import Field

from app.config import settings
from app.db.supabase import get_supabase_client

# Dictionary to store conversation chains, shared between all functions
conversation_chains = {}


class CustomConversationMemory(BaseMemory):
    """Custom memory class that handles both chat history and context."""
    
    chat_history: str = Field(default="")
    context: str = Field(default="")
        
    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables."""
        return ["chat_history", "context"]
        
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables."""
        return {
            "chat_history": self.chat_history,
            "context": self.context
        }
        
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to memory."""
        if "input" in inputs:
            human = f"Human: {inputs['input']}\n"
            ai = f"AI: {outputs['response']}\n"
            self.chat_history = self.chat_history + human + ai
            
    def update_context(self, new_context: str) -> None:
        """Update the context."""
        self.context = new_context
        
    def clear(self) -> None:
        """Clear memory."""
        self.chat_history = ""
        self.context = ""


def get_chat_model(streaming: bool = False):
    """
    Get the chat model for the application.
    
    Args:
        streaming: Whether to enable streaming for the model
        
    Returns:
        ChatOpenAI: A configured chat model
    """
    logging.info(f"Initializing chat model with streaming={streaming}")
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY,
        streaming=streaming
    )


def get_chat_prompt():
    """
    Get the chat prompt template.
    
    Returns:
        ChatPromptTemplate: A configured chat prompt template
    """
    template = """
    You are a helpful and friendly AI assistant. 

    {context}
    
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
    logging.info("Initializing chat chain")
    chat_model = get_chat_model()
    prompt = get_chat_prompt()
    embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    
    # Using global conversation_chains - no longer creating a new dictionary
    # conversation_chains = {}
    
    def process_chat(inputs: Dict[str, Any]) -> Dict[str, Any]:
        global conversation_chains
        messages = inputs.get("messages", [])
        conversation_id = inputs.get("conversation_id")
        
        logging.info(f"Processing chat request. Conversation ID: {conversation_id}")
        
        # If no conversation_id, create a new one
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logging.info(f"Created new conversation ID: {conversation_id}")
        
        # Get relevant documents from Supabase
        supabase = get_supabase_client()
        last_user_message = next((m["content"] for m in reversed(messages) 
                                if m["role"] == "user"), "")
        
        # Get embedding for the query
        query_embedding = embeddings.embed_query(last_user_message)
        
        # Query document embeddings
        response = supabase.rpc(
            'match_documents',
            params={
                'query_embedding': query_embedding,
                'match_threshold': 0.8,
                'match_count': 3
            }
        ).execute()
        
        relevant_docs = ""
        if response.data:
            relevant_docs = "\n".join([doc['chunk_text'] for doc in response.data])
            logging.info(f"Found {len(response.data)} relevant documents")
        
        # Get or create conversation chain for this conversation_id
        if conversation_id not in conversation_chains:
            # Format previous messages for initial chat history
            previous_messages = messages[:-1] if messages else []
            initial_chat_history = "\n".join([
                f"{'Human' if m['role'] == 'user' else 'AI'}: {m['content']}"
                for m in previous_messages
            ])
            
            # Create custom memory with both chat history and context
            memory = CustomConversationMemory(
                chat_history=initial_chat_history if initial_chat_history else "",
                context=relevant_docs
            )
            
            conversation_chains[conversation_id] = {
                'chain': ConversationChain(
                    llm=chat_model,
                    memory=memory,
                    prompt=prompt,
                    verbose=True
                ),
                'context_docs': [doc['chunk_text'] for doc in response.data] if response.data else []
            }
            logging.info(f"Created new conversation chain for ID: {conversation_id}")
            logging.info(f"Initial context length: {len(relevant_docs)}")
        else:
            # For existing conversations, update the context documents
            if response.data:
                logging.info(f"Adding new docs to existing conversation {conversation_id}")
                new_docs = [doc['chunk_text'] for doc in response.data]
                existing_docs = conversation_chains[conversation_id].get('context_docs', [])
                
                # Add new documents that aren't already in the context
                for doc in new_docs:
                    if doc not in existing_docs:
                        existing_docs.append(doc)
                
                # Update the stored context documents
                conversation_chains[conversation_id]['context_docs'] = existing_docs
                
                # Build the combined context
                relevant_docs = "\n".join(existing_docs)
                logging.info(f"Using combined context with {len(existing_docs)} documents")
        
        chain_data = conversation_chains[conversation_id]
        chain = chain_data['chain']
        
        # Log context size
        context_docs = chain_data.get('context_docs', [])
        logging.info(f"Context contains {len(context_docs)} documents")
        
        # Update context for this interaction
        if isinstance(chain.memory, CustomConversationMemory):
            new_context = "\n".join(context_docs) if context_docs else ""
            chain.memory.update_context(new_context)
            logging.info(f"Updated memory context, length: {len(new_context)}")
        
        # Process with the chain
        response = chain({"input": last_user_message})
        
        logging.info("Successfully generated response")
        
        return {
            "response": response['response'],  # Chain returns a dict with 'response' key
            "conversation_id": conversation_id
        }
    
    return process_chat


async def get_streaming_chat_chain(
    messages: List[Dict[str, str]],
    conversation_id: Optional[str] = None
) -> AsyncIterator[Dict[str, Any]]:
    """
    Get a streaming chat chain that yields response chunks.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        conversation_id: Optional conversation ID
        
    Yields:
        Dict with response chunk and conversation_id
    """
    global conversation_chains
    logging.info(f"Initializing streaming chat chain. Conversation ID: {conversation_id}")
    
    # If no conversation_id, create a new one
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        logging.info(f"Created new conversation ID: {conversation_id}")
    
    # Extract the last user message
    last_user_message = next((m["content"] for m in reversed(messages) 
                            if m["role"] == "user"), "")
    
    # Format previous messages for chat history
    previous_messages = messages[:-1] if messages else []
    chat_history = "\n".join([
        f"{'Human' if m['role'] == 'user' else 'AI'}: {m['content']}"
        for m in previous_messages
    ])
    logging.info(f"Chat history: {chat_history}")
    logging.info(f"Last user message: {last_user_message}")
    
    # Get context from existing conversation if available
    context = ""
    if conversation_id in conversation_chains:
        context_docs = conversation_chains[conversation_id].get('context_docs', [])
        if context_docs:
            context = "\n".join(context_docs)
            logging.info(f"Retrieved context with {len(context_docs)} documents for streaming")
    
    # Get streaming model and prompt
    chat_model = get_chat_model(streaming=True)
    prompt = get_chat_prompt()
    
    # Create streaming chain
    chain = prompt | chat_model | StrOutputParser()
    
    # Process with streaming
    full_response = ""
    logging.debug("Starting streaming response generation")
    async for chunk in chain.astream({
        "chat_history": chat_history,
        "input": last_user_message,
        "context": context  # Use the retrieved context
    }):
        full_response += chunk
        logging.debug(f"Generated chunk of length: {len(chunk)}")
        yield {
            "chunk": chunk,
            "full_response": full_response,
            "conversation_id": conversation_id
        } 