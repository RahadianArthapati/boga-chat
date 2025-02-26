"""
Query router module for Boga Chat.
This module handles the decision-making process for whether to use 
RAG (Retrieval Augmented Generation) for a given query.
"""
import logging
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Define the router prompt
ROUTER_PROMPT_TEMPLATE = """
You are a query router for a chat system with a document retrieval capability.
Your job is to analyze the user's query and decide whether to use document retrieval 
(RAG) to enhance the response or not.

Use document retrieval if:
1. The query is asking for specific information that might be in documents
2. The query refers to company/product details, data, or facts
3. The query is about specific processes, guidelines, or historical information
4. The query mentions "documents", "records", or specific document types

Do not use document retrieval if:
1. The query is a greeting or small talk
2. The query is asking for general opinions or creative content
3. The query is a follow-up that clearly relates to the conversation flow
4. The query is asking about capabilities of the chatbot itself

USER QUERY: {query}

RESPONSE STRICTLY AS JSON:
{{"use_rag": true/false, "reasoning": "Brief explanation of your decision"}}
"""

def get_router_model():
    """
    Get the routing model.
    
    Returns:
        ChatOpenAI: A configured chat model for routing decisions
    """
    return ChatOpenAI(
        model="gpt-4o-mini",  # Using a faster, cheaper model for routing decisions
        temperature=0,  # We want deterministic routing decisions
        api_key=settings.OPENAI_API_KEY
    )

def get_router_chain():
    """
    Get the router chain for deciding whether to use RAG.
    
    Returns:
        A chain that outputs a JSON with the routing decision
    """
    router_prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
    router_model = get_router_model()
    output_parser = JsonOutputParser()
    
    router_chain = router_prompt | router_model | output_parser
    return router_chain

async def should_use_rag(query: str) -> Dict[str, Any]:
    """
    Determine whether to use RAG for a given query.
    
    Args:
        query: The user's query
        
    Returns:
        Dict with the routing decision and reasoning
    """
    try:
        router_chain = get_router_chain()
        result = await router_chain.ainvoke({"query": query})
        logger.info(f"Router decision for query '{query[:30]}...': {result}")
        return result
    except Exception as e:
        logger.error(f"Error in routing decision: {str(e)}", exc_info=True)
        # Default to not using RAG if there's an error
        return {"use_rag": False, "reasoning": "Error in routing decision"} 