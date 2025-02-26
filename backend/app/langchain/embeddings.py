"""
Document embeddings module for Boga Chat.
This module handles the creation, storage, and retrieval of document embeddings
using OpenAI embeddings and Supabase pgvector.
"""
import uuid
from typing import Dict, List, Any, Optional, Union
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.db.supabase import get_supabase_client

logger = logging.getLogger(__name__)


def get_embeddings_model():
    """
    Get the embeddings model for the application.
    
    Returns:
        OpenAIEmbeddings: A configured embeddings model
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.OPENAI_API_KEY
    )


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split text into chunks for embedding.
    
    Args:
        text: The text to split
        chunk_size: The size of each chunk
        chunk_overlap: The overlap between chunks
        
    Returns:
        List of Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    return text_splitter.create_documents([text])


async def store_document_embeddings(
    document_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Store document embeddings in Supabase.
    
    Args:
        document_id: The ID of the document
        text: The text to embed
        metadata: Optional metadata for the document
        chunk_size: The size of each chunk
        chunk_overlap: The overlap between chunks
        
    Returns:
        List of chunk IDs
    """
    try:
        # Split text into chunks
        chunks = split_text(text, chunk_size, chunk_overlap)
        
        # Get embeddings model
        embeddings_model = get_embeddings_model()
        
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Store each chunk with its embedding
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = embeddings_model.embed_query(chunk.page_content)
            
            # Generate chunk ID
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            # Prepare metadata
            chunk_metadata = metadata or {}
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            
            # Store in Supabase
            supabase.table("document_embeddings").insert({
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_text": chunk.page_content,
                "embedding": embedding,
                "metadata": chunk_metadata
            }).execute()
            
        logger.info(f"Stored {len(chunks)} embeddings for document {document_id}")
        return chunk_ids
        
    except Exception as e:
        logger.error(f"Error storing document embeddings: {str(e)}", exc_info=True)
        raise


async def search_documents(
    query: str,
    limit: int = 5,
    similarity_threshold: float = 0.7,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents using vector similarity.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0-1)
        metadata_filter: Optional filter for metadata fields
        
    Returns:
        List of matching documents with similarity scores
    """
    try:
        # Get embeddings model
        embeddings_model = get_embeddings_model()
        
        # Generate query embedding
        query_embedding = embeddings_model.embed_query(query)
        
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Since the match_documents function might not exist, use a direct query
        # with the cosine similarity calculation
        response = supabase.table("document_embeddings") \
            .select("*") \
            .execute()
            
        if not response.data:
            return []
            
        # Calculate similarity manually
        results = []
        for item in response.data:
            # Skip items without embeddings
            if not item.get("embedding"):
                continue
                
            # Calculate cosine similarity
            embedding = item["embedding"]
            similarity = 1 - cosine_distance(query_embedding, embedding)
            
            # Apply similarity threshold
            if similarity < similarity_threshold:
                continue
                
            # Apply metadata filter if provided
            if metadata_filter:
                item_metadata = item.get("metadata", {})
                if not all(item_metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
            
            # Add to results
            results.append({
                "document_id": item["document_id"],
                "chunk_id": item["chunk_id"],
                "chunk_text": item["chunk_text"],
                "similarity": similarity,
                "metadata": item.get("metadata", {})
            })
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[:limit]
            
        return results
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}", exc_info=True)
        raise


# Helper function to calculate cosine distance
def cosine_distance(vec1, vec2):
    """
    Calculate cosine distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine distance (0-2, where 0 is identical)
    """
    import numpy as np
    
    # Convert to numpy arrays
    # Handle string embeddings by parsing them
    if isinstance(vec1, str):
        try:
            # Remove brackets and split by commas
            vec1 = vec1.strip('[]').split(',')
            vec1 = np.array([float(x.strip()) for x in vec1])
        except Exception:
            # If parsing fails, return maximum distance
            return 1.0
    else:
        vec1 = np.array(vec1)
        
    if isinstance(vec2, str):
        try:
            # Remove brackets and split by commas
            vec2 = vec2.strip('[]').split(',')
            vec2 = np.array([float(x.strip()) for x in vec2])
        except Exception:
            # If parsing fails, return maximum distance
            return 1.0
    else:
        vec2 = np.array(vec2)
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 1.0
        
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    
    # Convert to distance (0-2)
    return 1.0 - cosine_similarity


async def get_document_by_id(document_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all chunks for a specific document.
    
    Args:
        document_id: The ID of the document to retrieve
        
    Returns:
        List of document chunks
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Query for all chunks of the document
        response = supabase.table("document_embeddings") \
            .select("*") \
            .eq("document_id", document_id) \
            .order("metadata->chunk_index", desc=False) \
            .execute()
        
        if not response.data:
            return []
            
        return response.data
        
    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}", exc_info=True)
        raise 