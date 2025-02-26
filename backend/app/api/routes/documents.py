"""
Document API endpoints for Boga Chat.
"""
from typing import Dict, List, Optional, Any
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from app.langchain.embeddings import (
    store_document_embeddings,
    search_documents,
    get_document_by_id
)
from app.db.supabase import get_supabase_client

router = APIRouter()
logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    title: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None
    tags: Optional[List[str]] = None
    additional: Optional[Dict[str, Any]] = None


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    document_id: str
    chunk_count: int
    metadata: Optional[DocumentMetadata] = None


class DocumentSearchRequest(BaseModel):
    """Document search request model."""
    query: str
    limit: int = 5
    similarity_threshold: float = Field(0.45, ge=0.0, le=1.0)
    metadata_filter: Optional[Dict[str, Any]] = None


class DocumentChunk(BaseModel):
    """Document chunk model."""
    document_id: str
    chunk_id: str
    chunk_text: str
    similarity: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentSearchResponse(BaseModel):
    """Document search response model."""
    results: List[DocumentChunk]


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    supabase = Depends(get_supabase_client)
):
    """
    Upload a document and create embeddings.
    
    Args:
        file: The document file to upload
        title: Optional document title
        author: Optional document author
        source: Optional document source
        tags: Optional comma-separated tags
        
    Returns:
        DocumentUploadResponse: The document ID and chunk count
    """
    try:
        # Read file content
        content = await file.read()
        text = content.decode("utf-8")
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Prepare metadata
        metadata = DocumentMetadata(
            title=title or file.filename,
            author=author,
            source=source,
            tags=tags.split(",") if tags else None
        )
        
        # Store document embeddings
        chunk_ids = await store_document_embeddings(
            document_id=document_id,
            text=text,
            metadata=metadata.dict(exclude_none=True)
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            chunk_count=len(chunk_ids),
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")


@router.post("/search", response_model=DocumentSearchResponse)
async def search_document_embeddings(
    request: DocumentSearchRequest,
    supabase = Depends(get_supabase_client)
):
    """
    Search for documents using vector similarity.
    
    Args:
        request: The search request
        
    Returns:
        DocumentSearchResponse: The search results
    """
    try:
        # Search documents
        results = await search_documents(
            query=request.query,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
            metadata_filter=request.metadata_filter
        )
        
        # Format results
        formatted_results = [
            DocumentChunk(
                document_id=result["document_id"],
                chunk_id=result["chunk_id"],
                chunk_text=result["chunk_text"],
                similarity=result["similarity"],
                metadata=result["metadata"]
            )
            for result in results
        ]
        
        return DocumentSearchResponse(results=formatted_results)
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")


@router.get("/{document_id}", response_model=List[DocumentChunk])
async def get_document(
    document_id: str,
    supabase = Depends(get_supabase_client)
):
    """
    Retrieve a document by ID.
    
    Args:
        document_id: The ID of the document to retrieve
        
    Returns:
        List[DocumentChunk]: The document chunks
    """
    try:
        # Get document
        chunks = await get_document_by_id(document_id)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Format chunks
        formatted_chunks = [
            DocumentChunk(
                document_id=chunk["document_id"],
                chunk_id=chunk["chunk_id"],
                chunk_text=chunk["chunk_text"],
                metadata=chunk.get("metadata")
            )
            for chunk in chunks
        ]
        
        return formatted_chunks
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    supabase = Depends(get_supabase_client)
):
    """
    Delete a document by ID.
    
    Args:
        document_id: The ID of the document to delete
        
    Returns:
        Dict: Success message
    """
    try:
        # Delete document chunks
        response = supabase.table("document_embeddings") \
            .delete() \
            .eq("document_id", document_id) \
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}") 