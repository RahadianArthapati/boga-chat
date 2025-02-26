#!/usr/bin/env python
"""
Test script for querying document embeddings.
This script tests the retrieval of relevant documents from the vector database.
"""
import asyncio
from app.langchain.rag import get_relevant_documents
from app.db.supabase import get_supabase_client

async def test_document_query(query: str):
    """
    Test document query functionality with a given query.
    
    Args:
        query: The search query to test
    """
    print(f"Testing query: {query}")
    
    # Updated threshold based on test results
    # Most relevant documents had scores between 0.45-0.53
    documents = await get_relevant_documents(
        query=query,
        limit=5,  # Keeping increased limit to see more results
        threshold=0.45  # Optimized threshold based on test results
    )
    
    print(f"\nFound {len(documents)} relevant documents:")
    for i, doc in enumerate(documents, 1):
        print(f"\n--- Document {i} ---")
        print(f"Similarity Score: {doc.get('similarity', 'N/A')}")
        print(f"Content: {doc.get('chunk_text', 'N/A')[:200]}...")  # First 200 chars
        print(f"Metadata: {doc.get('metadata', {})}")

# Example usage
async def main():
    """Run the test queries."""
    test_queries = [
        "apa saja brand dari boga?",
        "menu di shaburi apa saja?",
        "lokasi paradise dynasty dimana saja?"
    ]
    
    # Make sure Supabase connection is working
    try:
        client = get_supabase_client()
        response = client.table("document_embeddings").select("count").execute()
        count = len(response.data)
        print(f"Connected to Supabase. Found {count} document embeddings.")
    except Exception as e:
        print(f"Error connecting to Supabase: {str(e)}")
        print("Make sure your .env file is set up correctly with SUPABASE_URL and SUPABASE_KEY.")
        return
    
    print("\n" + "="*50)
    print("Starting document query tests")
    print("="*50 + "\n")
    
    for query in test_queries:
        await test_document_query(query)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main()) 