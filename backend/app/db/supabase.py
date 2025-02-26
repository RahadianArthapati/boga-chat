"""
Supabase connection module for Boga Chat.
"""
from functools import lru_cache

import supabase
from supabase.client import Client

from app.config import settings


@lru_cache()
def get_supabase_client() -> Client:
    """
    Create and return a Supabase client.
    
    Returns:
        Client: A Supabase client instance
    """
    client = supabase.create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_KEY
    )
    return client


def setup_supabase_tables():
    """
    Set up the necessary tables in Supabase if they don't exist.
    This should be called during application startup.
    """
    client = get_supabase_client()
    
    # Check if we can connect to Supabase
    try:
        # This is a simple query to check connection
        client.table("conversations").select("id").limit(1).execute()
        print("Successfully connected to Supabase")
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        print("Make sure the tables are created in Supabase with the correct schema:")
        print("""
        -- Create conversations table
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY,
            messages JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create vector extension if not exists
        CREATE EXTENSION IF NOT EXISTS vector;
        
        -- Create embeddings table for vector search
        CREATE TABLE IF NOT EXISTS embeddings (
            id UUID PRIMARY KEY,
            content TEXT NOT NULL,
            embedding VECTOR(1536),
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """)
        raise 