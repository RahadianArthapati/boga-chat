-- Enable the pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the document_embeddings table if it doesn't exist
CREATE TABLE IF NOT EXISTS document_embeddings (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_document_embeddings_document_id ON document_embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_chunk_id ON document_embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_created_at ON document_embeddings(created_at);

-- Create a function to match documents based on vector similarity
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(1536),
    similarity_threshold FLOAT,
    match_count INT
)
RETURNS TABLE (
    id BIGINT,
    document_id TEXT,
    chunk_id TEXT,
    chunk_text TEXT,
    similarity FLOAT,
    metadata JSONB,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        document_embeddings.id,
        document_embeddings.document_id,
        document_embeddings.chunk_id,
        document_embeddings.chunk_text,
        1 - (document_embeddings.embedding <=> query_embedding) AS similarity,
        document_embeddings.metadata,
        document_embeddings.created_at
    FROM document_embeddings
    WHERE 1 - (document_embeddings.embedding <=> query_embedding) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- Create a function to get all documents
CREATE OR REPLACE FUNCTION get_all_documents()
RETURNS TABLE (
    document_id TEXT,
    title TEXT,
    author TEXT,
    source TEXT,
    date TEXT,
    chunk_count INT,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        de.document_id,
        de.metadata->>'title' AS title,
        de.metadata->>'author' AS author,
        de.metadata->>'source' AS source,
        de.metadata->>'date' AS date,
        COUNT(de.chunk_id)::INT AS chunk_count,
        MIN(de.created_at) AS created_at
    FROM document_embeddings de
    GROUP BY de.document_id, de.metadata->>'title', de.metadata->>'author', de.metadata->>'source', de.metadata->>'date'
    ORDER BY MIN(de.created_at) DESC;
END;
$$; 