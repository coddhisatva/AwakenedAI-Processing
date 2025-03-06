-- This SQL script sets up the tables and indexes for the Awakened AI project in Supabase
-- Run this in the Supabase SQL Editor before using the system

-- Enable pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table to store document metadata
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    author TEXT,
    filepath TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on filepath for faster lookups
CREATE INDEX IF NOT EXISTS idx_documents_filepath ON documents(filepath);

-- Chunks table to store text chunks with their embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(1536),  -- OpenAI Ada-002 embeddings are 1536 dimensions
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on document_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- Create vector index for similarity search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Function for similarity search
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 5
) RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    document_id UUID,
    similarity FLOAT
) LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        chunks.id,
        chunks.content,
        chunks.metadata,
        chunks.document_id,
        1 - (chunks.embedding <=> query_embedding) AS similarity
    FROM chunks
    ORDER BY chunks.embedding <=> query_embedding
    LIMIT match_count;
END;
$$; 