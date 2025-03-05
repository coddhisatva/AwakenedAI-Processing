"""
Supabase adapter for vector storage and retrieval.
"""
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import numpy as np
from supabase import create_client, Client
from pgvector.psycopg import register_vector

load_dotenv()

class SupabaseAdapter:
    """Adapter for storing and retrieving vectors from Supabase with pgvector."""
    
    def __init__(self):
        """Initialize the Supabase client."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
    def add_document(self, title: str, author: str, filepath: str) -> str:
        """
        Add a document to the database.
        
        Args:
            title: Document title
            author: Document author
            filepath: Path to the document file
            
        Returns:
            The document ID
        """
        response = self.supabase.table("documents").insert({
            "title": title,
            "author": author,
            "filepath": filepath
        }).execute()
        
        return response.data[0]["id"]
    
    def add_chunk(self, document_id: str, content: str, metadata: Dict[str, Any], embedding: List[float]) -> str:
        """
        Add a chunk with its embedding to the database.
        
        Args:
            document_id: The document ID this chunk belongs to
            content: The text content of the chunk
            metadata: Metadata associated with the chunk
            embedding: Vector embedding of the chunk
            
        Returns:
            The chunk ID
        """
        response = self.supabase.table("chunks").insert({
            "document_id": document_id,
            "content": content,
            "metadata": metadata,
            "embedding": embedding
        }).execute()
        
        return response.data[0]["id"]
    
    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query embedding.
        
        Args:
            query_embedding: The embedding of the query
            limit: Maximum number of results to return
            
        Returns:
            List of chunks with their similarity scores
        """
        # Execute raw SQL for vector similarity search
        response = self.supabase.rpc(
            "match_chunks", 
            {
                "query_embedding": query_embedding,
                "match_count": limit
            }
        ).execute()
        
        return response.data
    
    def count_documents(self) -> int:
        """Get the total number of documents in the database."""
        response = self.supabase.table("documents").select("id", count="exact").execute()
        return response.count
    
    def count_chunks(self) -> int:
        """Get the total number of chunks in the database."""
        response = self.supabase.table("chunks").select("id", count="exact").execute()
        return response.count
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID."""
        response = self.supabase.table("documents").select("*").eq("id", document_id).execute()
        if response.data:
            return response.data[0]
        return None
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a chunk by its ID."""
        response = self.supabase.table("chunks").select("*").eq("id", chunk_id).execute()
        if response.data:
            return response.data[0]
        return None 