"""
Supabase adapter for vector storage and retrieval.
"""
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import numpy as np
from supabase import create_client, Client
from pgvector.psycopg import register_vector
import time
import random
import logging

logger = logging.getLogger(__name__)

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
    
    def get_document_by_filepath(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Get a document by its filepath."""
        response = self.supabase.table("documents").select("*").eq("filepath", filepath).execute()
        if response.data:
            return response.data[0]
        return None
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a chunk by its ID."""
        response = self.supabase.table("chunks").select("*").eq("id", chunk_id).execute()
        if response.data:
            return response.data[0]
        return None
    
    def add_chunks_batch(self, chunks_data: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple chunks with their embeddings to the database in a single batch.
        
        Args:
            chunks_data: List of dictionaries containing document_id, content, metadata, and embedding
                
        Returns:
            List of chunk IDs
        """
        if not chunks_data:
            return []
        
        # Retry parameters
        max_retries = 3
        retry_delay = 1  # Starting delay in seconds
        
        for attempt in range(max_retries):
            try:
                response = self.supabase.table("chunks").insert(chunks_data).execute()
                return [chunk["id"] for chunk in response.data]
            except Exception as e:
                if attempt < max_retries - 1:  # Don't sleep after the last attempt
                    # Calculate the exponential backoff with jitter
                    jitter = random.uniform(0, 0.1 * retry_delay)
                    sleep_time = retry_delay + jitter
                    
                    logger.warning(f"Batch insert attempt {attempt + 1} failed: {str(e)}. "
                                   f"Retrying in {sleep_time:.2f} seconds...")
                    
                    time.sleep(sleep_time)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Last attempt failed
                    logger.error(f"All {max_retries} batch insert attempts failed: {str(e)}")
                    raise  # Re-raise the last exception 