"""
Supabase adapter for vector storage and retrieval.
"""
import os
import time
import random
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import numpy as np
from supabase import create_client, Client
from pgvector.psycopg import register_vector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        
    def add_document(self, title: str, author: str, filepath: str, batch_id: str = None) -> str:
        """
        Add a document to the database.
        
        Args:
            title: Document title
            author: Document author
            filepath: Path to the document file
            batch_id: Identifier for the processing batch
            
        Returns:
            The document ID
        """
        document_data = {
            "title": title,
            "author": author,
            "filepath": filepath
        }
        
        # Add batch_id if provided
        if batch_id:
            document_data["batch_id"] = batch_id
        
        response = self.supabase.table("documents").insert(document_data).execute()
        
        return response.data[0]["id"]
    
    def add_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple chunks with their embeddings to the database in a single batch operation.
        
        Args:
            chunks: List of chunk dictionaries, each containing:
                - document_id: The document ID this chunk belongs to
                - content: The text content of the chunk
                - metadata: Metadata associated with the chunk
                - embedding: Vector embedding of the chunk
            
        Returns:
            List of chunk IDs
        """
        if not chunks:
            return []
        
        # Prepare the data for insertion
        chunk_data = []
        for chunk in chunks:
            chunk_data.append({
                "document_id": chunk["document_id"],
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "embedding": chunk["embedding"]
            })
        
        # Implement retry with exponential backoff
        max_retries = 5
        base_delay = 1  # starting delay in seconds
        chunk_ids = []
        
        for attempt in range(max_retries):
            try:
                # Execute batch insert with transaction
                response = self.supabase.table("chunks").insert(chunk_data).execute()
                
                # Extract the IDs
                chunk_ids = [item["id"] for item in response.data]
                
                # If successful, break the retry loop
                break
                
            except Exception as e:
                # Calculate delay with exponential backoff and some randomness
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                
                # If this is the final attempt, re-raise the exception
                if attempt == max_retries - 1:
                    logger.error(f"Failed to insert batch after {max_retries} attempts: {str(e)}")
                    raise
                
                logger.warning(f"Batch insert attempt {attempt+1} failed: {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        
        return chunk_ids
    
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