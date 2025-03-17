"""
Supabase adapter for vector storage and retrieval.
"""
import os
import time
import random
import logging
import json
from pathlib import Path
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
    
    def __init__(self, manifest_path: Optional[str] = None):
        """Initialize the Supabase client."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Set manifest path if provided
        if manifest_path:
            self.manifest_path = Path(manifest_path)
        
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
        
        # Return document ID if successful
        if response.data and len(response.data) > 0:
            document_id = response.data[0]["id"]
            return document_id
        
        return None
    
    def update_manifest(self, filepath: str) -> None:
        """
        Update the manifest to indicate a document has been successfully processed.
        
        Args:
            filepath: Path to the file that was added to the database
        """
        try:
            # Load existing manifest
            if self.manifest_path.exists():
                with open(self.manifest_path, 'r') as f:
                    manifest = json.load(f)
            else:
                manifest = {}
            
            # Add the file to the manifest
            file_path = Path(filepath)
            manifest[file_path.name] = {
                "filepath": str(filepath),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "direct_processing"
            }
            
            # Write updated manifest
            with open(self.manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
                
            logger.info(f"Updated manifest with successfully processed document: {file_path.name}")
        except Exception as e:
            logger.error(f"Error updating manifest: {e}")
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks (via CASCADE constraint).
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            response = self.supabase.table("documents").delete().eq("id", document_id).execute()
            success = len(response.data) > 0
            if success:
                logger.info(f"Successfully deleted document with ID {document_id}")
            else:
                logger.warning(f"No document found with ID {document_id} to delete")
            return success
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
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