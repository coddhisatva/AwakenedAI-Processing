#!/usr/bin/env python3
"""
Migration script to transfer data from ChromaDB to Supabase.
"""
import os
import sys
import time
from typing import Dict, List, Any
from dotenv import load_dotenv

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.chroma_db import ChromaDBAdapter
from database.supabase_adapter import SupabaseAdapter

load_dotenv()

def extract_metadata_from_chroma(doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize metadata from ChromaDB format."""
    # Extract basic metadata
    title = doc_metadata.get("title", "Unknown")
    author = doc_metadata.get("author", "Unknown")
    filepath = doc_metadata.get("source", "")
    
    # Extract page numbers and other metadata to include in chunk metadata
    page_info = {}
    if "page" in doc_metadata:
        page_info["page"] = doc_metadata["page"]
    if "page_content" in doc_metadata:
        page_info["page_content"] = doc_metadata["page_content"]
        
    return {
        "document": {
            "title": title,
            "author": author,
            "filepath": filepath
        },
        "chunk": page_info
    }

def migrate_data():
    """Migrate data from ChromaDB to Supabase."""
    print("Starting migration from ChromaDB to Supabase...")
    
    # Initialize database adapters
    chroma_db = ChromaDBAdapter()
    supabase_db = SupabaseAdapter()
    
    # Get all documents from ChromaDB
    print("Retrieving data from ChromaDB...")
    collection = chroma_db.client.get_collection("documents")
    result = collection.get()
    
    documents = result["documents"]
    metadatas = result["metadatas"]
    embeddings = result["embeddings"]
    ids = result["ids"]
    
    print(f"Found {len(documents)} chunks to migrate")
    
    # Map to track which document IDs we've already created
    document_map = {}
    
    # Process each chunk
    for i, (doc_text, doc_metadata, doc_embedding, doc_id) in enumerate(zip(documents, metadatas, embeddings, ids)):
        # Extract metadata
        metadata = extract_metadata_from_chroma(doc_metadata)
        doc_info = metadata["document"]
        chunk_info = metadata["chunk"]
        
        # Create document if it doesn't exist
        doc_filepath = doc_info["filepath"]
        if doc_filepath not in document_map:
            print(f"Creating document record for: {doc_info['title']}")
            document_id = supabase_db.add_document(
                title=doc_info["title"],
                author=doc_info["author"],
                filepath=doc_filepath
            )
            document_map[doc_filepath] = document_id
        
        # Add chunk
        supabase_db.add_chunk(
            document_id=document_map[doc_filepath],
            content=doc_text,
            metadata=chunk_info,
            embedding=doc_embedding
        )
        
        # Print progress
        if (i + 1) % 100 == 0 or (i + 1) == len(documents):
            print(f"Migrated {i + 1}/{len(documents)} chunks")
    
    print("Migration complete!")
    print(f"Migrated {len(document_map)} documents and {len(documents)} chunks")

if __name__ == "__main__":
    start_time = time.time()
    migrate_data()
    end_time = time.time()
    print(f"Migration completed in {end_time - start_time:.2f} seconds") 