#!/usr/bin/env python3
"""
Check the contents of the Supabase vector database.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage.vector_store import SupabaseVectorStore

# Load environment variables
load_dotenv()

def main():
    # Initialize Supabase vector store
    store = SupabaseVectorStore()
    
    # First, get the chunks count
    count_response = store.supabase.table("chunks").select("id", count="exact").execute()
    chunks_count = count_response.count if hasattr(count_response, 'count') else 0
    
    print(f"Supabase collection count: {chunks_count} chunks")
    
    if chunks_count == 0:
        print("No chunks found in Supabase.")
        return
    
    # Collect unique filenames
    filenames = set()
    sources = set()
    
    # Process in batches to avoid memory issues
    batch_size = 100
    for i in range(0, chunks_count, batch_size):
        print(f"Processing chunks {i} to {min(i + batch_size, chunks_count)}...")
        
        # Get a batch of chunks with their metadata
        batch = store.supabase.table("chunks").select("metadata").range(i, i + batch_size - 1).execute()
        
        if not batch.data:
            continue
            
        for item in batch.data:
            metadata = item.get('metadata', {})
            
            # Check various fields that might contain source information
            if isinstance(metadata, dict):
                # Try the common fields for document sources
                for field in ['source', 'filename', 'file', 'path', 'filepath']:
                    if field in metadata and metadata[field]:
                        if isinstance(metadata[field], str):
                            sources.add(metadata[field])
                            # Also add basename to filenames
                            filenames.add(os.path.basename(metadata[field]))
    
    # Print unique sources
    print(f"\nFound {len(sources)} unique source paths in collection:")
    for source in sorted(sources):
        print(f"- {source}")
        
    # Print unique filenames
    print(f"\nFound {len(filenames)} unique filenames in collection:")
    for filename in sorted(filenames):
        print(f"- {filename}")

if __name__ == "__main__":
    main() 