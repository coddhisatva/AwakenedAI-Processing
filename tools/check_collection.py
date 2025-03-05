#!/usr/bin/env python3
"""
Check the contents of the ChromaDB collection.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage.vector_store import ChromaVectorStore

# Load environment variables
load_dotenv()

def main():
    # Initialize vector store
    data_dir = Path(os.path.join(os.path.dirname(__file__), "../data"))
    db_dir = data_dir / "vector_db"
    collection_name = "awakened_ai_default"
    
    store = ChromaVectorStore(
        persist_directory=str(db_dir),
        collection_name=collection_name
    )
    
    # Get collection stats
    collection_count = store.collection.count()
    print(f"Collection count: {collection_count} chunks")
    
    # Collect unique filenames
    filenames = set()
    
    # Process in batches to avoid memory issues
    batch_size = 500
    for i in range(0, collection_count, batch_size):
        end_idx = min(i + batch_size, collection_count)
        print(f"Processing chunks {i} to {end_idx}...")
        batch = store.collection.get(limit=batch_size, offset=i)
        
        for meta in batch['metadatas']:
            if 'filename' in meta:
                filenames.add(meta['filename'])
            elif 'source' in meta:
                filenames.add(os.path.basename(meta['source']))
    
    # Print unique documents
    print(f"\nFound {len(filenames)} unique documents in collection:")
    for filename in sorted(filenames):
        print(f"- {filename}")

if __name__ == "__main__":
    main() 