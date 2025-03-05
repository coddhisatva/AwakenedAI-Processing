#!/usr/bin/env python3
"""
Estimate processing time for all files in the raw directory.
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage.vector_store import ChromaVectorStore

# Load environment variables
load_dotenv()

def main():
    # Get paths
    data_dir = Path(os.path.join(os.path.dirname(__file__), "../data"))
    raw_dir = data_dir / "raw"
    db_dir = data_dir / "vector_db"
    
    # Connect to vector store
    store = ChromaVectorStore(
        persist_directory=str(db_dir),
        collection_name="awakened_ai_default"
    )
    
    # Get current stats
    collection_count = store.collection.count()
    print(f"Current collection: {collection_count} chunks")
    
    # Get processed files
    processed_files = set()
    batch_size = 500
    for i in range(0, collection_count, batch_size):
        end_idx = min(i + batch_size, collection_count)
        batch = store.collection.get(limit=batch_size, offset=i)
        
        for meta in batch['metadatas']:
            if 'filename' in meta:
                processed_files.add(meta['filename'])
            elif 'source' in meta:
                processed_files.add(os.path.basename(meta['source']))
    
    print(f"Processed files: {len(processed_files)}")
    for file in sorted(processed_files):
        print(f"- {file}")
    
    # Get all files in raw directory
    all_files = []
    file_sizes = {}
    extensions = defaultdict(int)
    
    for file in os.listdir(raw_dir):
        file_path = os.path.join(raw_dir, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            file_sizes[file] = size_mb
            all_files.append(file)
            
            # Count extensions
            ext = os.path.splitext(file)[1].lower()
            extensions[ext] += 1
    
    # Calculate remaining files
    remaining_files = [f for f in all_files if f not in processed_files]
    
    print(f"\nRemaining files to process: {len(remaining_files)}")
    total_remaining_mb = sum(file_sizes[f] for f in remaining_files)
    print(f"Total size of remaining files: {total_remaining_mb:.2f} MB")
    
    # Count file types
    print("\nFile types in raw directory:")
    for ext, count in extensions.items():
        print(f"- {ext}: {count} files")
    
    # Estimate processing time
    if len(processed_files) > 0:
        # Calculate average processing rate from processed files
        processed_mb = sum(file_sizes[f] for f in processed_files if f in file_sizes)
        # Estimate 5 minutes per MB as a very rough estimate (can be adjusted based on actual times)
        minutes_per_mb = 5
        estimated_minutes = total_remaining_mb * minutes_per_mb
        
        print(f"\nEstimated processing time:")
        print(f"- Based on size: {estimated_minutes:.1f} minutes ({estimated_minutes/60:.1f} hours)")
        
        # If we know specific times, we could add more precise calculations
        
    else:
        print("\nCannot estimate processing time without data on already processed files.")

if __name__ == "__main__":
    main() 