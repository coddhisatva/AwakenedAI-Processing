#!/usr/bin/env python3
"""
Very simple script to directly show which sources were processed.
"""
import os
import sys
import json
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import directly from the source file to avoid any complex logic
import supabase
from supabase import create_client, Client

# Get environment variables
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    print("Error: SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    sys.exit(1)

# Create Supabase client
client = create_client(supabase_url, supabase_key)

# Just get some raw data from chunks table
print("Querying Supabase chunks table...")
response = client.table("chunks").select("*").limit(10).execute()

if not response.data:
    print("No chunks found in Supabase database")
    sys.exit(1)

# Print sample chunk data
print(f"\nFound {len(response.data)} chunks. Here's the first one:")
sample_chunk = response.data[0]
print(json.dumps(sample_chunk, indent=2))

# Try to extract sources from metadata
print("\nAttempting to find source information in chunks...")
sources = set()
metadata_response = client.table("chunks").select("metadata").execute()

for chunk in metadata_response.data:
    metadata = chunk.get("metadata", {})
    
    # Try different potential source fields
    source = None
    for field in ["source", "filename", "file", "path", "filepath"]:
        if field in metadata:
            source = metadata[field]
            break
    
    if source:
        sources.add(source)

# Print sources
if sources:
    print("\nFound these document sources:")
    for source in sorted(sources):
        print(f"- {source}")
else:
    print("\nCouldn't find any source information in metadata")
    
    # Dump all metadata fields as a last resort
    print("\nDumping all metadata fields to help troubleshooting:")
    metadata_fields = set()
    for chunk in metadata_response.data[:20]:  # Limit to first 20 chunks
        metadata = chunk.get("metadata", {})
        for key in metadata.keys():
            metadata_fields.add(key)
    
    print("Metadata fields found:", metadata_fields)
    
    if metadata_fields:
        print("\nSample values for each field:")
        for field in metadata_fields:
            values = set()
            for chunk in metadata_response.data[:20]:
                metadata = chunk.get("metadata", {})
                if field in metadata:
                    value = metadata[field]
                    if isinstance(value, (str, int, float, bool)):
                        values.add(str(value))
            print(f"- {field}: {', '.join(values)[:100]}{'...' if len(', '.join(values)) > 100 else ''}") 