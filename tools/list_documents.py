#!/usr/bin/env python3
"""
List all documents in the Supabase database.
"""
import os
import sys
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """List all documents in the database."""
    try:
        # Load environment variables
        load_dotenv()
        
        print("Connecting to Supabase...")
        # Import here to catch any import errors
        from database.supabase_adapter import SupabaseAdapter
        adapter = SupabaseAdapter()
        print("Successfully connected to Supabase")
        
        # Initialize counters
        doc_count = 0
        chunk_count = 0
        
        # Get document count
        try:
            doc_count = adapter.count_documents()
            print(f"\nTotal documents in database: {doc_count}")
        except Exception as e:
            print(f"Error getting document count: {str(e)}")
        
        # Get all documents
        try:
            print("\nRetrieving document details...")
            response = adapter.supabase.table("documents").select("*").execute()
            documents = response.data
            
            if not documents:
                print("No documents found in the database.")
            else:
                # Filter for valid documents (non-empty filepath)
                valid_documents = [doc for doc in documents if doc.get('filepath')]
                invalid_documents = [doc for doc in documents if not doc.get('filepath')]
                
                # Sort by creation time (newest first) to see recent entries
                for doc_list in [valid_documents, invalid_documents]:
                    if doc_list:
                        for doc in doc_list:
                            if 'created_at' in doc:
                                try:
                                    doc['_created_timestamp'] = datetime.fromisoformat(doc['created_at'].replace('Z', '+00:00'))
                                except:
                                    doc['_created_timestamp'] = datetime.min
                        doc_list.sort(key=lambda x: x.get('_created_timestamp', datetime.min), reverse=True)
                
                print(f"\nFound {len(documents)} total documents:")
                print(f"- {len(valid_documents)} valid documents (with filepath)")
                print(f"- {len(invalid_documents)} invalid documents (without filepath)")
                
                # Look for our recently processed documents based on known filenames
                known_files = [
                    "Frank Rudolph Young - Cyclomancy - The Secret of Psychic Power Control.pdf",
                    "frank-rudolph-young-yoga-for-men-only.pdf",
                    "How_to_Get_Rich_-_Felix_Dennis.pdf",
                    "Erikson, Thomas - Surrounded by Idiots (2019).epub"
                ]
                
                # Search for these files in all documents
                found_known_docs = []
                for doc in documents:
                    filepath = doc.get('filepath', '')
                    for known_file in known_files:
                        if known_file in filepath or known_file == filepath:
                            found_known_docs.append(doc)
                            break
                
                if found_known_docs:
                    print(f"\nFound {len(found_known_docs)} of our known documents:")
                    print("-" * 80)
                    for i, doc in enumerate(found_known_docs, 1):
                        print(f"{i}. Title: {doc.get('title', 'Unknown')}")
                        print(f"   Author: {doc.get('author', 'Unknown')}")
                        print(f"   Path: {doc.get('filepath', 'Unknown')}")
                        print(f"   Created: {doc.get('created_at', 'Unknown')}")
                        print(f"   ID: {doc.get('id', 'Unknown')}")
                        print("-" * 80)
                else:
                    print("\nNone of our known documents found in the database!")
                
                if valid_documents:
                    print("\nMOST RECENT VALID DOCUMENTS:")
                    print("-" * 80)
                    
                    for i, doc in enumerate(valid_documents[:10], 1):
                        print(f"{i}. Title: {doc.get('title', 'Unknown')}")
                        print(f"   Author: {doc.get('author', 'Unknown')}")
                        print(f"   Path: {doc.get('filepath', 'Unknown')}")
                        print(f"   Created: {doc.get('created_at', 'Unknown')}")
                        print("-" * 80)
                
                if invalid_documents:
                    # Group by creation time to see patterns
                    creation_times = {}
                    for doc in invalid_documents:
                        created_at = doc.get('created_at', 'Unknown')
                        if created_at != 'Unknown':
                            created_at = created_at.split('T')[0]  # Just get the date part
                        creation_times[created_at] = creation_times.get(created_at, 0) + 1
                    
                    print("\nINVALID DOCUMENTS BY CREATION DATE:")
                    for date, count in sorted(creation_times.items()):
                        print(f"- {date}: {count} documents")
                    
                    print("\nRECENT INVALID DOCUMENTS (SAMPLE):")
                    print("-" * 80)
                    
                    for i, doc in enumerate(invalid_documents[:5], 1):
                        print(f"{i}. ID: {doc.get('id', 'Unknown')}")
                        print(f"   Title: {doc.get('title', 'Unknown')}")
                        print(f"   Created: {doc.get('created_at', 'Unknown')}")
                        print(f"   Other fields: {json.dumps({k:v for k,v in doc.items() if k not in ['id', 'title', 'created_at', '_created_timestamp']})}")
                        print("-" * 80)
                        
                    # Check if these are related to our processing
                    today_invalid = [doc for doc in invalid_documents if doc.get('created_at', '').startswith('2025-03-08')]
                    if today_invalid:
                        print(f"\nFound {len(today_invalid)} invalid documents created today!")
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Get chunk count
        try:
            chunk_count = adapter.count_chunks()
            print(f"\nTotal chunks in database: {chunk_count}")
            if doc_count > 0:
                print(f"Average chunks per document: {chunk_count / doc_count:.2f}")
        except Exception as e:
            print(f"Error getting chunk count: {str(e)}")
        
        # Display summary at the end
        print("\n" + "=" * 50)
        print("DATABASE SUMMARY")
        print("=" * 50)
        print(f"Total documents: {doc_count}")
        if 'valid_documents' in locals():
            print(f"Valid documents (with filepath): {len(valid_documents)}")
            print(f"Invalid documents (no filepath): {len(invalid_documents)}")
        print(f"Total chunks: {chunk_count}")
        if doc_count > 0:
            print(f"Overall chunks per document: {chunk_count / doc_count:.2f}")
        if 'valid_documents' in locals() and len(valid_documents) > 0:
            print(f"Chunks per valid document: {chunk_count / len(valid_documents):.2f}")
        print("=" * 50)
        
        # Return the count for potential scripting use
        return doc_count
            
    except ImportError as e:
        print(f"Error importing required modules: {str(e)}")
        print("Make sure you have installed all requirements and are in the correct directory.")
        return 0
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    count = main()
    sys.exit(0)  # Exit successfully 