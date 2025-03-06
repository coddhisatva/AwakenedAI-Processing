import logging
import os
import sys
from pathlib import Path
import json
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage.vector_store import SupabaseVectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the vector store functionality."""
    
    # Create vector store
    vector_store = SupabaseVectorStore()
    
    # Sample documents
    documents = [
        {
            "id": "doc1",
            "content": "The quick brown fox jumps over the lazy dog",
            "metadata": {
                "source": "sample_book.pdf",
                "page": 1,
                "chapter": "Introduction",
                "title": "Sample Book",
                "author": "Test Author"
            }
        },
        {
            "id": "doc2",
            "content": "A fast fox leaps over a sleeping canine",
            "metadata": {
                "source": "sample_book.pdf",
                "page": 2,
                "chapter": "Introduction",
                "title": "Sample Book",
                "author": "Test Author"
            }
        },
        {
            "id": "doc3",
            "content": "The benefits of artificial intelligence in modern society",
            "metadata": {
                "source": "tech_book.pdf",
                "page": 15,
                "chapter": "AI Overview",
                "title": "Technology Concepts",
                "author": "Tech Writer"
            }
        },
        {
            "id": "doc4",
            "content": "Machine learning algorithms are transforming how we approach data analysis",
            "metadata": {
                "source": "tech_book.pdf",
                "page": 16,
                "chapter": "AI Overview",
                "title": "Technology Concepts",
                "author": "Tech Writer"
            }
        }
    ]
    
    # Add documents
    logger.info("Adding documents to Supabase vector store...")
    doc_ids = vector_store.add_documents(documents)
    logger.info(f"Added {len(doc_ids)} documents with IDs: {doc_ids}")
    
    # Test search
    logger.info("\nTesting search functionality...")
    query = "fox jumping over dog"
    results = vector_store.search(query, k=2)
    
    logger.info(f"Search results for: '{query}'")
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}:")
        logger.info(f"  ID: {result['id']}")
        logger.info(f"  Text: {result['text']}")
        logger.info(f"  Metadata: {result['metadata']}")
        logger.info(f"  Score: {result['score']}")
    
    # Test document retrieval by ID
    if len(doc_ids) > 0:
        logger.info("\nTesting document retrieval by ID...")
        doc = vector_store.get_document(doc_ids[0])
        if doc:
            logger.info(f"Retrieved document:")
            logger.info(f"  ID: {doc['id']}")
            logger.info(f"  Text: {doc['text']}")
            logger.info(f"  Metadata: {doc['metadata']}")
        else:
            logger.info("Document not found")

if __name__ == "__main__":
    main()
