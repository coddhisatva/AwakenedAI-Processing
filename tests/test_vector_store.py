import logging
import os
from pathlib import Path
import json
from dotenv import load_dotenv
from src.storage.vector_store import ChromaVectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the vector store functionality."""
    # Set up directories
    vector_db_dir = "data/vector_db"
    os.makedirs(vector_db_dir, exist_ok=True)
    
    # Get collection name from environment variables or use default
    collection_name = os.getenv("COLLECTION_NAME", "awakened_ai_test")
    
    logger.info(f"Using collection name: {collection_name}")
    
    # Create vector store
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=vector_db_dir
    )
    
    # Sample documents
    documents = [
        {
            "id": "doc1",
            "content": "The quick brown fox jumps over the lazy dog",
            "metadata": {
                "source": "sample_book.pdf",
                "page": 1,
                "chapter": "Introduction"
            }
        },
        {
            "id": "doc2",
            "content": "A fast fox leaps over a sleeping canine",
            "metadata": {
                "source": "sample_book.pdf",
                "page": 2,
                "chapter": "Introduction"
            }
        },
        {
            "id": "doc3",
            "content": "The benefits of artificial intelligence in modern society",
            "metadata": {
                "source": "tech_book.pdf",
                "page": 15,
                "chapter": "AI Overview"
            }
        },
        {
            "id": "doc4",
            "content": "Machine learning algorithms are transforming how we approach data analysis",
            "metadata": {
                "source": "tech_book.pdf",
                "page": 16,
                "chapter": "AI Overview"
            }
        }
    ]
    
    # Add documents
    logger.info("Adding documents to vector store...")
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
    
    # Test metadata filtering
    logger.info("\nTesting metadata filtering...")
    filter_results = vector_store.search(
        query="artificial intelligence", 
        k=5,
        filter_dict={"source": "tech_book.pdf"}
    )
    
    logger.info(f"Filtered search results:")
    for i, result in enumerate(filter_results):
        logger.info(f"Result {i+1}:")
        logger.info(f"  ID: {result['id']}")
        logger.info(f"  Text: {result['text']}")
        logger.info(f"  Metadata: {result['metadata']}")
        logger.info(f"  Score: {result['score']}")
    
    # Test document retrieval by ID
    logger.info("\nTesting document retrieval by ID...")
    doc = vector_store.get_document("doc1")
    if doc:
        logger.info(f"Retrieved document:")
        logger.info(f"  ID: {doc['id']}")
        logger.info(f"  Text: {doc['text']}")
        logger.info(f"  Metadata: {doc['metadata']}")
    else:
        logger.info("Document not found")

if __name__ == "__main__":
    main()
