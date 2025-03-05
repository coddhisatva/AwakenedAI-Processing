import logging
import os
from pathlib import Path
from embedding.embedder import DocumentEmbedder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the document embedder."""
    # Set up directories
    chunks_dir = "data/chunks"
    embeddings_dir = "data/embeddings"
    
    # Get embedding model from environment variables or use default
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    logger.info(f"Using embedding model: {embedding_model}")
    
    # Create embedder
    embedder = DocumentEmbedder(
        chunks_dir=chunks_dir,
        embeddings_dir=embeddings_dir,
        embedding_model=embedding_model,
        batch_size=10  # Process 10 chunks at a time to avoid rate limits
    )
    
    # Process a limited number of files for testing
    stats = embedder.embed_documents(limit=2)  # Only process 2 files for testing
    
    # Print results
    logger.info(f"Embedding results: {stats}")
    
    # Print sample embeddings from one document (if available)
    embedding_files = list(Path(embeddings_dir).glob("*.json"))
    if embedding_files:
        sample_file = embedding_files[0]
        logger.info(f"Sample embeddings from {sample_file.name}:")
        
        import json
        with open(sample_file, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
            
        # Print number of embeddings and first embedding dimensions
        logger.info(f"Number of chunks with embeddings: {len(embeddings)}")
        if embeddings and 'embedding' in embeddings[0]:
            # Print first 5 dimensions of the first embedding
            logger.info(f"First embedding dimensions (first 5): {embeddings[0]['embedding'][:5]}")
            logger.info(f"Embedding dimension: {len(embeddings[0]['embedding'])}")

if __name__ == "__main__":
    main() 