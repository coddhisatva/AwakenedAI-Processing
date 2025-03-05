import logging
import os
from pathlib import Path
from processing.chunker import SemanticChunker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the semantic chunker."""
    # Set up directories
    processed_dir = "data/processed"
    chunks_dir = "data/chunks"
    
    # Get chunk size and overlap from environment variables or use defaults
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
    
    logger.info(f"Using chunk size: {chunk_size}, chunk overlap: {chunk_overlap}")
    
    # Create chunker
    chunker = SemanticChunker(
        processed_dir=processed_dir,
        chunks_dir=chunks_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Process all documents
    stats = chunker.chunk_all_documents()
    
    # Print results
    logger.info(f"Chunking results: {stats}")
    
    # Print sample chunks from one document (if available)
    chunk_files = list(Path(chunks_dir).glob("*.json"))
    if chunk_files:
        sample_file = chunk_files[0]
        logger.info(f"Sample chunks from {sample_file.name}:")
        
        import json
        with open(sample_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        # Print number of chunks and first chunk
        logger.info(f"Number of chunks: {len(chunks)}")
        if chunks:
            logger.info(f"First chunk: {chunks[0]['content'][:200]}...")

if __name__ == "__main__":
    main() 