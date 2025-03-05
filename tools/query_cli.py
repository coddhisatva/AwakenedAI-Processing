import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extraction.extractor import DocumentExtractor
from src.processing.chunker import SemanticChunker
from src.embedding.embedder import DocumentEmbedder
from src.storage.vector_store import SupabaseVectorStore
from src.pipeline.rag_pipeline import RAGPipeline
from src.interface.query import QueryEngine

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_pipeline():
    # Define default directories for all pipeline components
    data_dir = Path(os.path.join(os.path.dirname(__file__), "../data"))
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    chunks_dir = data_dir / "chunks"
    embeddings_dir = data_dir / "embeddings"
    
    # Create directories if they don't exist
    for dir_path in [raw_dir, processed_dir, chunks_dir, embeddings_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize each component with its required arguments
    extractor = DocumentExtractor(raw_dir=raw_dir, processed_dir=processed_dir)
    chunker = SemanticChunker(processed_dir=processed_dir, chunks_dir=chunks_dir)
    embedder = DocumentEmbedder(chunks_dir=chunks_dir, embeddings_dir=embeddings_dir)
    
    # Initialize Supabase vector store
    vector_store = SupabaseVectorStore()
    
    # Return the RAG pipeline
    return RAGPipeline(
        extractor=extractor,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store
    )

def main():
    """CLI entry point for querying the RAG system."""
    parser = argparse.ArgumentParser(description="Query the Awakened AI RAG system")
    parser.add_argument("query", type=str, help="The query to process")
    parser.add_argument("--results", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    logger.info("Setting up RAG pipeline...")
    pipeline = setup_pipeline()
    
    query_engine = QueryEngine(pipeline)
    
    logger.info(f"Processing query: {args.query}")
    response = query_engine.ask(args.query, args.results)
    
    print("\n" + "=" * 80)
    print(response)
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
