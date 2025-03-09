#!/usr/bin/env python3
"""
Process a sample of documents for testing the RAG system.
This allows quick population of the vector store without processing the entire library.
"""

import os
import sys
import argparse
import logging
import random
import json
from pathlib import Path

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from src.extraction.extractor import DocumentExtractor
from src.processing.chunker import SemanticChunker
from src.embedding.embedder import DocumentEmbedder
from src.storage.vector_store import SupabaseVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,%f",
)
logger = logging.getLogger(__name__)

def setup_components(args):
    """Initialize all pipeline components."""
    # Define directories
    data_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data")))
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    chunks_dir = data_dir / "chunks"
    embeddings_dir = data_dir / "embeddings"
    
    # Create directories if they don't exist
    for dir_path in [raw_dir, processed_dir, chunks_dir, embeddings_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize components
    extractor = DocumentExtractor(raw_dir=raw_dir, processed_dir=processed_dir)
    chunker = SemanticChunker(processed_dir=processed_dir, chunks_dir=chunks_dir)
    embedder = DocumentEmbedder(chunks_dir=chunks_dir, embeddings_dir=embeddings_dir)
    
    # Initialize Supabase vector store
    vector_store = SupabaseVectorStore()
    
    return {
        "extractor": extractor,
        "chunker": chunker,
        "embedder": embedder,
        "vector_store": vector_store,
        "raw_dir": raw_dir,
    }

def process_documents(components, files_to_process):
    """Process a batch of documents and add them to the vector store."""
    extractor = components["extractor"]
    chunker = components["chunker"]
    embedder = components["embedder"]
    vector_store = components["vector_store"]
    
    logger.info(f"Processing {len(files_to_process)} documents")
    
    # Step 1: Extract text from documents
    processed_files = []
    for file_path in files_to_process:
        try:
            # Process the PDF file directly
            try:
                extractor.extract_pdf(Path(file_path))
                processed_path = Path(extractor.processed_dir) / f"{Path(file_path).stem}.json"
                processed_files.append(processed_path)
                logger.info(f"Extracted document: {file_path} -> {processed_path}")
            except Exception as e:
                logger.error(f"Error extracting document {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
    
    # Step 2: Process documents into chunks
    chunk_files = []
    for processed_file in processed_files:
        try:
            # Use the internal _chunk_document method
            num_chunks = chunker._chunk_document(processed_file)
            chunk_file = Path(chunker.chunks_dir) / f"{processed_file.stem}_chunks.json"
            if chunk_file.exists():
                chunk_files.append(chunk_file)
                logger.info(f"Created chunk file: {chunk_file} with {num_chunks} chunks")
            else:
                logger.warning(f"No chunks file found at {chunk_file}")
        except Exception as e:
            logger.error(f"Error chunking document {processed_file}: {str(e)}")
    
    # Step 3: Generate embeddings for chunks
    # Process each chunk file directly through the embedder
    embedded_chunk_files = []
    for chunk_file in chunk_files:
        try:
            num_embedded = embedder._embed_document_chunks(chunk_file)
            embedding_file = Path(embedder.embeddings_dir) / f"{chunk_file.stem.replace('_chunks', '')}_embeddings.json"
            if embedding_file.exists():
                embedded_chunk_files.append(embedding_file)
                logger.info(f"Generated embeddings file: {embedding_file} with {num_embedded} chunks")
            else:
                logger.warning(f"No embeddings file found at {embedding_file}")
        except Exception as e:
            logger.error(f"Error embedding chunks in {chunk_file}: {e}")
    
    # Step 4: Add embedded chunks to vector store
    total_chunks_added = 0
    for embedding_file in embedded_chunk_files:
        try:
            with open(embedding_file, 'r', encoding='utf-8') as f:
                embedded_chunks = json.load(f)
            
            # Make sure embedded_chunks is a list
            if isinstance(embedded_chunks, dict) and 'chunks' in embedded_chunks:
                embedded_chunks = embedded_chunks['chunks']
                
            # Add to vector store
            vector_store.add_documents(embedded_chunks)
            total_chunks_added += len(embedded_chunks)
            logger.info(f"Added {len(embedded_chunks)} chunks from {embedding_file.name} to Supabase")
        except Exception as e:
            logger.error(f"Error adding embeddings from {embedding_file} to vector store: {e}")
    
    logger.info(f"Added a total of {total_chunks_added} chunks to Supabase")
    return total_chunks_added

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Process a sample of documents for the RAG system")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        help="Directory containing documents to process. Default is data/raw",
        default=os.path.join("data", "raw")
    )
    parser.add_argument(
        "--count", 
        type=int, 
        help="Number of documents to process (randomly selected). Default is 5",
        default=5
    )
    parser.add_argument(
        "--specific_files", 
        nargs="+", 
        help="Specific files to process (full paths)"
    )
    parser.add_argument(
        "--extensions", 
        nargs="+", 
        default=["pdf", "epub", "txt"], 
        help="File extensions to include (default: pdf epub txt)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Set up components
    components = setup_components(args)
    
    # Determine which files to process
    if args.specific_files:
        files_to_process = [Path(file) for file in args.specific_files if os.path.isfile(file)]
        logger.info(f"Processing {len(files_to_process)} specified files")
    else:
        # Find all files with specified extensions in the input directory
        input_dir = Path(args.input_dir)
        all_files = []
        for ext in args.extensions:
            all_files.extend(list(input_dir.glob(f"**/*.{ext}")))
        
        if not all_files:
            logger.error(f"No files found with extensions {args.extensions} in {input_dir}")
            return
        
        # Select a random sample if there are more files than the requested count
        logger.info(f"Found {len(all_files)} total files with extensions {args.extensions}")
        if len(all_files) > args.count:
            files_to_process = random.sample(all_files, args.count)
        else:
            files_to_process = all_files
        
        logger.info(f"Selected {len(files_to_process)} files for processing")
    
    # Process the documents
    total_chunks = process_documents(components, files_to_process)
    
    logger.info(f"Processing complete. Added {total_chunks} chunks to Supabase.")
    if total_chunks > 0:
        logger.info(f"You can now query the system with: python tools/query_cli.py \"Your question here\"")

if __name__ == "__main__":
    main()
