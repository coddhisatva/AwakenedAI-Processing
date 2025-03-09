#!/usr/bin/env python3
"""
Process all documents in the raw directory for the RAG system.
This script is designed for processing the full collection of documents.
It uses the RAGPipeline which is optimized for production-scale processing.

Optimized approach:
1. Check files against local manifest first (fastest check)
2. For files not in manifest, check against database
3. Process only files not in manifest AND not in database
4. Update manifest with both newly processed files and database duplicates
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from src.extraction.extractor import DocumentExtractor
from src.processing.chunker import SemanticChunker
from src.embedding.embedder import DocumentEmbedder
from src.storage.vector_store import SupabaseVectorStore
from src.pipeline.rag_pipeline import RAGPipeline
from database.supabase_adapter import SupabaseAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,%f",
)
logger = logging.getLogger(__name__)

# Path for the processed documents manifest
MANIFEST_FILE = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed_documents.json")))

class Timer:
    """Utility class for timing operations."""
    
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        self.end_time = time.time()
        
    @property
    def elapsed(self):
        """Return elapsed time in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def elapsed_str(self):
        """Return formatted elapsed time string."""
        seconds = self.elapsed
        return str(timedelta(seconds=seconds))
    
    def log(self, message=None):
        """Log elapsed time with optional message."""
        msg = f"{self.name}: {self.elapsed_str}"
        if message:
            msg = f"{msg} - {message}"
        logger.info(msg)

def load_processed_docs_manifest() -> Dict[str, Dict[str, Any]]:
    """
    Load the manifest of processed documents.
    
    Returns:
        Dictionary with filenames as keys and processing metadata as values
    """
    if not MANIFEST_FILE.exists():
        logger.info(f"No manifest file found at {MANIFEST_FILE}. Creating a new one.")
        return {}
    
    try:
        with open(MANIFEST_FILE, 'r') as f:
            manifest = json.load(f)
        logger.info(f"Loaded manifest with {len(manifest)} processed documents.")
        return manifest
    except Exception as e:
        logger.error(f"Error loading manifest file: {e}")
        return {}

def update_processed_docs_manifest(manifest: Dict[str, Dict[str, Any]], 
                                  files: List[Path], 
                                  source_type: str) -> None:
    """
    Update the manifest with new files.
    
    Args:
        manifest: Current manifest dictionary
        files: List of file paths to add
        source_type: Type of source ('direct_processing' or 'database_duplicate')
    """
    count = 0
    
    # Update the manifest with new files
    for file_path in files:
        if file_path.name not in manifest:
            manifest[file_path.name] = {
                "filepath": str(file_path),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": source_type
            }
            count += 1
    
    # Write the updated manifest
    try:
        with open(MANIFEST_FILE, 'w') as f:
            json.dump(manifest, f, indent=2)
        if count > 0:
            logger.info(f"Updated manifest file with {count} new {source_type} documents.")
    except Exception as e:
        logger.error(f"Error updating manifest file: {e}")

def setup_pipeline(args):
    """Initialize the RAG pipeline with all components."""
    # Define directories
    data_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data")))
    raw_dir = Path(args.input_dir)
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
    
    # Initialize the RAG pipeline
    pipeline = RAGPipeline(
        extractor=extractor, 
        chunker=chunker, 
        embedder=embedder, 
        vector_store=vector_store
    )
    
    return pipeline, extractor, vector_store

def find_all_files(directory_path: Path, extensions: List[str], skip_ocr: bool = False) -> List[Path]:
    """
    Get all files with the specified extensions in the directory.
    
    Args:
        directory_path: Path to search for files
        extensions: List of file extensions to include
        skip_ocr: Whether to skip files that likely need OCR
    """
    all_files = []
    for ext in extensions:
        all_files.extend(list(directory_path.glob(f"**/*.{ext}")))
    
    # Skip OCR files if requested
    skipped_ocr_count = 0
    if skip_ocr:
        # Known OCR-needing files
        ocr_files = [f for f in all_files if any(pattern in f.name.lower() 
                     for pattern in ["frank-rudolph-young", "cyclomancy", "yoga-for-men-only"])]
        
        if ocr_files:
            logger.info(f"Skipping {len(ocr_files)} files that need OCR: {', '.join(f.name for f in ocr_files)}")
            all_files = [f for f in all_files if f not in ocr_files]
            skipped_ocr_count = len(ocr_files)
    
    return all_files, skipped_ocr_count

def categorize_files(files: List[Path], 
                    manifest: Dict[str, Dict[str, Any]], 
                    supabase_adapter: SupabaseAdapter,
                    force_reprocess: bool = False) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Categorize files into three groups:
    1. Already in manifest (skip)
    2. Not in manifest but in database (add to manifest as database_duplicate)
    3. Not in manifest and not in database (process normally)
    
    Args:
        files: List of file paths to categorize
        manifest: Current manifest dictionary
        supabase_adapter: SupabaseAdapter instance
        force_reprocess: Whether to process all files regardless of manifest
        
    Returns:
        Tuple of (manifest_files, database_files, new_files)
    """
    manifest_files = []
    database_files = []
    new_files = []
    
    # Get set of filenames already in manifest
    manifest_filenames = set(manifest.keys())
    
    # Stats for timing database checks
    db_check_count = 0
    db_match_count = 0
    
    # First pass: identify files in manifest
    with Timer("Manifest check") as manifest_timer:
        for file_path in files:
            if file_path.name in manifest_filenames and not force_reprocess:
                manifest_files.append(file_path)
            else:
                # For files not in manifest or being forced to reprocess, 
                # check if they're in database
                db_check_count += 1
                existing_doc = None
                with Timer("Database lookup") as db_timer:
                    existing_doc = supabase_adapter.get_document_by_filepath(str(file_path))
                
                if existing_doc and not force_reprocess:
                    database_files.append(file_path)
                    db_match_count += 1
                else:
                    new_files.append(file_path)
    
    # Log results
    if manifest_files:
        logger.info(f"Found {len(manifest_files)} files already in manifest")
        
    if database_files:
        logger.info(f"Found {len(database_files)} files in database but not in manifest")
    
    if db_check_count > 0:
        avg_db_check_time = manifest_timer.elapsed / db_check_count if db_check_count > 0 else 0
        logger.info(f"Performed {db_check_count} database checks, found {db_match_count} matches")
        logger.info(f"Average database check time: {avg_db_check_time:.4f} seconds")
        
    logger.info(f"Found {len(new_files)} new files to process")
    
    return manifest_files, database_files, new_files

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Process all documents for the RAG system")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        help="Directory containing documents to process. Default is data/raw",
        default=os.path.join("data", "raw")
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Number of documents to process in each batch. Default is 5",
        default=5
    )
    parser.add_argument(
        "--extensions", 
        nargs="+", 
        default=["pdf", "epub", "txt"], 
        help="File extensions to include (default: pdf epub txt)"
    )
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="Process all files even if they are in the manifest or database"
    )
    parser.add_argument(
        "--update_manifest_only",
        action="store_true",
        help="Only update the manifest with files found in the database, don't process any files"
    )
    parser.add_argument(
        "--skip_ocr",
        action="store_true",
        help="Skip files that need OCR processing"
    )
    
    args = parser.parse_args()
    
    # Start overall timer
    start_time = time.time()
    overall_timer = Timer("Total processing time")
    overall_timer.__enter__()
    
    # Performance statistics
    stats = {
        "total_files": 0,
        "manifest_files": 0,
        "database_files": 0,
        "new_files": 0,
        "skipped_ocr_files": 0,
        "successful_files": 0,
        "failed_files": 0,
        "total_chunks": 0,
        "timings": {}
    }
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Load the manifest of processed documents
        with Timer("Loading manifest") as t:
            manifest = load_processed_docs_manifest()
        stats["timings"]["load_manifest"] = t.elapsed
        
        # Set up the RAG pipeline and related components
        with Timer("Setting up pipeline") as t:
            pipeline, extractor, vector_store = setup_pipeline(args)
        stats["timings"]["setup_pipeline"] = t.elapsed
        
        # Initialize Supabase adapter for checking database
        supabase_adapter = SupabaseAdapter()
        
        # Get the list of all files in the directory
        with Timer("Finding files") as t:
            input_dir = Path(args.input_dir)
            all_files, skipped_ocr_count = find_all_files(input_dir, args.extensions, args.skip_ocr)
            stats["skipped_ocr_files"] = skipped_ocr_count
        stats["timings"]["find_files"] = t.elapsed
        
        if not all_files:
            logger.error(f"No files found with extensions {args.extensions} in {input_dir}")
            return
            
        stats["total_files"] = len(all_files)
        logger.info(f"Found {len(all_files)} total files with extensions {args.extensions}")
        
        # Categorize files
        with Timer("Categorizing files") as t:
            manifest_files, database_files, new_files = categorize_files(
                all_files, 
                manifest, 
                supabase_adapter,
                force_reprocess=args.force_reprocess
            )
        stats["timings"]["categorize_files"] = t.elapsed
        stats["manifest_files"] = len(manifest_files)
        stats["database_files"] = len(database_files)
        stats["new_files"] = len(new_files)
        
        # Update manifest with database files
        if database_files:
            with Timer("Updating manifest with database files") as t:
                update_processed_docs_manifest(manifest, database_files, "database_duplicate")
            stats["timings"]["update_manifest_db"] = t.elapsed
        
        # If only updating manifest, we're done
        if args.update_manifest_only:
            logger.info("Manifest updated with database duplicates. Exiting without processing files.")
            return
        
        # If no new files to process, we're done
        if not new_files and not args.force_reprocess:
            logger.info("No new files to process. All documents are either in manifest or database.")
            return
        
        # Process only the files that need processing
        processing_timer = Timer("Document processing")
        processing_timer.__enter__()
        
        try:
            # If forcing reprocessing of all files
            if args.force_reprocess:
                logger.info(f"Force reprocessing {len(all_files)} files")
                with Timer("Full pipeline processing") as t:
                    total_chunks = pipeline.process_directory(str(input_dir), batch_size=args.batch_size)
                stats["timings"]["full_pipeline"] = t.elapsed
                stats["total_chunks"] = total_chunks
                stats["successful_files"] = len(all_files)
                
                # Update manifest with all files as directly processed
                with Timer("Updating manifest with all files") as t:
                    update_processed_docs_manifest(manifest, all_files, "direct_processing")
                stats["timings"]["update_manifest_all"] = t.elapsed
            else:
                # Only process new files
                logger.info(f"Processing {len(new_files)} new files")
                
                # Use the pipeline to process the files
                # This still leverages RAGPipeline but only on files we know need processing
                successful_files = []
                failed_files = []
                per_file_times = []
                
                # Process in batches
                for i in range(0, len(new_files), args.batch_size):
                    batch = new_files[i:i+args.batch_size]
                    logger.info(f"Processing batch {i//args.batch_size + 1}/{(len(new_files)-1)//args.batch_size + 1}")
                    
                    for file_path in batch:
                        file_timer = Timer(f"Processing {file_path.name}")
                        file_timer.__enter__()
                        try:
                            if file_path.suffix.lower() == '.pdf':
                                extractor._process_pdf(file_path)
                            elif file_path.suffix.lower() == '.epub':
                                extractor._process_epub(file_path)
                            else:
                                logger.warning(f"Unsupported file type: {file_path}")
                                continue
                            successful_files.append(file_path)
                            file_timer.__exit__(None, None, None)
                            per_file_times.append(file_timer.elapsed)
                            logger.info(f"Processed {file_path.name} in {file_timer.elapsed_str}")
                        except Exception as e:
                            file_timer.__exit__(None, None, None)
                            logger.error(f"Error processing {file_path}: {e}")
                            failed_files.append(file_path)
                
                stats["successful_files"] = len(successful_files)
                stats["failed_files"] = len(failed_files)
                
                if per_file_times:
                    stats["avg_file_time"] = sum(per_file_times) / len(per_file_times)
                    stats["min_file_time"] = min(per_file_times)
                    stats["max_file_time"] = max(per_file_times)
                
                # Update manifest with successfully processed files
                if successful_files:
                    with Timer("Updating manifest with processed files") as t:
                        update_processed_docs_manifest(manifest, successful_files, "direct_processing")
                    stats["timings"]["update_manifest_processed"] = t.elapsed
                    
                logger.info(f"Processing complete. Successfully processed {len(successful_files)} files.")
                if failed_files:
                    logger.error(f"Failed to process {len(failed_files)} files.")
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise
        finally:
            processing_timer.__exit__(None, None, None)
            stats["timings"]["document_processing"] = processing_timer.elapsed
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
    finally:
        # Stop overall timer and add to stats
        overall_timer.__exit__(None, None, None)
        stats["total_time"] = overall_timer.elapsed
    
    # Display performance summary
    logger.info("\n" + "="*40)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*40)
    logger.info(f"Total processing time: {str(timedelta(seconds=stats['total_time']))}")
    logger.info(f"Total files found: {stats['total_files']}")
    if stats.get("skipped_ocr_files", 0) > 0:
        logger.info(f"Files skipped due to OCR: {stats['skipped_ocr_files']}")
    logger.info(f"Files in manifest: {stats['manifest_files']}")
    logger.info(f"Files in database: {stats['database_files']}")
    logger.info(f"New files: {stats['new_files']}")
    logger.info(f"Successfully processed: {stats['successful_files']}")
    if "failed_files" in stats:
        logger.info(f"Failed to process: {stats['failed_files']}")
    
    # Display timing details
    logger.info("\nTiming Details:")
    for stage, elapsed in stats["timings"].items():
        logger.info(f"  {stage}: {str(timedelta(seconds=elapsed))}")
    
    # Display per-file statistics if available
    if "avg_file_time" in stats:
        logger.info("\nPer-file Statistics:")
        logger.info(f"  Average processing time: {str(timedelta(seconds=stats['avg_file_time']))}")
        logger.info(f"  Minimum processing time: {str(timedelta(seconds=stats['min_file_time']))}")
        logger.info(f"  Maximum processing time: {str(timedelta(seconds=stats['max_file_time']))}")
    
    # Display completion message
    logger.info("\nDocument processing completed!")
    logger.info(f"You can now query the system with: python tools/query_cli.py \"Your question here\"")
    logger.info(f"Or use the interactive chat: python tools/chat_cli.py")
    logger.info(f"Manifest updated: {MANIFEST_FILE}")

if __name__ == "__main__":
    main() 