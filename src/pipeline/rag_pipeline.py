import os
import json
import time
import argparse
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

from dotenv import load_dotenv
from src.extraction.extractor import DocumentExtractor
from src.processing.chunker import SemanticChunker
from src.embedding.embedder import DocumentEmbedder
from src.storage.vector_store import VectorStoreBase, SupabaseVectorStore
from database.supabase_adapter import SupabaseAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class RAGPipeline:
    """
    Complete RAG pipeline that handles all document processing steps:
    - Intelligent document filtering (manifest & database checks)
    - Text extraction
    - Semantic chunking
    - Embedding generation
    - Vector database storage
    
    The pipeline also provides:
    - Manifest management for tracking processed files
    - Performance statistics and timing
    - OCR file handling
    - Command-line interface
    """
    
    def __init__(
        self,
        extractor: DocumentExtractor,
        chunker: SemanticChunker,
        embedder: DocumentEmbedder,
        vector_store: VectorStoreBase,
        manifest_path: Optional[str] = None,
        skip_ocr: bool = False
    ):
        """
        Initialize the RAG pipeline with all components.
        
        Args:
            extractor: Document extractor for text extraction
            chunker: Semantic chunker for breaking documents into chunks
            embedder: Document embedder for creating vector embeddings
            vector_store: Vector store for storing embeddings
            manifest_path: Path to the manifest file (default: data/processed_documents.json)
            skip_ocr: Whether to skip files that need OCR processing
        """
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.skip_ocr = skip_ocr
        
        # Set up database adapter for duplicate checking
        self.db_adapter = SupabaseAdapter()
        
        # Statistics tracking
        self.stats = {
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
        
        # Set up manifest path
        if manifest_path:
            self.manifest_path = Path(manifest_path)
        else:
            # Default manifest path
            self.manifest_path = Path(os.path.join("data", "processed_documents.json"))
        
        # Create manifest directory if it doesn't exist
        self.manifest_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Load manifest
        self.manifest = self._load_manifest()
        
        logger.info("RAG Pipeline initialized with all components")
    
    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the manifest of processed documents.
        
        Returns:
            Dictionary with filenames as keys and processing metadata as values
        """
        if not self.manifest_path.exists():
            logger.info(f"No manifest file found at {self.manifest_path}. Creating a new one.")
            return {}
        
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
            logger.info(f"Loaded manifest with {len(manifest)} processed documents.")
            return manifest
        except Exception as e:
            logger.error(f"Error loading manifest file: {e}")
            return {}
    
    def _update_manifest(self, files: List[Path], source_type: str) -> None:
        """
        Update the manifest with new files.
        
        Args:
            files: List of file paths to add
            source_type: Type of source ('direct_processing' or 'database_duplicate')
        """
        count = 0
        
        # Update the manifest with new files
        for file_path in files:
            if file_path.name not in self.manifest:
                self.manifest[file_path.name] = {
                    "filepath": str(file_path),
                    "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": source_type
                }
                count += 1
        
        # Write the updated manifest
        try:
            with open(self.manifest_path, 'w') as f:
                json.dump(self.manifest, f, indent=2)
            if count > 0:
                logger.info(f"Updated manifest file with {count} new {source_type} documents.")
        except Exception as e:
            logger.error(f"Error updating manifest file: {e}")
    
    def _needs_ocr(self, file_path: Path) -> bool:
        """
        Check if a PDF file likely needs OCR processing.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            True if the file likely needs OCR, False otherwise
        """
        if file_path.suffix.lower() != '.pdf':
            return False
            
        # Simple check based on filename patterns for known OCR-needing files
        ocr_patterns = [
            "frank-rudolph-young",
            "cyclomancy",
            "yoga-for-men-only"
        ]
        
        for pattern in ocr_patterns:
            if pattern.lower() in file_path.name.lower():
                return True
                
        # For other files, we'd need to actually check if text can be extracted
        # This is a simple approximation for this script
        return False
    
    def _find_all_files(self, directory_path: Path, extensions: List[str]) -> Tuple[List[Path], int]:
        """
        Get all files with the specified extensions in the directory.
        
        Args:
            directory_path: Path to search for files
            extensions: List of file extensions to include
            
        Returns:
            Tuple of (all_files, skipped_ocr_count)
        """
        all_files = []
        for ext in extensions:
            all_files.extend(list(directory_path.glob(f"**/*.{ext}")))
        
        # Skip OCR files if requested
        skipped_ocr_count = 0
        if self.skip_ocr:
            # Find OCR-needing files
            ocr_files = [f for f in all_files if self._needs_ocr(f)]
            
            if ocr_files:
                logger.info(f"Skipping {len(ocr_files)} files that need OCR: {', '.join(f.name for f in ocr_files)}")
                all_files = [f for f in all_files if f not in ocr_files]
                skipped_ocr_count = len(ocr_files)
        
        return all_files, skipped_ocr_count
    
    def _categorize_files(self, files: List[Path], force_reprocess: bool = False) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Categorize files into three groups:
        1. Already in manifest (skip)
        2. Not in manifest but in database (add to manifest as database_duplicate)
        3. Not in manifest and not in database (process normally)
        
        Args:
            files: List of file paths to categorize
            force_reprocess: Whether to process all files regardless of manifest
            
        Returns:
            Tuple of (manifest_files, database_files, new_files)
        """
        manifest_files = []
        database_files = []
        new_files = []
        
        # Get set of filenames already in manifest
        manifest_filenames = set(self.manifest.keys())
        
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
                        existing_doc = self.db_adapter.get_document_by_filepath(str(file_path))
                    
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
    
    def process_files(self, file_paths: List[str], batch_size: int = 5) -> int:
        """
        Process a specific list of files through the pipeline.
        
        Args:
            file_paths: List of file paths to process
            batch_size: Number of documents to process in each batch
            
        Returns:
            Number of chunks created
        """
        logger.info(f"Processing {len(file_paths)} specific files")
        
        # Extract text from documents
        extracted_docs = []
        for file_path in file_paths:
            path = Path(file_path)
            if path.suffix.lower() == '.pdf':
                doc = self.extractor.extract_pdf(path)
                if doc:
                    extracted_docs.append(doc)
            elif path.suffix.lower() == '.epub':
                doc = self.extractor.extract_epub(path)
                if doc:
                    extracted_docs.append(doc)
            else:
                logger.warning(f"Unsupported file type: {path}")
        
        logger.info(f"Extracted {len(extracted_docs)} documents")
        
        # Process the extracted documents through chunking, embedding, and storage
        return self._process_extracted_docs(extracted_docs, batch_size)
    
    def _process_extracted_docs(self, extracted_docs: List[Dict[str, Any]], batch_size: int) -> int:
        """
        Process extracted documents through chunking, embedding, and storage.
        
        Args:
            extracted_docs: List of extracted documents
            batch_size: Number of documents to process in each batch
            
        Returns:
            Number of chunks created
        """
        total_chunks = 0
        doc_ids = []
        
        # Process documents in batches
        for i in range(0, len(extracted_docs), batch_size):
            batch = extracted_docs[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(extracted_docs)-1)//batch_size + 1}")
            
            # Create chunks for each document
            all_chunks = []
            for doc in batch:
                chunks = self.chunker.create_chunks(doc["text"])
                # Add document metadata to each chunk
                for chunk in chunks:
                    chunk["metadata"] = {
                        "source": doc["source"],
                        "title": doc.get("title", "Unknown"),
                        "page": doc.get("page", None),
                        "chapter": doc.get("chapter", None)
                    }
                all_chunks.extend(chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks from batch")
            
            # Create embeddings for chunks
            embedded_chunks = self.embedder.create_embeddings(all_chunks)
            logger.info(f"Created embeddings for {len(embedded_chunks)} chunks")
            
            # Store chunks in vector database
            ids = self.vector_store.add_documents(embedded_chunks)
            doc_ids.extend(ids)
            total_chunks += len(ids)
            
            logger.info(f"Stored {len(ids)} chunks in vector database")
            
        logger.info(f"Pipeline processing complete. Total chunks stored: {total_chunks}")
        return total_chunks
    
    def process_directory(self, directory_path: str, batch_size: int = 5, 
                         extensions: List[str] = ["pdf", "epub", "txt"],
                         force_reprocess: bool = False, 
                         update_manifest_only: bool = False) -> Dict[str, Any]:
        """
        Process all documents in a directory through the pipeline,
        with intelligent filtering using manifest and database checks.
        
        Args:
            directory_path: Path to directory containing documents
            batch_size: Number of documents to process in each batch
            extensions: List of file extensions to include
            force_reprocess: Whether to reprocess files already in manifest or database
            update_manifest_only: Only update manifest with database files, don't process any
            
        Returns:
            Statistics about the processing run
        """
        start_time = time.time()
        overall_timer = Timer("Total processing time")
        overall_timer.__enter__()
        
        try:
            # Find all files in the directory
            with Timer("Finding files") as t:
                input_dir = Path(directory_path)
                all_files, skipped_ocr_count = self._find_all_files(input_dir, extensions)
            self.stats["timings"]["find_files"] = t.elapsed
            
            if not all_files:
                logger.error(f"No files found with extensions {extensions} in {input_dir}")
                return self.stats
                
            self.stats["total_files"] = len(all_files)
            self.stats["skipped_ocr_files"] = skipped_ocr_count
            logger.info(f"Found {len(all_files)} total files with extensions {extensions}")
            
            # Categorize files
            with Timer("Categorizing files") as t:
                manifest_files, database_files, new_files = self._categorize_files(
                    all_files, 
                    force_reprocess=force_reprocess
                )
            self.stats["timings"]["categorize_files"] = t.elapsed
            self.stats["manifest_files"] = len(manifest_files)
            self.stats["database_files"] = len(database_files)
            self.stats["new_files"] = len(new_files)
            
            # Update manifest with database files
            if database_files:
                with Timer("Updating manifest with database files") as t:
                    self._update_manifest(database_files, "database_duplicate")
                self.stats["timings"]["update_manifest_db"] = t.elapsed
            
            # If only updating manifest, we're done
            if update_manifest_only:
                logger.info("Manifest updated with database duplicates. Exiting without processing files.")
                return self.stats
            
            # If no new files to process, we're done
            if not new_files and not force_reprocess:
                logger.info("No new files to process. All documents are either in manifest or database.")
                return self.stats
            
            # Process files
            processing_timer = Timer("Document processing")
            processing_timer.__enter__()
            
            try:
                # If forcing reprocessing of all files
                if force_reprocess:
                    logger.info(f"Force reprocessing {len(all_files)} files")
                    
                    # Get list of all file paths as strings
                    all_file_paths = [str(f) for f in all_files]
                    
                    # Process files directly
                    total_chunks = self.process_files(all_file_paths, batch_size)
                    
                    self.stats["total_chunks"] = total_chunks
                    self.stats["successful_files"] = len(all_files)
                    
                    # Update manifest with all files as directly processed
                    with Timer("Updating manifest with all files") as t:
                        self._update_manifest(all_files, "direct_processing")
                    self.stats["timings"]["update_manifest_all"] = t.elapsed
                else:
                    # Only process new files
                    logger.info(f"Processing {len(new_files)} new files")
                    successful_files = []
                    failed_files = []
                    
                    # Create a temporary directory for processing only the new files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        logger.info(f"Created temporary directory for processing: {temp_path}")
                        
                        # Create symbolic links to the new files in the temp directory
                        for file_path in new_files:
                            if file_path.exists():
                                link_path = temp_path / file_path.name
                                os.symlink(file_path, link_path)
                        
                        # Process all files in the temporary directory
                        try:
                            # Get the file paths in the temp directory
                            temp_file_paths = [str(f) for f in temp_path.glob("*.*")]
                            
                            # Process the files
                            total_chunks = self.process_files(temp_file_paths, batch_size)
                            successful_files = new_files
                            self.stats["total_chunks"] = total_chunks
                        except Exception as e:
                            logger.error(f"Error processing directory: {e}")
                            failed_files = new_files
                    
                    self.stats["successful_files"] = len(successful_files)
                    self.stats["failed_files"] = len(failed_files)
                    
                    # Update manifest with successfully processed files
                    if successful_files:
                        with Timer("Updating manifest with processed files") as t:
                            self._update_manifest(successful_files, "direct_processing")
                        self.stats["timings"]["update_manifest_processed"] = t.elapsed
                    
                    logger.info(f"Processing complete. Successfully processed {len(successful_files)} files.")
                    if failed_files:
                        logger.error(f"Failed to process {len(failed_files)} files.")
            
            except Exception as e:
                logger.error(f"Error processing documents: {e}")
                raise
            finally:
                processing_timer.__exit__(None, None, None)
                self.stats["timings"]["document_processing"] = processing_timer.elapsed
            
        except Exception as e:
            logger.error(f"Error in main processing: {e}")
        finally:
            # Stop overall timer and add to stats
            overall_timer.__exit__(None, None, None)
            self.stats["total_time"] = overall_timer.elapsed
        
        # Log performance summary
        self._log_performance_summary()
        
        return self.stats
    
    def _log_performance_summary(self) -> None:
        """Log a summary of the processing performance."""
        logger.info("\n" + "="*40)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*40)
        logger.info(f"Total processing time: {str(timedelta(seconds=self.stats['total_time']))}")
        logger.info(f"Total files found: {self.stats['total_files']}")
        if self.stats.get("skipped_ocr_files", 0) > 0:
            logger.info(f"Files skipped due to OCR: {self.stats['skipped_ocr_files']}")
        logger.info(f"Files in manifest: {self.stats['manifest_files']}")
        logger.info(f"Files in database: {self.stats['database_files']}")
        logger.info(f"New files: {self.stats['new_files']}")
        logger.info(f"Successfully processed: {self.stats['successful_files']}")
        if "failed_files" in self.stats:
            logger.info(f"Failed to process: {self.stats['failed_files']}")
        
        # Display timing details
        logger.info("\nTiming Details:")
        for stage, elapsed in self.stats["timings"].items():
            logger.info(f"  {stage}: {str(timedelta(seconds=elapsed))}")
        
        # Display completion message
        logger.info("\nDocument processing completed!")
        logger.info(f"You can now query the system with: python tools/query_cli.py \"Your question here\"")
        logger.info(f"Or use the interactive chat: python tools/chat_cli.py")
        logger.info(f"Manifest updated: {self.manifest_path}")
    
    def query(self, query_text: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query the RAG pipeline for relevant documents.
        
        Args:
            query_text: The query text
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Querying RAG pipeline: '{query_text}'")
        return self.vector_store.search(query_text, k, filter_dict)

    @classmethod
    def run_cli(cls):
        """Run the RAG pipeline from the command line."""
        # Parse arguments
        parser = argparse.ArgumentParser(description="RAG Pipeline for document processing")
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
        parser.add_argument(
            "--manifest_path",
            type=str,
            help="Path to the manifest file. Default is data/processed_documents.json",
            default=os.path.join("data", "processed_documents.json")
        )
        
        args = parser.parse_args()
        
        # Load environment variables
        load_dotenv()
        
        # Define directories
        data_dir = Path(os.path.abspath(os.path.join("data")))
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
        vector_store = SupabaseVectorStore()
        
        # Create and run the pipeline
        pipeline = cls(
            extractor=extractor, 
            chunker=chunker, 
            embedder=embedder, 
            vector_store=vector_store,
            manifest_path=args.manifest_path,
            skip_ocr=args.skip_ocr
        )
        
        # Process the directory
        pipeline.process_directory(
            directory_path=args.input_dir,
            batch_size=args.batch_size,
            extensions=args.extensions,
            force_reprocess=args.force_reprocess,
            update_manifest_only=args.update_manifest_only
        )

if __name__ == "__main__":
    RAGPipeline.run_cli()
