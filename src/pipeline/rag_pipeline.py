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
from src.pipeline.metrics import PipelineMetrics, PhaseTimer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        skip_ocr: bool = False,
        metrics_dir: Optional[str] = None
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
            metrics_dir: Optional directory to save metrics for persistence
        """
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.skip_ocr = skip_ocr
        
        # Set up database adapter for duplicate checking
        self.db_adapter = SupabaseAdapter(manifest_path=manifest_path)
        
        # Initialize metrics
        self.metrics = PipelineMetrics(metrics_dir)
        
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
        
        # Initialize failed documents manifest as None (will be set during processing)
        self.failed_manifest_path = None
        self.failed_manifest = {}
        
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
            # Check for both lowercase and uppercase versions of the extension
            all_files.extend(list(directory_path.glob(f"**/*.{ext.lower()}")))
            all_files.extend(list(directory_path.glob(f"**/*.{ext.upper()}")))
        
        # No pre-filtering for OCR needed anymore, just return all files
        return all_files, 0
    
    def _categorize_files(self, files: List[Path], force_reprocess: bool = False) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Categorize files into three groups:
        1. Already in manifest (skip)
        2. Not in manifest but in database (add to manifest as database_duplicate)
        3. Not in manifest and not in database (process normally)
        
        Files in the failed documents manifest will be skipped unless force_reprocess is True.
        
        Args:
            files: List of file paths to categorize
            force_reprocess: Whether to process all files regardless of manifest
            
        Returns:
            Tuple of (manifest_files, database_files, new_files)
        """
        manifest_files = []
        database_files = []
        new_files = []
        failed_files = []
        
        # Get set of filenames already in manifest
        manifest_filenames = set(self.manifest.keys())
        
        # Get set of filenames already in failed manifest
        failed_filenames = set(self.failed_manifest.keys()) if hasattr(self, 'failed_manifest') else set()
        
        # Stats for timing database checks
        db_check_count = 0
        db_match_count = 0
        
        # First pass: identify files in manifest or failed manifest
        with PhaseTimer("Manifest check", self.metrics) as manifest_timer:
            for file_path in files:
                if file_path.name in manifest_filenames and not force_reprocess:
                    manifest_files.append(file_path)
                elif file_path.name in failed_filenames and not force_reprocess:
                    failed_files.append(file_path)
                    logger.info(f"Skipping previously failed file: {file_path.name}")
                else:
                    # For files not in manifest/failed manifest or being forced to reprocess, 
                    # check if they're in database
                    db_check_count += 1
                    existing_doc = None
                    with PhaseTimer("Database lookup", self.metrics) as db_timer:
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
        
        if failed_files:
            logger.info(f"Found {len(failed_files)} files in failed documents manifest (will be skipped)")
            
        if db_check_count > 0:
            avg_db_check_time = manifest_timer.elapsed / db_check_count if db_check_count > 0 else 0
            logger.info(f"Performed {db_check_count} database checks, found {db_match_count} matches")
            logger.info(f"Average database check time: {avg_db_check_time:.4f} seconds")
            
        logger.info(f"Found {len(new_files)} new files to process")
        
        # Update metrics for failed files
        self.metrics.update("overall", "failed_manifest_files", len(failed_files))
        
        return manifest_files, database_files, new_files
    
    # Add this helper function for counting tokens
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        This is a simple approximation - OpenAI's tokenizer may count slightly differently.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Approximate number of tokens
        """
        # A simple approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4
        
    def _split_chunk_by_tokens(self, chunk: Dict[str, Any], max_tokens: int = 8000) -> List[Dict[str, Any]]:
        """
        Split a large chunk into smaller chunks based on token count.
        
        Args:
            chunk: The chunk to split
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of smaller chunks
        """
        text = chunk["content"]
        total_tokens = self._count_tokens(text)
        
        if total_tokens <= max_tokens:
            return [chunk]
            
        # If chunk is too large, split it
        logger.warning(f"Chunk too large ({total_tokens} tokens). Splitting into smaller chunks.")
        
        # Estimate characters per token
        chars_per_token = len(text) / total_tokens
        max_chars = int(max_tokens * chars_per_token * 0.9)  # 10% margin of safety
        
        # Split the text into smaller chunks
        result_chunks = []
        for i in range(0, len(text), max_chars):
            sub_text = text[i:i + max_chars]
            if sub_text:
                result_chunks.append({
                    "content": sub_text,
                    "metadata": chunk["metadata"].copy()
                })
                
        logger.info(f"Split large chunk into {len(result_chunks)} smaller chunks.")
        return result_chunks

    def process_files(self, file_paths: List[str], batch_size: int) -> int:
        """
        Process a list of files through the pipeline.
        
        Args:
            file_paths: List of file paths to process
            batch_size: Number of documents to process in each batch
            
        Returns:
            Number of chunks created and stored
        """
        extracted_docs = []
        failed_files = []
        
        # Process each file and extract text
        with PhaseTimer("Extraction", self.metrics, phase="extraction") as extraction_timer:
            for file_path in file_paths:
                path = Path(file_path)
                
                # Update extraction metrics
                self.metrics.increment("extraction", "files_processed")
                
                # Track format-specific metrics
                extension = path.suffix.lower()
                if extension == ".pdf":
                    self.metrics.increment("extraction", "pdf_files")
                elif extension == ".epub":
                    self.metrics.increment("extraction", "epub_files")
                else:
                    self.metrics.increment("extraction", "other_files")
                
                try:
                    # Extract document
                    doc_dict = None
                    if extension == ".pdf":
                        doc_dict = self.extractor.extract_pdf(path)
                    elif extension == ".epub":
                        doc_dict = self.extractor.extract_epub(path)
                    elif extension == ".txt":
                        doc_dict = self.extractor.extract_text(path)
                    else:
                        doc_dict = self.extractor.extract_document(path)
                    
                    if doc_dict:
                        # Add source info to the extracted doc
                        doc_dict["source"] = str(path)
                        
                        # Add batch_id to the metadata if available from the extractor
                        if hasattr(self.extractor, 'batch_id') and self.extractor.batch_id:
                            if 'metadata' not in doc_dict:
                                doc_dict['metadata'] = {}
                            doc_dict['metadata']['batch_id'] = self.extractor.batch_id
                        
                        extracted_docs.append(doc_dict)
                        # Update metrics for successful extraction
                        self.metrics.increment("extraction", "successful_files")
                    else:
                        # Update metrics for failed extraction
                        logger.error(f"Failed to extract content from {path}")
                        self.metrics.increment("extraction", "failed_files")
                        failed_files.append(path)
                
                except Exception as e:
                    # Enhanced error logging with detailed information about the error and file
                    error_type = type(e).__name__
                    error_msg = f"Error extracting {path} ({extension} file, size: {os.path.getsize(path)/1024:.1f} KB): {str(e)}"
                    
                    # Include original exception details if available
                    if hasattr(e, '__cause__') and e.__cause__:
                        error_msg += f" | Caused by: {type(e.__cause__).__name__}: {e.__cause__}"
                        
                    logger.error(error_msg)
                    
                    # Update metrics for failed extraction
                    self.metrics.increment("extraction", "failed_files")
                    failed_files.append(path)
        
        # Process the extracted documents
        logger.info(f"Extracted {len(extracted_docs)} documents")
        
        # Update failed manifest with files that failed during extraction
        if failed_files:
            self._update_failed_manifest(failed_files, "Extraction error")
        
        # Generate chunks, embeddings, and store in vector database
        total_chunks = self._process_extracted_docs(extracted_docs, batch_size)
        
        # Update the overall metrics for successful files
        self.metrics.update("overall", "successful_files", len(extracted_docs))
        
        return total_chunks
    
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
        failed_docs = []  # Track documents that fail in any stage
        
        # Process documents in batches
        for i in range(0, len(extracted_docs), batch_size):
            batch = extracted_docs[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(extracted_docs)-1)//batch_size + 1}")
            
            # Create chunks for each document
            all_chunks = []
            with PhaseTimer("Chunking", self.metrics, phase="chunking") as chunking_timer:
                for doc in batch:
                    # Update metrics for chunking
                    self.metrics.increment("chunking", "documents_processed")
                    
                    # Split text into sentences
                    sentences = self.chunker._split_into_sentences(doc["text"])
                    
                    # Create document metadata
                    metadata = {
                        "source": doc["source"],
                        "title": doc.get("title", "Unknown"),
                        "author": doc.get("author", "Unknown")
                    }
                    
                    # Create chunks from sentences
                    try:
                        chunks = self.chunker._create_chunks_from_sentences(sentences, metadata)
                        all_chunks.extend(chunks)
                        self.metrics.increment("chunking", "successful_documents")
                        self.metrics.increment("chunking", "chunks_created", len(chunks))
                    except Exception as e:
                        error_msg = f"Error creating chunks for {doc['source']}: {str(e)}"
                        logger.error(error_msg)
                        self.metrics.increment("chunking", "failed_documents")
                        # Add to failed docs list
                        failed_docs.append(Path(doc["source"]))
                        # Update failed manifest immediately
                        self._update_failed_manifest([Path(doc["source"])], f"Chunking error: {str(e)}")
            
            logger.info(f"Created {len(all_chunks)} chunks from batch")
            
            # If no chunks were created, skip to the next batch
            if not all_chunks:
                logger.warning("No chunks created in this batch. Skipping to next batch.")
                continue
                
            # Check for and split large chunks before embedding
            token_safe_chunks = []
            with PhaseTimer("Token checking", self.metrics, phase="chunking") as token_timer:
                for chunk in all_chunks:
                    # Split large chunks into smaller ones
                    split_chunks = self._split_chunk_by_tokens(chunk, max_tokens=8000)
                    token_safe_chunks.extend(split_chunks)
                    
            if len(token_safe_chunks) > len(all_chunks):
                logger.info(f"Split {len(all_chunks)} chunks into {len(token_safe_chunks)} smaller chunks to comply with token limits")
                # Update chunking metrics with the split chunks
                self.metrics.update("chunking", "chunks_created", len(token_safe_chunks))
            
            # Generate embeddings for the chunks using batch processing
            embedded_chunks = []
            with PhaseTimer("Embedding", self.metrics, phase="embedding") as embedding_timer:
                try:
                    # Update metrics
                    self.metrics.increment("embedding", "chunks_processed", len(token_safe_chunks))
                    
                    # Use the batch embedding method to process all chunks at once
                    embedded_chunks = self.embedder.create_embeddings(token_safe_chunks)
                    
                    # Update metrics with embedding results
                    self.metrics.increment("embedding", "successful_chunks", len(embedded_chunks))
                    if len(embedded_chunks) < len(token_safe_chunks):
                        self.metrics.increment("embedding", "failed_chunks", len(token_safe_chunks) - len(embedded_chunks))
                    
                    # Transfer embedding stats to our unified metrics
                    if hasattr(self.embedder, 'stats'):
                        if "tokens_processed" in self.embedder.stats:
                            self.metrics.update("embedding", "tokens_processed", self.embedder.stats["tokens_processed"])
                        if "api_calls" in self.embedder.stats:
                            self.metrics.update("embedding", "api_calls", self.embedder.stats["api_calls"])
                        if "api_errors" in self.embedder.stats:
                            self.metrics.update("embedding", "api_errors", self.embedder.stats["api_errors"])
                        if "total_batches" in self.embedder.stats:
                            self.metrics.update("embedding", "total_batches", self.embedder.stats["total_batches"])
                        if "average_batch_size" in self.embedder.stats:
                            self.metrics.update("embedding", "avg_batch_size", self.embedder.stats["average_batch_size"])
                    
                    logger.info(f"Created embeddings for {len(embedded_chunks)} chunks using batch processing")
                except Exception as e:
                    error_msg = f"Error generating embeddings: {str(e)}"
                    logger.error(error_msg)
                    embedded_chunks = []
                    self.metrics.increment("embedding", "failed_chunks", len(token_safe_chunks))
                    
                    # Add all sources from this batch to failed docs
                    sources = set()
                    for chunk in token_safe_chunks:
                        if "metadata" in chunk and "source" in chunk["metadata"]:
                            sources.add(chunk["metadata"]["source"])
                    
                    # Update failed manifest with the sources
                    for source in sources:
                        failed_docs.append(Path(source))
                    
                    if sources:
                        self._update_failed_manifest([Path(s) for s in sources], f"Embedding error: {str(e)}")
            
            # Store chunks in vector database using the correct method
            ids = []
            with PhaseTimer("Storage", self.metrics, phase="storage") as storage_timer:
                if embedded_chunks:
                    # Update metrics
                    self.metrics.increment("storage", "chunks_stored", len(embedded_chunks))
                    
                    try:
                        ids = self.vector_store.add_documents(embedded_chunks)
                        doc_ids.extend(ids)
                        total_chunks += len(ids)
                        self.metrics.increment("storage", "successful_chunks", len(ids))
                        
                        # If this is a SupabaseVectorStore, we can estimate batch info
                        if isinstance(self.vector_store, SupabaseVectorStore):
                            # A rough estimate based on recommended batch size of 200
                            batches = (len(embedded_chunks) + 199) // 200
                            self.metrics.increment("storage", "total_batches", batches)
                            self.metrics.update("storage", "avg_batch_size", len(embedded_chunks) / batches if batches > 0 else 0)
                        
                        logger.info(f"Stored {len(ids)} chunks in vector database")
                    except Exception as e:
                        error_msg = f"Error storing chunks: {str(e)}"
                        logger.error(error_msg)
                        self.metrics.increment("storage", "failed_chunks", len(embedded_chunks))
                        
                        # Add all sources from this batch to failed docs
                        sources = set()
                        for chunk in embedded_chunks:
                            if "metadata" in chunk and "source" in chunk["metadata"]:
                                sources.add(chunk["metadata"]["source"])
                        
                        # Update failed manifest with the sources
                        for source in sources:
                            failed_docs.append(Path(source))
                        
                        if sources:
                            self._update_failed_manifest([Path(s) for s in sources], f"Storage error: {str(e)}")
            
        # Update the total chunks count in the overall metrics
        self.metrics.update("overall", "total_chunks", total_chunks)
        
        # Log the number of failed documents
        if failed_docs:
            failed_doc_count = len(set(str(doc) for doc in failed_docs))
            logger.info(f"Pipeline processing complete. Total chunks stored: {total_chunks}. Failed documents: {failed_doc_count}")
        else:
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
        # Reset metrics for this run
        self.metrics.reset()
        
        try:
            # Find all files in the directory
            with PhaseTimer("Finding files", self.metrics) as t:
                input_dir = Path(directory_path)
                all_files, skipped_ocr_count = self._find_all_files(input_dir, extensions)
            
            # Extract the subdirectory name from the directory path
            subdir = Path(directory_path).name
            
            # Load the failed documents manifest for this subdirectory
            self.failed_manifest = self._load_failed_manifest(subdir)
            
            if not all_files:
                logger.error(f"No files found with extensions {extensions} in {input_dir}")
                return self.metrics.stats
                
            # Update overall metrics
            self.metrics.update("overall", "total_files", len(all_files))
            self.metrics.update("overall", "skipped_ocr_files", skipped_ocr_count)
            logger.info(f"Found {len(all_files)} total files with extensions {extensions}")
            
            # Categorize files
            with PhaseTimer("Categorizing files", self.metrics) as t:
                manifest_files, database_files, new_files = self._categorize_files(
                    all_files, 
                    force_reprocess=force_reprocess
                )
            
            # Update metrics for categorization
            self.metrics.update("overall", "manifest_files", len(manifest_files))
            self.metrics.update("overall", "database_files", len(database_files))
            self.metrics.update("overall", "new_files", len(new_files))
            
            # If only updating manifest, we're done
            if update_manifest_only:
                logger.info("Manifest updated with database duplicates. Exiting without processing files.")
                self.metrics.log_summary()
                if self.metrics.metrics_dir:
                    self.metrics.save()
                return self.metrics.stats
            
            # If no new files to process, we're done
            if not new_files and not force_reprocess:
                logger.info("No new files to process. All documents are either in manifest or database.")
                self.metrics.log_summary()
                if self.metrics.metrics_dir:
                    self.metrics.save()
                return self.metrics.stats
            
            # Process files
            with PhaseTimer("Document processing", self.metrics) as processing_timer:
                try:
                    # If forcing reprocessing of all files
                    if force_reprocess:
                        logger.info(f"Force reprocessing {len(all_files)} files")
                        
                        # Get list of all file paths as strings
                        all_file_paths = [str(f) for f in all_files]
                        
                        # Process files directly
                        total_chunks = self.process_files(all_file_paths, batch_size)
                        
                        self.metrics.update("overall", "total_chunks", total_chunks)
                        self.metrics.update("overall", "successful_files", len(all_files))
                        
                        # Update manifest with all files as directly processed
                        # We don't need to do this anymore since the adapter updates the manifest on insertion
                        # But we keep the timing for metrics
                        with PhaseTimer("Updating manifest with all files", self.metrics) as t:
                            # The adapter will handle manifest updates when files are added to the DB
                            pass
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
                                    os.symlink(file_path.absolute(), link_path)
                            
                            # Process all files in the temporary directory
                            try:
                                # Get the file paths in the temp directory
                                temp_file_paths = [str(f) for f in temp_path.glob("*.*")]
                                
                                # Process the files
                                total_chunks = self.process_files(temp_file_paths, batch_size)
                                successful_files = new_files
                                self.metrics.update("overall", "total_chunks", total_chunks)
                            except Exception as e:
                                error_msg = f"Error processing directory: {e}"
                                logger.error(error_msg)
                                failed_files = new_files
                                # Update failed manifest with the failed files
                                self._update_failed_manifest(failed_files, f"Directory processing error: {str(e)}")
                        
                        self.metrics.update("overall", "successful_files", len(successful_files))
                        self.metrics.update("overall", "failed_files", len(failed_files))
                        
                        # Update manifest with successfully processed files
                        # We don't need to do this anymore since the adapter updates the manifest on insertion
                        # But we keep the timing for metrics
                        if successful_files:
                            with PhaseTimer("Updating manifest with processed files", self.metrics) as t:
                                # The adapter will handle manifest updates when files are added to the DB
                                pass
                        logger.info(f"Processing complete. Successfully processed {len(successful_files)} files.")
                        if failed_files:
                            logger.error(f"Failed to process {len(failed_files)} files.")
                
                except Exception as e:
                    error_msg = f"Error processing documents: {e}"
                    logger.error(error_msg)
                    # If there was an error at this level, mark all new files as failed
                    if new_files:
                        self._update_failed_manifest(new_files, error_msg)
                        self.metrics.update("overall", "failed_files", len(new_files))
                    raise
            
        except Exception as e:
            error_msg = f"Error in main processing: {e}"
            logger.error(error_msg)
            # If we have the subdirectory loaded and there are files, mark them as failed
            if hasattr(self, 'failed_manifest_path') and self.failed_manifest_path and all_files:
                self._update_failed_manifest(all_files, error_msg)
        
        # Log performance summary
        self._log_performance_summary()
        
        return self.metrics.stats
    
    def _log_performance_summary(self) -> None:
        """Log a summary of the processing performance."""
        # Use the new metrics summary method
        self.metrics.log_summary()
        
        # Save metrics if persistence is enabled
        if self.metrics.metrics_dir:
            self.metrics.save()
    
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

    def _load_failed_manifest(self, subdir: str) -> Dict[str, Dict[str, Any]]:
        """
        Load the manifest of failed documents for a specific subdirectory.
        
        Args:
            subdir: Subdirectory name (e.g., 'group0A')
            
        Returns:
            Dictionary with filenames as keys and failure metadata as values
        """
        # Set up the failed documents directory and file path
        failed_dir = Path(os.path.join("data", "failed", subdir))
        self.failed_manifest_path = failed_dir / "failed_documents.json"
        
        # Create directory if it doesn't exist
        failed_dir.mkdir(exist_ok=True, parents=True)
        
        if not self.failed_manifest_path.exists():
            logger.info(f"No failed manifest file found at {self.failed_manifest_path}. Creating a new one.")
            return {}
        
        try:
            with open(self.failed_manifest_path, 'r') as f:
                failed_manifest = json.load(f)
            logger.info(f"Loaded failed manifest with {len(failed_manifest)} failed documents.")
            return failed_manifest
        except Exception as e:
            logger.error(f"Error loading failed manifest file: {e}")
            return {}
    
    def _update_failed_manifest(self, files: List[Path], error_message: str) -> None:
        """
        Update the failed documents manifest with new failed files.
        
        Args:
            files: List of file paths that failed to process
            error_message: Error message or reason for failure
        """
        if not self.failed_manifest_path:
            logger.error("Failed manifest path not set. Cannot update failed manifest.")
            return
            
        count = 0
        
        # Update the failed manifest with new files
        for file_path in files:
            if file_path.name not in self.failed_manifest:
                self.failed_manifest[file_path.name] = {
                    "filepath": str(file_path),
                    "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error": error_message
                }
                count += 1
        
        # Write the updated failed manifest
        try:
            with open(self.failed_manifest_path, 'w') as f:
                json.dump(self.failed_manifest, f, indent=2)
            if count > 0:
                logger.info(f"Updated failed manifest file with {count} new failed documents.")
        except Exception as e:
            logger.error(f"Error updating failed manifest file: {e}")

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
            "--subdir",
            type=str,
            help="Subdirectory within input_dir to process (e.g., 'group0' or 'WIZARD-AI-2')",
            required=True
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
            help="Skip OCR processing for files that fail text extraction"
        )
        parser.add_argument(
            "--manifest_path",
            type=str,
            help="Path to the manifest file. Default is data/processed_documents.json",
            default=os.path.join("data", "processed_documents.json")
        )
        parser.add_argument(
            "--metrics_dir",
            type=str,
            help="Directory to save metrics data. Default is data/metrics",
            default=os.path.join("data", "metrics")
        )
        
        args = parser.parse_args()
        
        # Load environment variables
        load_dotenv()
        
        # Define directories
        data_dir = Path(os.path.abspath(os.path.join("data")))
        # Combine input_dir with the required subdir
        raw_dir = Path(args.input_dir) / args.subdir
        logger.info(f"Processing subdirectory: {raw_dir}")
            
        processed_dir = data_dir / "processed"
        chunks_dir = data_dir / "chunks"
        embeddings_dir = data_dir / "embeddings"
        metrics_dir = Path(args.metrics_dir)
        
        # Create directories if they don't exist
        for dir_path in [raw_dir, processed_dir, chunks_dir, embeddings_dir, metrics_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        extractor = DocumentExtractor(raw_dir=raw_dir, processed_dir=processed_dir, skip_ocr=args.skip_ocr)
        chunker = SemanticChunker(processed_dir=processed_dir, chunks_dir=chunks_dir)
        embedder = DocumentEmbedder(chunks_dir=chunks_dir, embeddings_dir=embeddings_dir)
        vector_store = SupabaseVectorStore(manifest_path=args.manifest_path)
        
        # Create and run the pipeline
        pipeline = cls(
            extractor=extractor, 
            chunker=chunker, 
            embedder=embedder, 
            vector_store=vector_store,
            manifest_path=args.manifest_path,
            skip_ocr=args.skip_ocr,
            metrics_dir=args.metrics_dir
        )
        
        # Process the directory
        pipeline.process_directory(
            directory_path=str(raw_dir),  # Use the combined path with subdir
            batch_size=args.batch_size,
            extensions=args.extensions,
            force_reprocess=args.force_reprocess,
            update_manifest_only=args.update_manifest_only
        )

if __name__ == "__main__":
    RAGPipeline.run_cli()
