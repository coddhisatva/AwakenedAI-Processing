import os
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import nltk
from nltk.tokenize import sent_tokenize

# Allow importing the metrics framework
try:
    from src.pipeline.metrics import PipelineMetrics, PhaseTimer
except ImportError:
    # For when the chunker is used standalone
    PipelineMetrics = None
    PhaseTimer = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Make sure we have all required NLTK resources
nltk.download('punkt')

class SemanticChunker:
    """
    Handles semantic chunking of document content for better retrieval.
    """
    
    def __init__(
        self, 
        processed_dir: str, 
        chunks_dir: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        unified_metrics: Optional[PipelineMetrics] = None
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            processed_dir: Directory containing processed documents
            chunks_dir: Directory to store chunked documents
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size to keep
            unified_metrics: Optional unified metrics object for integration with the pipeline
        """
        self.processed_dir = Path(processed_dir)
        self.chunks_dir = Path(chunks_dir)
        self.chunks_dir.mkdir(exist_ok=True, parents=True)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Store the unified metrics
        self.unified_metrics = unified_metrics
        
        # Track processing statistics
        self.stats = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0,
            "chunking_time": 0
        }
    
    def chunk_all_documents(self) -> Dict[str, Any]:
        """
        Process all documents in the processed directory.
        
        Returns:
            Statistics about the chunking process
        """
        logger.info(f"Starting chunking of documents from {self.processed_dir}")
        
        # Use a PhaseTimer if unified metrics are available
        timer_context = (
            PhaseTimer("Chunking all documents", self.unified_metrics, phase="chunking")
            if self.unified_metrics and PhaseTimer
            else None
        )
        
        # Start overall timing
        start_time = time.time()
        if timer_context:
            timer_context.__enter__()
        
        try:
            # Find all JSON files
            json_files = list(self.processed_dir.glob("**/*.json"))
            logger.info(f"Found {len(json_files)} processed documents")
            
            # Process each file
            for file_path in json_files:
                self.stats["total_documents"] += 1
                if self.unified_metrics:
                    self.unified_metrics.increment("chunking", "documents_processed")
                
                try:
                    # Skip book_metadata.csv if it exists
                    if file_path.name == "book_metadata.csv":
                        continue
                        
                    num_chunks = self._chunk_document(file_path)
                    self.stats["successful_documents"] += 1
                    self.stats["total_chunks"] += num_chunks
                    
                    if self.unified_metrics:
                        self.unified_metrics.increment("chunking", "successful_documents")
                        self.unified_metrics.increment("chunking", "chunks_created", num_chunks)
                    
                    logger.info(f"Successfully chunked {file_path.name} into {num_chunks} chunks")
                except Exception as e:
                    self.stats["failed_documents"] += 1
                    if self.unified_metrics:
                        self.unified_metrics.increment("chunking", "failed_documents")
                    
                    logger.error(f"Failed to chunk {file_path}: {str(e)}")
            
            # Calculate total chunking time
            self.stats["chunking_time"] = time.time() - start_time
            
            # Calculate throughput
            if self.stats["chunking_time"] > 0:
                chunks_per_second = self.stats["total_chunks"] / self.stats["chunking_time"]
                if self.unified_metrics:
                    self.unified_metrics.update("chunking", "throughput", chunks_per_second)
            
            logger.info(f"Chunking complete. Processed {self.stats['successful_documents']} documents successfully, {self.stats['failed_documents']} failed. Created {self.stats['total_chunks']} total chunks.")
            return self.stats
            
        finally:
            # Close the timer if we created one
            if timer_context:
                timer_context.__exit__(None, None, None)
    
    def _chunk_document(self, file_path: Path) -> int:
        """
        Chunk a single document into semantic chunks.
        
        Args:
            file_path: Path to the processed document
            
        Returns:
            Number of chunks created
        """
        # Create a timer for this document if using unified metrics
        timer_context = (
            PhaseTimer(f"Chunking {file_path.name}", self.unified_metrics, phase="chunking")
            if self.unified_metrics and PhaseTimer
            else None
        )
        
        if timer_context:
            timer_context.__enter__()
            
        try:
            # Load the processed document
            with open(file_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            
            # Extract content and metadata
            content = document.get("content", "")
            metadata = document.get("metadata", {})
            
            if not content:
                logger.warning(f"No content found in {file_path}")
                return 0
            
            # Split the content into sentences
            sentences = self._split_into_sentences(content)
            
            # Create chunks from sentences
            chunks = self._create_chunks_from_sentences(sentences, metadata)
            
            # Save chunks to file
            output_path = self.chunks_dir / f"{file_path.stem}_chunks.json"
            self._save_chunks(output_path, chunks)
            
            return len(chunks)
        finally:
            # Close the timer if we created one
            if timer_context:
                timer_context.__exit__(None, None, None)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Clean the text
        text = self._clean_text(text)
        
        # Use a simpler approach if NLTK fails
        try:
            # Split into sentences using NLTK
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {str(e)}. Using fallback method.")
            # Fallback to a simple regex-based sentence splitter
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        return sentences
    
    def _clean_text(self, text: str) -> str:
        """
        Clean the text for better sentence splitting.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with a single one
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with a single one
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _create_chunks_from_sentences(self, sentences: List[str], doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from a list of sentences.
        
        Args:
            sentences: List of sentences
            doc_metadata: Document metadata
            
        Returns:
            List of chunks with metadata
        """
        if not sentences:
            return []
            
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed the chunk size and we already have content
            if current_chunk_size + sentence_length > self.chunk_size and current_chunk:
                # Save the current chunk
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk_with_metadata(chunk_text, doc_metadata, len(chunks)))
                
                # Start a new chunk with overlap
                overlap_start = max(0, len(current_chunk) - int(self.chunk_overlap / (sentence_length + 1)))
                current_chunk = current_chunk[overlap_start:]
                current_chunk_size = sum(len(s) for s in current_chunk) + len(current_chunk) - 1  # Include spaces
            
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_chunk_size += sentence_length + 1  # +1 for the space
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk_with_metadata(chunk_text, doc_metadata, len(chunks)))
        
        # Update metrics
        if self.unified_metrics:
            self.unified_metrics.increment("chunking", "chunks_created", len(chunks))
        
        return chunks
    
    def _create_chunk_with_metadata(
        self, 
        text: str, 
        doc_metadata: Dict[str, Any], 
        chunk_index: int
    ) -> Dict[str, Any]:
        """
        Create a chunk with metadata.
        
        Args:
            text: Chunk text
            doc_metadata: Document metadata
            chunk_index: Index of the chunk
            
        Returns:
            Chunk with metadata
        """
        return {
            "content": text,
            "metadata": {
                **doc_metadata,
                "chunk_index": chunk_index,
                "chunk_size": len(text)
            }
        }
    
    def _save_chunks(self, output_path: Path, chunks: List[Dict[str, Any]]) -> None:
        """
        Save chunks to a JSON file.
        
        Args:
            output_path: Path to save the chunks
            chunks: List of chunks with metadata
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_path}") 