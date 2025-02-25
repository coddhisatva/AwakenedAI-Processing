import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import nltk
from nltk.tokenize import sent_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
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
        min_chunk_size: int = 100
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            processed_dir: Directory containing processed documents
            chunks_dir: Directory to store chunked documents
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size to keep
        """
        self.processed_dir = Path(processed_dir)
        self.chunks_dir = Path(chunks_dir)
        self.chunks_dir.mkdir(exist_ok=True, parents=True)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Track chunking statistics
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_chunks_per_document": 0,
            "documents_processed": 0,
            "documents_failed": 0
        }
    
    def chunk_all_documents(self) -> Dict[str, Any]:
        """
        Process all documents in the processed directory.
        
        Returns:
            Statistics about the chunking process
        """
        logger.info(f"Starting chunking of documents from {self.processed_dir}")
        
        document_count = 0
        chunk_count = 0
        
        for file_path in self.processed_dir.glob("**/*.json"):
            if file_path.is_file():
                self.stats["total_documents"] += 1
                try:
                    num_chunks = self._chunk_document(file_path)
                    chunk_count += num_chunks
                    document_count += 1
                    self.stats["documents_processed"] += 1
                except Exception as e:
                    self.stats["documents_failed"] += 1
                    logger.error(f"Failed to chunk {file_path}: {str(e)}")
        
        # Calculate average chunks per document
        if document_count > 0:
            self.stats["avg_chunks_per_document"] = chunk_count / document_count
        
        self.stats["total_chunks"] = chunk_count
        
        logger.info(f"Chunking complete. Created {chunk_count} chunks from {document_count} documents.")
        return self.stats
    
    def _chunk_document(self, file_path: Path) -> int:
        """
        Chunk a single document into semantic chunks.
        
        Args:
            file_path: Path to the processed document JSON file
            
        Returns:
            Number of chunks created
        """
        logger.info(f"Chunking document: {file_path}")
        
        try:
            # Load the document
            with open(file_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            
            content = document.get("content", "")
            metadata = document.get("metadata", {})
            
            if not content:
                logger.warning(f"No content found in {file_path}")
                return 0
            
            # Create chunks
            chunks = self._create_semantic_chunks(content, metadata)
            
            # Save chunks
            output_dir = self.chunks_dir / file_path.stem
            output_dir.mkdir(exist_ok=True, parents=True)
            
            for i, chunk in enumerate(chunks):
                chunk_file = output_dir / f"chunk_{i+1:04d}.json"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Created {len(chunks)} chunks for {file_path}")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error chunking {file_path}: {str(e)}")
            raise
    
    def _create_semantic_chunks(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from document content.
        
        Args:
            content: Document content text
            metadata: Document metadata
            
        Returns:
            List of chunk objects with content and metadata
        """
        # Split content into paragraphs
        paragraphs = self._split_into_paragraphs(content)
        
        # Create initial chunks by combining paragraphs
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed the chunk size and we already have content,
            # save the current chunk and start a new one
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)
        
        # Create overlapping chunks if needed
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._create_overlapping_chunks(paragraphs)
        
        # Format chunks with metadata
        formatted_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_obj = {
                "chunk_id": i + 1,
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i + 1,
                    "total_chunks": len(chunks)
                }
            }
            formatted_chunks.append(chunk_obj)
        
        return formatted_chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Document text
            
        Returns:
            List of paragraphs
        """
        # Split by double newlines to get paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _create_overlapping_chunks(self, paragraphs: List[str]) -> List[str]:
        """
        Create chunks with overlap from paragraphs.
        
        Args:
            paragraphs: List of document paragraphs
            
        Returns:
            List of text chunks with overlap
        """
        chunks = []
        current_chunk = ""
        current_size = 0
        last_paragraph_index = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph_size = len(paragraph)
            
            # If adding this paragraph would exceed the chunk size and we already have content,
            # save the current chunk and start a new one with overlap
            if current_size + paragraph_size > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, i - max(1, self.chunk_overlap // 100))  # Use paragraph count for overlap
                current_chunk = "\n\n".join(paragraphs[overlap_start:i]) + "\n\n" + paragraph
                current_size = sum(len(p) for p in paragraphs[overlap_start:i+1]) + (i - overlap_start) * 4  # Add newlines
                last_paragraph_index = i
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                    current_size += paragraph_size + 4  # Add newlines
                else:
                    current_chunk = paragraph
                    current_size = paragraph_size
                last_paragraph_index = i
        
        # Add the last chunk if it's not empty
        if current_chunk and current_size >= self.min_chunk_size:
            chunks.append(current_chunk)
        
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Alternative chunking method that respects sentence boundaries.
        
        Args:
            text: Document text
            
        Returns:
            List of chunks that respect sentence boundaries
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed the chunk size and we already have content,
            # save the current chunk and start a new one
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Start new chunk with overlap if possible
                if self.chunk_overlap > 0:
                    # Find a good starting point for overlap
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    # Find the first sentence boundary in the overlap
                    sentence_boundary = overlap_text.find('. ') + 2
                    if sentence_boundary > 2:  # Found a boundary
                        current_chunk = current_chunk[-self.chunk_overlap + sentence_boundary:] + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it's not empty
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)
        
        return chunks 