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
        
        # Track processing statistics
        self.stats = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0
        }
    
    def chunk_all_documents(self) -> Dict[str, Any]:
        """
        Process all documents in the processed directory.
        
        Returns:
            Statistics about the chunking process
        """
        logger.info(f"Starting chunking of documents from {self.processed_dir}")
        
        # Find all JSON files
        json_files = list(self.processed_dir.glob("**/*.json"))
        logger.info(f"Found {len(json_files)} processed documents")
        
        # Process each file
        for file_path in json_files:
            self.stats["total_documents"] += 1
            try:
                # Skip book_metadata.csv if it exists
                if file_path.name == "book_metadata.csv":
                    continue
                    
                num_chunks = self._chunk_document(file_path)
                self.stats["successful_documents"] += 1
                self.stats["total_chunks"] += num_chunks
                logger.info(f"Successfully chunked {file_path.name} into {num_chunks} chunks")
            except Exception as e:
                self.stats["failed_documents"] += 1
                logger.error(f"Failed to chunk {file_path}: {str(e)}")
        
        logger.info(f"Chunking complete. Processed {self.stats['successful_documents']} documents successfully, {self.stats['failed_documents']} failed. Created {self.stats['total_chunks']} total chunks.")
        return self.stats
    
    def _chunk_document(self, file_path: Path) -> int:
        """
        Chunk a single document into semantic chunks.
        
        Args:
            file_path: Path to the processed document
            
        Returns:
            Number of chunks created
        """
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
        Clean text by removing extra whitespace and normalizing.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with a single one
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with a single one
        text = re.sub(r' +', ' ', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def _create_chunks_from_sentences(
        self, 
        sentences: List[str], 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from sentences with overlap.
        
        Args:
            sentences: List of sentences
            metadata: Document metadata
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed the chunk size and we already have content,
            # save the current chunk and start a new one with overlap
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk with metadata
                chunk_text = " ".join(current_chunk)
                chunk = self._create_chunk_with_metadata(chunk_text, metadata, len(chunks) + 1)
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_size = 0
                overlap_chunk = []
                
                # Add sentences from the end of the previous chunk for overlap
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += len(s) + 1  # +1 for space
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
        
        # Add the last chunk if it's not empty and meets minimum size
        if current_chunk and current_size >= self.min_chunk_size:
            chunk_text = " ".join(current_chunk)
            chunk = self._create_chunk_with_metadata(chunk_text, metadata, len(chunks) + 1)
            chunks.append(chunk)
        
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