import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from tqdm import tqdm
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class DocumentEmbedder:
    """
    Generates embeddings for document chunks using OpenAI's API.
    """
    
    def __init__(
        self, 
        chunks_dir: str, 
        embeddings_dir: str,
        embedding_model: str = "text-embedding-ada-002",
        batch_size: int = 100,
        max_retries: int = 5,
        retry_delay: int = 5
    ):
        """
        Initialize the document embedder.
        
        Args:
            chunks_dir: Directory containing document chunks
            embeddings_dir: Directory to store embeddings
            embedding_model: OpenAI embedding model to use
            batch_size: Number of chunks to process in one batch
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
        """
        self.chunks_dir = Path(chunks_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(exist_ok=True, parents=True)
        
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Track processing statistics
        self.stats = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "api_calls": 0,
            "api_errors": 0
        }
    
    def embed_documents(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate embeddings for all document chunks.
        
        Args:
            limit: Optional limit on the number of files to process (for testing)
            
        Returns:
            Statistics about the embedding process
        """
        logger.info(f"Starting embedding generation from {self.chunks_dir}")
        
        # Find all chunk files
        chunk_files = list(self.chunks_dir.glob("**/*_chunks.json"))
        logger.info(f"Found {len(chunk_files)} chunk files")
        
        # Apply limit if specified
        if limit is not None:
            chunk_files = chunk_files[:limit]
            logger.info(f"Processing only {limit} files for testing")
        
        # Process each file
        for file_path in chunk_files:
            self.stats["total_files"] += 1
            try:
                num_chunks = self._embed_document_chunks(file_path)
                self.stats["successful_files"] += 1
                logger.info(f"Successfully embedded {file_path.name} with {num_chunks} chunks")
            except Exception as e:
                self.stats["failed_files"] += 1
                logger.error(f"Failed to embed {file_path}: {str(e)}")
        
        logger.info(f"Embedding complete. Processed {self.stats['successful_files']} files successfully, {self.stats['failed_files']} failed.")
        logger.info(f"Generated embeddings for {self.stats['successful_chunks']} chunks, {self.stats['failed_chunks']} failed.")
        logger.info(f"Made {self.stats['api_calls']} API calls with {self.stats['api_errors']} errors.")
        
        return self.stats
    
    def _embed_document_chunks(self, file_path: Path) -> int:
        """
        Generate embeddings for chunks in a single document.
        
        Args:
            file_path: Path to the chunks file
            
        Returns:
            Number of chunks successfully embedded
        """
        # Load the chunks
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"Processing {len(chunks)} chunks from {file_path.name}")
        self.stats["total_chunks"] += len(chunks)
        
        # Process chunks in batches
        embedded_chunks = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            
            try:
                # Generate embeddings for the batch
                embedded_batch = self._generate_embeddings_batch(batch)
                embedded_chunks.extend(embedded_batch)
                self.stats["successful_chunks"] += len(embedded_batch)
                
                # Add a small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                self.stats["failed_chunks"] += len(batch)
                logger.error(f"Failed to embed batch: {str(e)}")
                
                # If we hit a rate limit, wait longer
                if "rate limit" in str(e).lower():
                    logger.warning(f"Rate limit hit, waiting {self.retry_delay * 2} seconds")
                    time.sleep(self.retry_delay * 2)
        
        # Save the embedded chunks
        output_path = self.embeddings_dir / f"{file_path.stem.replace('_chunks', '')}_embeddings.json"
        self._save_embeddings(output_path, embedded_chunks)
        
        return len(embedded_chunks)
    
    def _generate_embeddings_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a batch of chunks using OpenAI's API.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of chunks with embeddings
        """
        # Extract texts to embed
        texts = [chunk["content"] for chunk in chunks]
        
        # Try to generate embeddings with retries
        for attempt in range(self.max_retries):
            try:
                self.stats["api_calls"] += 1
                
                # Call OpenAI API to generate embeddings
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                
                # Extract embeddings from response
                embeddings = [data.embedding for data in response.data]
                
                # Add embeddings to chunks
                embedded_chunks = []
                for i, chunk in enumerate(chunks):
                    embedded_chunk = {
                        "content": chunk["content"],
                        "metadata": chunk["metadata"],
                        "embedding": embeddings[i]
                    }
                    embedded_chunks.append(embedded_chunk)
                
                return embedded_chunks
                
            except Exception as e:
                self.stats["api_errors"] += 1
                logger.warning(f"API error on attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retrying
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    # Max retries reached, re-raise the exception
                    raise
    
    def _save_embeddings(self, output_path: Path, embedded_chunks: List[Dict[str, Any]]) -> None:
        """
        Save embedded chunks to a JSON file.
        
        Args:
            output_path: Path to save the embeddings
            embedded_chunks: List of chunks with embeddings
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(embedded_chunks, f, ensure_ascii=False)
        
        logger.info(f"Saved {len(embedded_chunks)} embedded chunks to {output_path}") 