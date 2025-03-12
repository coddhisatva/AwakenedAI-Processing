import os
import json
import logging
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tiktoken
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
        batch_size: int = 500,  # Increased as per batch strategy
        max_retries: int = 5,
        retry_delay: int = 5,
        max_tokens_per_batch: int = 8000  # Just below the 8,191 token limit
    ):
        """
        Initialize the document embedder.
        
        Args:
            chunks_dir: Directory containing document chunks
            embeddings_dir: Directory to store embeddings
            embedding_model: OpenAI embedding model to use
            batch_size: Maximum number of chunks to process in one batch (will be reduced if token limit is exceeded)
            max_retries: Maximum number of retries for API calls
            retry_delay: Base delay between retries in seconds
            max_tokens_per_batch: Maximum number of tokens allowed in a batch
        """
        self.chunks_dir = Path(chunks_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(exist_ok=True, parents=True)
        
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_tokens_per_batch = max_tokens_per_batch
        
        # Initialize token counter
        if "ada" in embedding_model:
            # For text-embedding-ada-002, use cl100k_base encoding (same as used by OpenAI)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            # For other models, use their specific encoding if available
            self.tokenizer = tiktoken.encoding_for_model(embedding_model)
        
        # Track rate limits
        self.last_minute_requests = []
        self.rate_limit_per_minute = 3000  # Assuming Tier 2 (pay-as-you-go)
        
        # Track processing statistics
        self.stats = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "api_calls": 0,
            "api_errors": 0,
            "tokens_processed": 0,
            "embedding_time": 0,
            "average_batch_size": 0,
            "total_batches": 0
        }
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings as lists of floats
        """
        if not texts:
            return []
            
        # Log the batch size
        logger.info(f"Generating embeddings for batch of {len(texts)} texts")
        
        # Count tokens for monitoring
        total_tokens = sum(len(self.tokenizer.encode(text)) for text in texts)
        self.stats["tokens_processed"] += total_tokens
        
        # Self-throttle if needed
        self._respect_rate_limits()
        
        # Track timing for performance monitoring
        start_time = time.time()
        
        # Try to generate embeddings with retries and exponential backoff
        for attempt in range(self.max_retries):
            try:
                self.stats["api_calls"] += 1
                self.last_minute_requests.append(time.time())
                
                # Call OpenAI API to generate embeddings
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                
                # Extract embeddings from response
                embeddings = [data.embedding for data in response.data]
                
                # Update timing statistics
                embedding_time = time.time() - start_time
                self.stats["embedding_time"] += embedding_time
                
                # Update batch statistics
                self.stats["total_batches"] += 1
                self.stats["average_batch_size"] = (
                    (self.stats["average_batch_size"] * (self.stats["total_batches"] - 1) + len(texts))
                    / self.stats["total_batches"]
                )
                
                # Log performance metrics
                embeddings_per_second = len(embeddings) / embedding_time if embedding_time > 0 else 0
                tokens_per_second = total_tokens / embedding_time if embedding_time > 0 else 0
                logger.info(f"Batch embedding completed in {embedding_time:.2f}s ({embeddings_per_second:.2f} embeddings/second, {tokens_per_second:.2f} tokens/second)")
                
                return embeddings
                
            except Exception as e:
                self.stats["api_errors"] += 1
                error_message = str(e).lower()
                
                # Add specific handling for different error types
                if "rate limit" in error_message:
                    wait_time = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time:.2f}s before retry.")
                    time.sleep(wait_time)
                    
                    # Reduce batch size if we keep hitting rate limits
                    if attempt >= 2 and len(texts) > 10:
                        reduced_size = max(10, len(texts) // 2)
                        logger.warning(f"Repeatedly hitting rate limits. Consider reducing batch_size from {self.batch_size} to {reduced_size}")
                        
                elif "token" in error_message and ("limit" in error_message or "exceed" in error_message):
                    # Token limit exceeded - this should be prevented by our token counting, but just in case
                    if len(texts) <= 1:
                        # Can't reduce further, something else is wrong
                        logger.error(f"Token limit exceeded with single text: {error_message}")
                        raise
                    
                    # Suggest reducing batch size for future calls
                    reduced_size = max(1, len(texts) // 2)
                    logger.warning(f"Token limit exceeded. Consider reducing batch_size to {reduced_size} or max_tokens_per_batch")
                    raise
                    
                elif "timeout" in error_message or "timed out" in error_message:
                    # For timeouts, use longer waits
                    wait_time = self.retry_delay * (3 ** attempt) + random.uniform(0, 2)
                    logger.warning(f"Request timed out. Waiting {wait_time:.2f}s before retry.")
                    time.sleep(wait_time)
                    
                else:
                    # General error handling with exponential backoff
                    wait_time = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"API error on attempt {attempt + 1}/{self.max_retries}: {error_message}. Retrying in {wait_time:.2f}s.")
                    time.sleep(wait_time)
                
                # If this is the last attempt, re-raise the exception
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to generate embeddings after {self.max_retries} attempts: {str(e)}")
                    raise
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create embeddings for a list of chunks, handling batching and token limits.
        
        Args:
            chunks: List of chunks to embed, each with at least a 'content' field
            
        Returns:
            List of chunks with added 'embedding' field
        """
        if not chunks:
            return []
            
        logger.info(f"Creating embeddings for {len(chunks)} chunks using batching")
        
        # Sort chunks by token count to optimize batch composition
        chunks_with_tokens = self._count_tokens_for_chunks(chunks)
        chunks_with_tokens.sort(key=lambda x: x[1])  # Sort by token count
        
        # Group chunks into batches that respect token limits
        batches = self._create_token_efficient_batches(chunks_with_tokens)
        logger.info(f"Divided {len(chunks)} chunks into {len(batches)} batches based on token limits")
        
        # Process each batch
        embedded_chunks = []
        for i, batch in enumerate(batches):
            batch_chunks = [chunk for chunk, _ in batch]
            batch_texts = [chunk["content"] for chunk in batch_chunks]
            
            try:
                # Get embeddings for the batch
                batch_embeddings = self.create_embeddings_batch(batch_texts)
                
                # Add embeddings to chunks
                for j, chunk in enumerate(batch_chunks):
                    embedded_chunk = chunk.copy()
                    embedded_chunk["embedding"] = batch_embeddings[j]
                    embedded_chunks.append(embedded_chunk)
                    
            except Exception as e:
                logger.error(f"Error processing batch {i+1}/{len(batches)}: {str(e)}")
                # Continue with other batches
        
        logger.info(f"Successfully created embeddings for {len(embedded_chunks)}/{len(chunks)} chunks")
        return embedded_chunks
    
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
        
        # Calculate final statistics
        self._log_final_statistics()
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
        
        # Process all chunks using batching
        embedded_chunks = self.create_embeddings(chunks)
        self.stats["successful_chunks"] += len(embedded_chunks)
        
        # Save the embedded chunks
        output_path = self.embeddings_dir / f"{file_path.stem.replace('_chunks', '')}_embeddings.json"
        self._save_embeddings(output_path, embedded_chunks)
        
        return len(embedded_chunks)
    
    def _count_tokens_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], int]]:
        """
        Count tokens for each chunk.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of tuples (chunk, token_count)
        """
        return [(chunk, len(self.tokenizer.encode(chunk["content"]))) for chunk in chunks]
    
    def _create_token_efficient_batches(self, chunks_with_tokens: List[Tuple[Dict[str, Any], int]]) -> List[List[Tuple[Dict[str, Any], int]]]:
        """
        Create batches that respect both the maximum batch size and token limits.
        
        Args:
            chunks_with_tokens: List of tuples (chunk, token_count)
            
        Returns:
            List of batches, where each batch is a list of (chunk, token_count) tuples
        """
        batches = []
        current_batch = []
        current_token_count = 0
        
        for chunk_with_tokens in chunks_with_tokens:
            chunk, token_count = chunk_with_tokens
            
            # Check if adding this chunk would exceed token limit or batch size
            if (current_token_count + token_count > self.max_tokens_per_batch or 
                len(current_batch) >= self.batch_size):
                # Current batch is full, start a new one
                if current_batch:  # Only append if we have items
                    batches.append(current_batch)
                current_batch = [chunk_with_tokens]
                current_token_count = token_count
            else:
                # Add to current batch
                current_batch.append(chunk_with_tokens)
                current_token_count += token_count
        
        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)
            
        return batches
    
    def _respect_rate_limits(self):
        """
        Self-throttle to respect OpenAI API rate limits.
        """
        # Remove requests older than 1 minute
        current_time = time.time()
        self.last_minute_requests = [t for t in self.last_minute_requests if current_time - t < 60]
        
        # Check if we're approaching the rate limit
        if len(self.last_minute_requests) >= self.rate_limit_per_minute * 0.95:
            # Wait until we're safely under the limit
            wait_time = 60 - (current_time - self.last_minute_requests[0]) + 1  # +1 for safety
            logger.warning(f"Approaching rate limit. Waiting {wait_time:.2f}s to avoid hitting the limit.")
            time.sleep(wait_time)
    
    def _log_final_statistics(self):
        """Log comprehensive statistics about the embedding process."""
        embeddings_per_second = 0
        if self.stats["embedding_time"] > 0:
            embeddings_per_second = self.stats["successful_chunks"] / self.stats["embedding_time"]
        
        tokens_per_second = 0
        if self.stats["embedding_time"] > 0:
            tokens_per_second = self.stats["tokens_processed"] / self.stats["embedding_time"]
        
        logger.info(f"Embedding complete. Processed {self.stats['successful_files']} files successfully, {self.stats['failed_files']} failed.")
        logger.info(f"Generated embeddings for {self.stats['successful_chunks']} chunks, {self.stats['failed_chunks']} failed.")
        logger.info(f"Made {self.stats['api_calls']} API calls with {self.stats['api_errors']} errors.")
        logger.info(f"Processed {self.stats['tokens_processed']} tokens in {self.stats['embedding_time']:.2f} seconds.")
        logger.info(f"Performance: {embeddings_per_second:.2f} embeddings/second, {tokens_per_second:.2f} tokens/second.")
        logger.info(f"Average batch size: {self.stats['average_batch_size']:.2f} chunks per batch.")
    
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