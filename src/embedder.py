import os
import json
import logging
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
        
        # Ensure OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Track embedding statistics
        self.stats = {
            "total_chunks": 0,
            "embedded_chunks": 0,
            "failed_chunks": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "documents_processed": 0
        }
    
    def embed_all_chunks(self) -> Dict[str, Any]:
        """
        Generate embeddings for all chunks in the chunks directory.
        
        Returns:
            Statistics about the embedding process
        """
        logger.info(f"Starting embedding generation for chunks in {self.chunks_dir}")
        
        # Get all chunk files
        chunk_files = list(self.chunks_dir.glob("**/*.json"))
        logger.info(f"Found {len(chunk_files)} chunk files")
        
        # Process in batches
        for i in range(0, len(chunk_files), self.batch_size):
            batch = chunk_files[i:i+self.batch_size]
            self._process_batch(batch)
        
        # Calculate estimated cost (as of current OpenAI pricing)
        # Ada embeddings cost $0.0001 per 1K tokens
        self.stats["estimated_cost"] = (self.stats["total_tokens"] / 1000) * 0.0001
        
        logger.info(f"Embedding generation complete. Processed {self.stats['embedded_chunks']} chunks.")
        logger.info(f"Total tokens: {self.stats['total_tokens']}, Estimated cost: ${self.stats['estimated_cost']:.2f}")
        
        return self.stats
    
    def _process_batch(self, chunk_files: List[Path]) -> None:
        """
        Process a batch of chunk files.
        
        Args:
            chunk_files: List of paths to chunk files
        """
        chunks_to_embed = []
        file_paths = []
        
        # Load chunks
        for file_path in chunk_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunk = json.load(f)
                
                chunks_to_embed.append(chunk)
                file_paths.append(file_path)
                self.stats["total_chunks"] += 1
            except Exception as e:
                logger.error(f"Error loading chunk file {file_path}: {str(e)}")
                self.stats["failed_chunks"] += 1
        
        # Generate embeddings
        if chunks_to_embed:
            try:
                # Extract text content for embedding
                texts = [chunk["content"] for chunk in chunks_to_embed]
                
                # Generate embeddings with retry logic
                embeddings, token_count = self._generate_embeddings_with_retry(texts)
                
                # Update stats
                self.stats["total_tokens"] += token_count
                
                # Save embeddings
                for i, (embedding, chunk, file_path) in enumerate(zip(embeddings, chunks_to_embed, file_paths)):
                    # Add embedding to chunk
                    chunk["embedding"] = embedding
                    
                    # Save to embeddings directory
                    doc_dir = file_path.parent.name
                    output_dir = self.embeddings_dir / doc_dir
                    output_dir.mkdir(exist_ok=True, parents=True)
                    
                    output_path = output_dir / file_path.name
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(chunk, f, ensure_ascii=False, indent=2)
                    
                    self.stats["embedded_chunks"] += 1
                
                # Track unique documents processed
                doc_dirs = set(file_path.parent.name for file_path in file_paths)
                self.stats["documents_processed"] = len(doc_dirs)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {str(e)}")
                self.stats["failed_chunks"] += len(chunks_to_embed)
    
    def _generate_embeddings_with_retry(self, texts: List[str]) -> tuple[List[List[float]], int]:
        """
        Generate embeddings with retry logic.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Tuple of (embeddings, token_count)
        """
        retries = 0
        token_count = 0
        
        while retries < self.max_retries:
            try:
                # Create embeddings
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                
                # Extract embeddings
                embeddings = [data.embedding for data in response.data]
                
                # Get token count
                token_count = response.usage.total_tokens
                
                return embeddings, token_count
                
            except Exception as e:
                retries += 1
                logger.warning(f"Embedding API call failed (attempt {retries}/{self.max_retries}): {str(e)}")
                
                if retries < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to generate embeddings after {self.max_retries} attempts")
                    raise
        
        # This should never be reached due to the raise in the loop
        raise RuntimeError("Failed to generate embeddings")
    
    def estimate_embedding_cost(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Estimate the cost of embedding all chunks based on a sample.
        
        Args:
            sample_size: Number of chunks to sample for estimation
            
        Returns:
            Dictionary with cost estimation details
        """
        # Get all chunk files
        chunk_files = list(self.chunks_dir.glob("**/*.json"))
        total_chunks = len(chunk_files)
        
        if total_chunks == 0:
            return {
                "estimated_total_chunks": 0,
                "estimated_total_tokens": 0,
                "estimated_total_cost": 0.0
            }
        
        # Sample chunks
        import random
        sample_size = min(sample_size, total_chunks)
        sample_files = random.sample(chunk_files, sample_size)
        
        # Load sample chunks and count tokens
        total_sample_tokens = 0
        
        for file_path in sample_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunk = json.load(f)
                
                # Estimate tokens (rough approximation: 4 chars per token)
                content = chunk.get("content", "")
                estimated_tokens = len(content) // 4
                total_sample_tokens += estimated_tokens
                
            except Exception as e:
                logger.error(f"Error loading sample chunk file {file_path}: {str(e)}")
        
        # Calculate averages and estimates
        avg_tokens_per_chunk = total_sample_tokens / sample_size
        estimated_total_tokens = avg_tokens_per_chunk * total_chunks
        
        # Ada embeddings cost $0.0001 per 1K tokens
        estimated_cost = (estimated_total_tokens / 1000) * 0.0001
        
        return {
            "sample_size": sample_size,
            "avg_tokens_per_chunk": avg_tokens_per_chunk,
            "estimated_total_chunks": total_chunks,
            "estimated_total_tokens": estimated_total_tokens,
            "estimated_total_cost": estimated_cost
        } 