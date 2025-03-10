import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import sys
import time

# Add import for SupabaseAdapter
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from database.supabase_adapter import SupabaseAdapter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreBase(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for documents similar to the query."""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        pass

class ChromaVectorStore(VectorStoreBase):
    """Chroma implementation of vector store."""
    
    def __init__(self, collection_name: str, persist_directory: str, embedding_function=None):
        """Initialize Chroma vector store."""
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Set up ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Set up embedding function (use OpenAI by default)
        if embedding_function is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-ada-002"
            )
        else:
            self.embedding_function = embedding_function
            
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        logger.info(f"Initialized Chroma vector store with collection: {collection_name}")
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with text, metadata, and embeddings
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add to vector store")
            return []
            
        ids = [str(doc.get("id", i)) for i, doc in enumerate(documents)]
        texts = [doc.get("content", doc.get("text", "")) for doc in documents]  # Handle both content and text keys
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        # If embeddings are pre-computed, use them
        if "embedding" in documents[0]:
            embeddings = [doc["embedding"] for doc in documents]
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
        else:
            # Let Chroma compute embeddings
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
        logger.info(f"Added {len(ids)} documents to vector store")
        return ids
    
    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of document dictionaries with text, metadata, and score
        """
        logger.info(f"Searching for: '{query}' with k={k}")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": results["distances"][0][i] if "distances" in results and results["distances"] else None
                })
                
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document dictionary or None if not found
        """
        try:
            result = self.collection.get(ids=[doc_id])
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0] if result["metadatas"] else {}
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None

class PineconeVectorStore(VectorStoreBase):
    """Pinecone implementation of vector store (placeholder for future use)."""
    
    def __init__(self, index_name: str, namespace: str = ""):
        """
        Initialize Pinecone vector store.
        
        Note: This is a skeleton implementation for future use.
        Pinecone will be implemented for production deployment.
        """
        self.index_name = index_name
        self.namespace = namespace
        logger.info("Note: Pinecone implementation is a placeholder for future production use")
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the vector store (placeholder).
        
        Args:
            documents: List of document dictionaries with text, metadata, and embeddings
            
        Returns:
            List of document IDs
        """
        logger.warning("Pinecone implementation is a placeholder. Documents not actually stored.")
        return [doc.get("id", str(i)) for i, doc in enumerate(documents)]
    
    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query (placeholder).
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            Empty list (placeholder)
        """
        logger.warning("Pinecone implementation is a placeholder. No search results returned.")
        return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID (placeholder).
        
        Args:
            doc_id: Document ID
            
        Returns:
            None (placeholder)
        """
        logger.warning("Pinecone implementation is a placeholder. No document returned.")
        return None

class SupabaseVectorStore(VectorStoreBase):
    """Supabase implementation of vector store."""
    
    def __init__(self):
        """Initialize Supabase vector store."""
        try:
            self.adapter = SupabaseAdapter()
            logger.info(f"Initialized Supabase vector store")
        except Exception as e:
            logger.error(f"Error initializing Supabase adapter: {str(e)}")
            raise
            
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with text, metadata, and embeddings
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add to vector store")
            return []
        
        document_ids = []
        batch_size = 200  # Optimal batch size from strategy document
        chunks_batch = []
        total_docs = len(documents)
        processed_docs = 0
        start_time = time.time()
        total_batches = 0
        
        logger.info(f"Starting batch insertion process for {total_docs} documents with batch size {batch_size}")
        
        # Check if we need to generate embeddings
        need_embeddings = False
        for doc in documents:
            if "embedding" not in doc or not doc["embedding"]:
                need_embeddings = True
                break
        
        # Generate embeddings if needed
        if need_embeddings:
            from src.embedding.embedder import DocumentEmbedder
            import os
            from pathlib import Path
            
            # Set up directories for the embedder
            data_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"))
            chunks_dir = data_dir / "chunks"
            embeddings_dir = data_dir / "embeddings"
            
            # Create directories if they don't exist
            chunks_dir.mkdir(exist_ok=True, parents=True)
            embeddings_dir.mkdir(exist_ok=True, parents=True)
            
            # Initialize embedder
            embedder = DocumentEmbedder(
                chunks_dir=str(chunks_dir), 
                embeddings_dir=str(embeddings_dir)
            )
            
            # Create embeddings for each document without one
            for doc in documents:
                if "embedding" not in doc or not doc["embedding"]:
                    content = doc.get("content", doc.get("text", ""))
                    doc["embedding"] = embedder.create_embedding(content)
        
        # Process each document
        for doc in documents:
            try:
                # Extract document metadata
                content = doc.get("content", doc.get("text", ""))
                metadata = doc.get("metadata", {})
                embedding = doc.get("embedding", [])
                
                # Extract document information from metadata
                title = metadata.get("title", "Unknown")
                author = metadata.get("author", "Unknown")
                filepath = metadata.get("source", "")
                
                # Check if we already have this document based on filepath
                doc_id = None
                if filepath:
                    existing_doc = self.adapter.get_document_by_filepath(filepath)
                    if existing_doc:
                        doc_id = existing_doc["id"]
                        logger.info(f"Document {filepath} already exists in database, using existing document ID")
                
                # Add document if it doesn't exist
                if doc_id is None:
                    doc_id = self.adapter.add_document(
                        title=title,
                        author=author,
                        filepath=filepath
                    )
                
                # Instead of adding the chunk immediately, add it to our batch
                chunks_batch.append({
                    "document_id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "embedding": embedding
                })
                
                # If we've reached the batch size, insert the batch
                if len(chunks_batch) >= batch_size:
                    batch_start = time.time()
                    
                    # Start a transaction for this batch
                    try:
                        self.adapter.begin_transaction()
                        batch_ids = self.adapter.add_chunks_batch(chunks_batch)
                        # Transaction is automatically committed in add_chunks_batch if it started there
                        
                        batch_duration = time.time() - batch_start
                        total_batches += 1
                        
                        document_ids.extend(batch_ids)
                        logger.info(f"Added batch {total_batches} with {len(batch_ids)} chunks to Supabase "
                                   f"in {batch_duration:.2f} seconds ({len(batch_ids)/batch_duration:.2f} chunks/second)")
                        chunks_batch = []  # Reset batch
                    except Exception as e:
                        # Transaction is automatically rolled back in add_chunks_batch if it started there
                        logger.error(f"Error in batch {total_batches + 1}: {str(e)}")
                        # Continue processing other documents despite this batch failure
                
            except Exception as e:
                logger.error(f"Error preparing document for Supabase: {str(e)}")
            
            # Update progress
            processed_docs += 1
            if processed_docs % 10 == 0 or processed_docs == total_docs:
                elapsed = time.time() - start_time
                docs_per_second = processed_docs / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {processed_docs}/{total_docs} documents processed "
                           f"({processed_docs/total_docs*100:.1f}%) - "
                           f"{docs_per_second:.2f} docs/second")
        
        # Insert any remaining chunks
        if chunks_batch:
            try:
                batch_start = time.time()
                
                # Start a transaction for this final batch
                self.adapter.begin_transaction()
                batch_ids = self.adapter.add_chunks_batch(chunks_batch)
                # Transaction is automatically committed in add_chunks_batch if it started there
                
                batch_duration = time.time() - batch_start
                total_batches += 1
                
                document_ids.extend(batch_ids)
                logger.info(f"Added final batch {total_batches} with {len(batch_ids)} chunks to Supabase "
                           f"in {batch_duration:.2f} seconds ({len(batch_ids)/batch_duration:.2f} chunks/second)")
            except Exception as e:
                # Transaction is automatically rolled back in add_chunks_batch if it started there
                logger.error(f"Error adding final batch to Supabase: {str(e)}")
                # Fallback to individual insertion if batch fails
                for chunk_data in chunks_batch:
                    try:
                        chunk_id = self.adapter.add_chunk(
                            document_id=chunk_data["document_id"],
                            content=chunk_data["content"],
                            metadata=chunk_data["metadata"],
                            embedding=chunk_data["embedding"]
                        )
                        document_ids.append(chunk_id)
                    except Exception as inner_e:
                        logger.error(f"Error adding individual chunk to Supabase: {str(inner_e)}")
        
        logger.info(f"Added a total of {len(document_ids)} chunks to Supabase")
        return document_ids
    
    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_dict: Optional metadata filters (not implemented for Supabase yet)
            
        Returns:
            List of document dictionaries with text, metadata, and score
        """
        from src.embedding.embedder import DocumentEmbedder
        import os
        from pathlib import Path
        
        logger.info(f"Searching Supabase for: '{query}' with k={k}")
        
        # Set up directories for the embedder
        data_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"))
        chunks_dir = data_dir / "chunks"
        embeddings_dir = data_dir / "embeddings"
        
        # Create directories if they don't exist
        chunks_dir.mkdir(exist_ok=True, parents=True)
        embeddings_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate embedding for the query
        embedder = DocumentEmbedder(chunks_dir=str(chunks_dir), embeddings_dir=str(embeddings_dir))
        query_embedding = embedder.create_embedding(query)
        
        # Search using the query embedding
        results = self.adapter.search(query_embedding=query_embedding, limit=k)
        
        # Format results to match the interface
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.get("id", ""),
                "text": result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "score": result.get("similarity", 0.0)
            })
        
        logger.info(f"Found {len(formatted_results)} results in Supabase")
        return formatted_results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID (in this case, a chunk ID)
            
        Returns:
            Document dictionary or None if not found
        """
        try:
            # Get the chunk by ID
            chunk = self.adapter.get_chunk_by_id(doc_id)
            
            if chunk:
                # Get the associated document
                document = self.adapter.get_document_by_id(chunk.get("document_id", ""))
                
                return {
                    "id": chunk.get("id", ""),
                    "text": chunk.get("content", ""),
                    "metadata": {
                        **chunk.get("metadata", {}),
                        "document_title": document.get("title", "") if document else "",
                        "document_author": document.get("author", "") if document else ""
                    }
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id} from Supabase: {str(e)}")
            return None