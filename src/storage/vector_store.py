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
            
            # Extract documents that need embeddings
            docs_needing_embeddings = [doc for doc in documents if "embedding" not in doc or not doc["embedding"]]
            
            if docs_needing_embeddings:
                logger.info(f"Generating embeddings for {len(docs_needing_embeddings)} documents using batch processing")
                
                # Use batch embedding to generate all embeddings at once
                embedded_docs = embedder.create_embeddings(docs_needing_embeddings)
                
                # Update original documents with embeddings
                embedding_map = {
                    doc["content"] if "content" in doc else doc["text"]: doc["embedding"] 
                    for doc in embedded_docs
                }
                
                for doc in documents:
                    content = doc.get("content", doc.get("text", ""))
                    if content in embedding_map:
                        doc["embedding"] = embedding_map[content]
        
        # Organize documents by source filepath to group by document
        docs_by_filepath = {}
        for doc in documents:
            content = doc.get("content", doc.get("text", ""))
            metadata = doc.get("metadata", {})
            embedding = doc.get("embedding", [])
            filepath = metadata.get("source", "")
            
            if filepath not in docs_by_filepath:
                docs_by_filepath[filepath] = {
                    "title": metadata.get("title", "Unknown"),
                    "author": metadata.get("author", "Unknown"),
                    "filepath": filepath,
                    "chunks": []
                }
            
            docs_by_filepath[filepath]["chunks"].append({
                "content": content,
                "metadata": metadata,
                "embedding": embedding
            })
        
        all_chunk_ids = []
        max_batch_size = 200  # Recommended batch size for Supabase
        
        try:
            # Process each document
            for filepath, doc_info in docs_by_filepath.items():
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
                        title=doc_info["title"],
                        author=doc_info["author"],
                        filepath=doc_info["filepath"]
                    )
                
                # Prepare chunks for batch insertion
                chunks_to_insert = []
                for chunk in doc_info["chunks"]:
                    chunks_to_insert.append({
                        "document_id": doc_id,
                        "content": chunk["content"],
                        "metadata": chunk["metadata"],
                        "embedding": chunk["embedding"]
                    })
                
                # Insert chunks in batches of max_batch_size
                total_insertion_time = 0
                total_chunks_inserted = 0
                
                for i in range(0, len(chunks_to_insert), max_batch_size):
                    batch = chunks_to_insert[i:i+max_batch_size]
                    
                    # Log batch information
                    batch_num = i // max_batch_size + 1
                    total_batches = (len(chunks_to_insert) - 1) // max_batch_size + 1
                    logger.info(f"Inserting batch {batch_num}/{total_batches} with {len(batch)} chunks for document {doc_info['title']}")
                    
                    # Measure insertion time
                    start_time = time.time()
                    
                    # Insert the batch
                    chunk_ids = self.adapter.add_chunks_batch(batch)
                    all_chunk_ids.extend(chunk_ids)
                    
                    # Calculate metrics
                    end_time = time.time()
                    batch_time = end_time - start_time
                    total_insertion_time += batch_time
                    total_chunks_inserted += len(chunk_ids)
                    
                    chunks_per_second = len(chunk_ids) / batch_time if batch_time > 0 else 0
                    
                    logger.info(f"Batch {batch_num}/{total_batches} insertion completed in {batch_time:.2f}s ({chunks_per_second:.2f} chunks/second)")
                
                # Log overall performance for this document
                if total_insertion_time > 0 and total_chunks_inserted > 0:
                    avg_chunks_per_second = total_chunks_inserted / total_insertion_time
                    logger.info(f"Document '{doc_info['title']}' processing completed: {total_chunks_inserted} chunks inserted in {total_insertion_time:.2f}s ({avg_chunks_per_second:.2f} chunks/second)")
            
            logger.info(f"Added {len(all_chunk_ids)} chunks to Supabase using batch insertion")
            return all_chunk_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to Supabase: {str(e)}")
            return all_chunk_ids
    
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
        
        # Use batch embedding method to keep code consistent
        query_embeddings = embedder.create_embeddings_batch([query])
        if query_embeddings:
            query_embedding = query_embeddings[0]
        else:
            logger.error("Failed to create query embedding")
            return []
        
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