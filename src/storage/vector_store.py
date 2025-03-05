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
                # This is a simple approach - you might want more sophisticated duplicate detection
                doc_id = None
                
                # Add document if it doesn't exist
                if doc_id is None:
                    doc_id = self.adapter.add_document(
                        title=title,
                        author=author,
                        filepath=filepath
                    )
                
                # Add the chunk with its embedding
                chunk_id = self.adapter.add_chunk(
                    document_id=doc_id,
                    content=content,
                    metadata=metadata,
                    embedding=embedding
                )
                
                document_ids.append(chunk_id)
                
            except Exception as e:
                logger.error(f"Error adding document to Supabase: {str(e)}")
        
        logger.info(f"Added {len(document_ids)} chunks to Supabase")
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
        
        logger.info(f"Searching Supabase for: '{query}' with k={k}")
        
        # Generate embedding for the query
        embedder = DocumentEmbedder()
        query_embedding = embedder.get_embedding(query)
        
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