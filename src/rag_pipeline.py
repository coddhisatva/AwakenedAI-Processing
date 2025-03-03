import os

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.extraction.extractor import DocumentExtractor
from src.processing.chunker import SemanticChunker
from src.embedding.embedder import DocumentEmbedder
from src.storage.vector_store import ChromaVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Pipeline that connects extraction, chunking, embedding, and storage."""
    
    def __init__(
        self,
        extractor: DocumentExtractor,
        chunker: SemanticChunker,
        embedder: DocumentEmbedder,
        vector_store: ChromaVectorStore
    ):
        """Initialize the RAG pipeline with all components."""
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        logger.info("RAG Pipeline initialized with all components")
        
    def process_directory(self, directory_path: str, batch_size: int = 10) -> int:
        """
        Process all documents in a directory through the pipeline.
        
        Args:
            directory_path: Path to directory containing documents
            batch_size: Number of documents to process in each batch
            
        Returns:
            Number of documents processed successfully
        """
        logger.info(f"Processing directory: {directory_path} with batch size {batch_size}")
        
        # Extract text from documents
        extracted_docs = self.extractor.extract_directory(directory_path)
        logger.info(f"Extracted {len(extracted_docs)} documents")
        
        total_chunks = 0
        doc_ids = []
        
        # Process documents in batches
        for i in range(0, len(extracted_docs), batch_size):
            batch = extracted_docs[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(extracted_docs)-1)//batch_size + 1}")
            
            # Create chunks for each document
            all_chunks = []
            for doc in batch:
                chunks = self.chunker.create_chunks(doc["text"])
                # Add document metadata to each chunk
                for chunk in chunks:
                    chunk["metadata"] = {
                        "source": doc["source"],
                        "title": doc.get("title", "Unknown"),
                        "page": doc.get("page", None),
                        "chapter": doc.get("chapter", None)
                    }
                all_chunks.extend(chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks from batch")
            
            # Create embeddings for chunks
            embedded_chunks = self.embedder.create_embeddings(all_chunks)
            logger.info(f"Created embeddings for {len(embedded_chunks)} chunks")
            
            # Store chunks in vector database
            ids = self.vector_store.add_documents(embedded_chunks)
            doc_ids.extend(ids)
            total_chunks += len(ids)
            
            logger.info(f"Stored {len(ids)} chunks in vector database")
            
        logger.info(f"Pipeline processing complete. Total chunks stored: {total_chunks}")
        return total_chunks

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
