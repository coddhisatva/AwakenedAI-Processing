# src/interface/query.py
import logging
from typing import List, Dict, Any, Optional
from src.pipeline.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryEngine:
    """Interface for querying the RAG system."""
    
    def __init__(self, pipeline: RAGPipeline):
        """Initialize with a RAG pipeline."""
        self.pipeline = pipeline
        logger.info("QueryEngine initialized")
    
    def query(self, 
              query_text: str, 
              k: int = 5, 
              filter_dict: Optional[Dict[str, Any]] = None,
              include_metadata: bool = True) -> Dict[str, Any]:
        """
        Query the system and return formatted results.
        
        Args:
            query_text: User's natural language query
            k: Number of results to return
            filter_dict: Optional metadata filters
            include_metadata: Whether to include metadata in response
            
        Returns:
            Dictionary with query, results, and sources
        """
        logger.info(f"Processing query: '{query_text}'")
        
        # Get raw results from pipeline
        results = self.pipeline.query(query_text, k, filter_dict)
        
        # Format results
        formatted_results = []
        sources = []
        
        for result in results:
            formatted_result = {
                "text": result["text"],
                "relevance": result["score"]
            }
            
            if include_metadata and "metadata" in result:
                formatted_result["metadata"] = result["metadata"]
                
                # Add to sources if not already included
                source = result["metadata"].get("source")
                if source and source not in [s.get("source") for s in sources]:
                    sources.append({
                        "source": source,
                        "title": result["metadata"].get("title", "Unknown"),
                        "chapter": result["metadata"].get("chapter")
                    })
            
            formatted_results.append(formatted_result)
        
        return {
            "query": query_text,
            "results": formatted_results,
            "sources": sources,
            "result_count": len(formatted_results)
        }
    
    def ask(self, question: str, max_results: int = 5) -> str:
        """
        Simplified interface for querying that returns a text response.
        
        Args:
            question: User's question
            max_results: Maximum number of results to include
            
        Returns:
            Formatted text response with results and sources
        """
        response = self.query(question, k=max_results)
        
        # Format as text
        text_response = f"Query: {response['query']}\n\n"
        text_response += f"Found {response['result_count']} relevant passages:\n\n"
        
        for i, result in enumerate(response["results"]):
            text_response += f"[{i+1}] {result['text'][:200]}...\n"
            if "metadata" in result:
                meta = result["metadata"]
                text_response += f"    Source: {meta.get('source', 'Unknown')}"
                if meta.get('page'):
                    text_response += f", Page: {meta.get('page')}"
                text_response += "\n"
            text_response += "\n"
        
        text_response += "Sources:\n"
        for source in response["sources"]:
            text_response += f"- {source.get('source', 'Unknown')}"
            if source.get('title'):
                text_response += f" ({source.get('title')})"
            text_response += "\n"
        
        return text_response