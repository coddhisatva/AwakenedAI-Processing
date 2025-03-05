# src/interface/query.py
import logging
from typing import List, Dict, Any, Optional, Union
from src.pipeline.rag_pipeline import RAGPipeline
from src.interface.llm import LLMService, RAGResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryEngine:
    """Interface for querying the RAG system."""
    
    def __init__(
        self, 
        pipeline: RAGPipeline,
        llm_service: Optional[LLMService] = None,
        default_result_count: int = 5
    ):
        """
        Initialize the query engine.
        
        Args:
            pipeline: RAG pipeline for retrieval
            llm_service: LLM service for generating responses (if None, only retrieval is used)
            default_result_count: Default number of results to retrieve
        """
        self.pipeline = pipeline
        self.default_result_count = default_result_count
        
        # Initialize LLM service if provided, otherwise create default
        if llm_service:
            self.llm_service = llm_service
        else:
            self.llm_service = LLMService()
            
        logger.info("QueryEngine initialized with LLM integration")
    
    def query(self, 
              query_text: str, 
              k: int = 5, 
              filter_dict: Optional[Dict[str, Any]] = None,
              include_metadata: bool = True) -> Dict[str, Any]:
        """
        Query the system and return formatted retrieval results without LLM synthesis.
        
        Args:
            query_text: User's natural language query
            k: Number of results to return
            filter_dict: Optional metadata filters
            include_metadata: Whether to include metadata in response
            
        Returns:
            Dictionary with query, results, and sources
        """
        logger.info(f"Processing retrieval query: '{query_text}'")
        
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
    
    def ask(
        self, 
        question: str, 
        max_results: int = None,
        system_prompt: Optional[str] = None
    ) -> Union[RAGResponse, str]:
        """
        Enhanced interface that uses LLM to generate a response from retrieved context.
        
        Args:
            question: User's question
            max_results: Maximum number of results to include in context
            system_prompt: Optional custom system prompt for the LLM
            
        Returns:
            RAGResponse containing answer, sources, and metadata
        """
        if max_results is None:
            max_results = self.default_result_count
            
        logger.info(f"Processing RAG question: '{question}'")
        
        # First retrieve relevant context
        retrieval_results = self.query(question, k=max_results)
        context_items = retrieval_results["results"]
        
        # Generate response using LLM
        response = self.llm_service.generate_response(
            query=question,
            context=context_items,
            system_prompt=system_prompt
        )
        
        logger.info(f"Generated response with {len(context_items)} context items")
        return response
    
    def ask_with_text_response(self, question: str, max_results: int = None) -> str:
        """
        Simplified interface that returns a formatted text response.
        
        Args:
            question: User's question
            max_results: Maximum number of results to include
            
        Returns:
            Formatted text response with answer and sources
        """
        response = self.ask(question, max_results=max_results)
        
        # Format as text
        text_response = f"Question: {question}\n\n"
        text_response += f"Answer: {response.answer}\n\n"
        
        text_response += "Sources:\n"
        for i, source in enumerate(response.sources):
            text_response += f"[{i+1}] "
            
            # Add title if available
            if source.get('title'):
                text_response += f"{source.get('title')}"
            else:
                text_response += f"Source {i+1}"
                
            # Add page if available
            if source.get('page'):
                text_response += f", Page {source.get('page')}"
                
            # Add file path if available
            if source.get('source'):
                text_response += f" (File: {source.get('source')})"
                
            text_response += "\n"
        
        return text_response