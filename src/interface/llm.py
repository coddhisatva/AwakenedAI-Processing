"""
LLM Integration for RAG responses.

This module handles the connection to language models for generating
responses based on retrieved context.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Union
import time
from dataclasses import dataclass

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Structured response from the RAG system"""
    answer: str
    sources: List[Dict[str, Any]]
    context_used: List[Dict[str, Any]]
    model_used: str
    processing_time: float

class LLMService:
    """Service for interacting with language models to generate responses"""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM service.
        
        Args:
            model_name: The OpenAI model to use
            temperature: Controls randomness (0-1)
            max_tokens: Maximum tokens in response
            api_key: OpenAI API key (uses env var if None)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Use provided API key or get from environment
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide api_key parameter.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"LLM Service initialized with model: {model_name}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_response(
        self, 
        query: str, 
        context: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None
    ) -> RAGResponse:
        """
        Generate a response using the language model with provided context.
        
        Args:
            query: The user's original question
            context: List of retrieved documents/chunks
            system_prompt: Custom system prompt (uses default if None)
            
        Returns:
            Structured RAGResponse with answer and metadata
        """
        start_time = time.time()
        
        # Format context for prompt
        formatted_context = self._format_context(context)
        
        # Create default system prompt if none provided
        if not system_prompt:
            system_prompt = (
                "You are a knowledgeable assistant that provides accurate, detailed answers based on the provided context. "
                "If the answer cannot be determined from the context, acknowledge this limitation. "
                "Cite specific sources when possible. Be concise but thorough."
            )
        
        # Create messages for chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{formatted_context}"}
        ]
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response content
            answer = response.choices[0].message.content
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create sources list with metadata from context
            sources = self._extract_sources(context)
            
            logger.info(f"Generated response in {processing_time:.2f}s using {self.model_name}")
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                context_used=context,
                model_used=self.model_name,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context items into a string for the prompt."""
        formatted_items = []
        
        for i, item in enumerate(context):
            # Extract text and metadata
            text = item.get("text", "")
            metadata = item.get("metadata", {})
            
            # Format metadata (document title, page, etc.)
            source_info = f"Source {i+1}"
            if "title" in metadata:
                source_info += f": {metadata['title']}"
            if "page" in metadata:
                source_info += f", Page {metadata['page']}"
            
            # Add formatted item
            formatted_items.append(f"{source_info}\n{text}\n")
        
        return "\n".join(formatted_items)
    
    def _extract_sources(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from context items."""
        sources = []
        
        for item in context:
            if "metadata" in item:
                source = {
                    "text_snippet": item.get("text", "")[:100] + "...",
                    **item.get("metadata", {})
                }
                sources.append(source)
        
        return sources 