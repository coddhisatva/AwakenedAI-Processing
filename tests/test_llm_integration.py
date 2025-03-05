"""
Test the LLM integration for the RAG system.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.interface.llm import LLMService, RAGResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_llm_service_initialization():
    """Test that the LLM service initializes correctly."""
    try:
        llm_service = LLMService()
        logger.info("✅ LLM service initialized successfully with default parameters")
    except Exception as e:
        logger.error(f"❌ LLM service initialization failed: {str(e)}")
        raise

def test_response_generation():
    """Test that the LLM can generate a response from provided context."""
    try:
        llm_service = LLMService(model_name="gpt-3.5-turbo")
        
        # Create some test context
        test_context = [
            {
                "text": "The Awakened AI project is a RAG-based knowledge system designed to process 10,000-15,000 ebooks.",
                "metadata": {
                    "source": "test_document.txt",
                    "title": "Awakened AI Documentation"
                }
            },
            {
                "text": "The system uses OpenAI embeddings with a vector database to store and retrieve information from documents.",
                "metadata": {
                    "source": "test_document.txt",
                    "title": "Awakened AI Documentation"
                }
            }
        ]
        
        # Test query
        test_query = "What is Awakened AI?"
        
        # Generate response
        response = llm_service.generate_response(test_query, test_context)
        
        # Check response structure
        assert response is not None
        assert isinstance(response, RAGResponse)
        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.sources is not None
        assert len(response.sources) > 0
        assert response.model_used == "gpt-3.5-turbo"
        assert response.processing_time > 0
        
        logger.info(f"✅ Response generation successful: {response.answer[:100]}...")
        logger.info(f"✅ Processing time: {response.processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Response generation failed: {str(e)}")
        raise

def run_tests():
    """Run all test functions."""
    logger.info("Starting LLM integration tests...")
    
    test_llm_service_initialization()
    test_response_generation()
    
    logger.info("All LLM integration tests completed successfully")

if __name__ == "__main__":
    run_tests() 