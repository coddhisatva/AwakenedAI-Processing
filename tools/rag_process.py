#!/usr/bin/env python3
"""
RAG Processing CLI

This script provides a command-line interface to the RAGPipeline for document processing.
All functionality is consolidated in the RAGPipeline class for a cleaner architecture.

Features:
- Intelligent document filtering (manifest & database checks)
- Text extraction from PDF and EPUB files
- Semantic chunking
- Embedding generation
- Vector database storage
- OCR file handling
- Performance monitoring
"""

import os
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    RAGPipeline.run_cli() 