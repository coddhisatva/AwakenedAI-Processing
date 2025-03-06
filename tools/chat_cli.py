#!/usr/bin/env python3
"""
CLI tool for chatting with the RAG system using LLM responses.
This demonstrates the full RAG pipeline with LLM-generated responses.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extraction.extractor import DocumentExtractor
from src.processing.chunker import SemanticChunker
from src.embedding.embedder import DocumentEmbedder
from src.storage.vector_store import SupabaseVectorStore
from src.pipeline.rag_pipeline import RAGPipeline
from src.interface.query import QueryEngine
from src.interface.llm import LLMService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize rich console for pretty output
console = Console()

def setup_pipeline(model_name=None):
    """Set up the RAG pipeline with all components."""
    # Define default directories
    data_dir = Path(os.path.join(os.path.dirname(__file__), "../data"))
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    chunks_dir = data_dir / "chunks"
    embeddings_dir = data_dir / "embeddings"
    
    # Create directories if they don't exist
    for dir_path in [raw_dir, processed_dir, chunks_dir, embeddings_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize each component with its required arguments
    extractor = DocumentExtractor(raw_dir=raw_dir, processed_dir=processed_dir)
    chunker = SemanticChunker(processed_dir=processed_dir, chunks_dir=chunks_dir)
    embedder = DocumentEmbedder(chunks_dir=chunks_dir, embeddings_dir=embeddings_dir)
    
    # Initialize Supabase vector store
    vector_store = SupabaseVectorStore()
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        extractor=extractor,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store
    )
    
    # Initialize LLM service with specified model or default
    llm_service = LLMService(model_name=model_name or "gpt-4-turbo")
    
    # Initialize query engine
    query_engine = QueryEngine(
        pipeline=pipeline,
        llm_service=llm_service,
        default_result_count=5
    )
    
    return query_engine

def display_response(response, show_sources=True):
    """Display the RAG response in a pretty format using rich."""
    # Create a markdown panel for the answer
    answer_md = Markdown(response.answer)
    console.print(Panel(answer_md, title="Answer", border_style="green"))
    
    # Display sources if requested
    if show_sources and response.sources:
        console.print()
        console.print(Text("Sources:", style="bold"))
        
        for i, source in enumerate(response.sources):
            source_text = f"[{i+1}] "
            
            # Check for title, filename, or any other identifier in order of preference
            if source.get('title'):
                source_text += source.get('title')
            elif source.get('filename'):
                source_text += source.get('filename')
            else:
                source_text += f"Source {i+1}"
                
            if source.get('page'):
                source_text += f", Page {source.get('page')}"
            elif source.get('num_pages'):
                source_text += f" ({source.get('num_pages')} pages)"
                
            console.print(Text(source_text, style="blue"))
    
    # Display processing time
    console.print()
    console.print(Text(f"Processing time: {response.processing_time:.2f}s", style="dim"))
    console.print(Text(f"Model used: {response.model_used}", style="dim"))

def interactive_chat(query_engine):
    """Run an interactive chat session with the RAG system."""
    console.print(Panel.fit(
        "Welcome to the Awakened AI Chat Interface\n"
        "Ask questions about your document collection!\n"
        "Type 'exit' or 'quit' to end the session.",
        title="Awakened AI",
        border_style="blue"
    ))
    
    history = []
    
    while True:
        # Get user input
        console.print()
        question = console.input("[bold blue]You:[/bold blue] ")
        
        # Check if user wants to exit
        if question.lower() in ('exit', 'quit', 'q', 'bye'):
            console.print("[bold green]Goodbye![/bold green]")
            break
        
        # Generate timestamp
        timestamp = time.strftime("%H:%M:%S")
        
        try:
            # Show thinking indicator
            with console.status("[bold green]Thinking...[/bold green]"):
                # Query the system
                response = query_engine.ask(question)
            
            # Display the response
            console.print()
            console.print(f"[bold purple]Awakened AI ({timestamp}):[/bold purple]")
            display_response(response)
            
            # Add to history
            history.append({
                "question": question,
                "answer": response.answer,
                "timestamp": timestamp
            })
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Chat with your document collection using LLM-powered responses")
    parser.add_argument("--model", type=str, help="OpenAI model to use (default: gpt-4-turbo)")
    args = parser.parse_args()
    
    try:
        # Set up the RAG system
        query_engine = setup_pipeline(model_name=args.model)
        
        # Start interactive chat
        interactive_chat(query_engine)
        
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Session terminated by user.[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 