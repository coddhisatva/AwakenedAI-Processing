# AwakenedAI-Processing
Processing repo, one of 2 repos for Awakened AI: A RAG-based knowledge system transforming a curated library of ebooks spanning history, psychology, health, philosophy, spirituality, and more, into a comprehensive AI knowledge base using vector embeddings and retrieval-augmented generation.

![AwakenedAI Cover](screenshots/AwakenedAI%20Cover.png)

Live Site: https://awakened-ai-web.vercel.app/

Web Repo: https://github.com/coddhisatva/AwakenedAI-Web

# Loom Video Demo
Click this link to check out a loom video demo of AwakenedAI: https://www.loom.com/share/ff25ceb50d1349119dcdadffd1145893

# Awakened AI - Technical Documentation

## Project Overview
Awakened AI is a Retrieval-Augmented Generation (RAG) based knowledge system designed to process and synthesize information from a large collection of ebooks (10,000-15,000) in various formats (primarily PDFs). The system extracts text from these documents, processes it into semantic chunks, generates embeddings, stores them in a vector database, and provides a query interface for retrieving and synthesizing information.

This repository contains the document processing pipeline, which is one component of the larger Awakened AI project.

## Overall Project Architecture

The Awakened AI project is structured as a multi-repository system:

1. **Processing Repository** (Current Repository)
   - Document extraction
   - Semantic chunking
   - Embedding generation
   - Vector storage integration
   - Basic query tools
   - Performance metrics and analytics
   
2. **Web Application Repository** (Separate Repository)
   - Frontend interface built with Next.js
   - Query API and RAG system implementation
   - User authentication
   - Document management

3. **Database** (Supabase-hosted PostgreSQL with pgvector)
   - Vector storage
   - Document metadata
   - User data

## System Architecture

### Data Processing Pipeline

The data processing pipeline consists of several stages:

```
Raw Documents → Text Extraction → Semantic Chunking → Embedding Generation → Vector Database Storage
```

#### Components:

1. **Document Extractor** (`src/extraction/extractor.py`)
   - Extracts text from various document formats (PDF, EPUB, etc.)
   - Creates JSON files with extracted text and metadata
   - Currently implemented for PDF files using PyPDF2
   - Extracts rich metadata (title, author, etc.) from document properties

2. **Semantic Chunker** (`src/processing/chunker.py`)
   - Splits documents into semantic chunks for better retrieval
   - Uses sentence boundaries to create coherent chunks
   - Maintains context with overlapping chunks
   - Preserves document metadata with each chunk

3. **Document Embedder** (`src/embedding/embedder.py`)
   - Generates vector embeddings for each chunk using OpenAI's embedding models
   - Handles API rate limiting and retries
   - Stores embeddings with chunk metadata
   - Supports single-text embedding for queries

4. **Vector Database** (`src/storage/vector_store.py`)
   - Stores embeddings and metadata for efficient retrieval
   - Supports semantic search capabilities
   - Implemented with Supabase's pgvector extension
   - Abstract interface allows for different backends

## Data Flow

1. **Document Extraction**:
   - Input: Raw documents from `data/raw/`
   - Process: Extract text and metadata using appropriate extractors
   - Output: JSON files in `data/processed/` containing text and metadata

2. **Semantic Chunking**:
   - Input: Processed documents from `data/processed/`
   - Process: Split text into semantic chunks with overlap
   - Output: JSON files in `data/chunks/` containing chunks with metadata

3. **Embedding Generation**:
   - Input: Document chunks from `data/chunks/`
   - Process: Generate embeddings using OpenAI's API
   - Output: JSON files in `data/embeddings/` containing embeddings and metadata

4. **Vector Database Storage**:
   - Input: Embeddings and metadata from `data/embeddings/`
   - Process: Store in Supabase with pgvector extension
   - Output: Searchable vector database in Supabase accessible by the Web Application

## Implementation Details

### Document Extractor (`src/extraction/extractor.py`)

The `DocumentExtractor` class handles the extraction of text from various document formats:

- Initialization with raw and processed directories
- Method to extract text from PDF files using PyPDF2
- Error handling for encrypted or problematic PDFs
- Rich metadata extraction (title, author, etc.)
- Fallback to filename as title when metadata is missing
- JSON output with extracted text and metadata

### Semantic Chunker (`src/processing/chunker.py`)

The `SemanticChunker` class handles the splitting of documents into semantic chunks:

- Initialization with processed and chunks directories
- Configuration for chunk size and overlap
- Sentence-based chunking to maintain semantic coherence
- Metadata preservation with each chunk
- JSON output with chunks and metadata

### Document Embedder (`src/embedding/embedder.py`)

The `DocumentEmbedder` class handles the generation of embeddings:

- Initialization with chunks and embeddings directories
- Configuration for embedding model and batch size
- API rate limiting and retry logic
- Support for both batch embedding and single-text embedding
- JSON output with embeddings and metadata

### Vector Store (`src/storage/vector_store.py`)

The `SupabaseVectorStore` class handles vector database operations:

- Connection to Supabase using the adapter
- Methods for adding documents with embeddings
- Generation of embeddings for documents when not provided
- Search functionality with metadata filtering
- Implementation of abstract `VectorStoreBase` interface

### Supabase Adapter (`database/supabase_adapter.py`)

The `SupabaseAdapter` class handles direct interactions with Supabase:

- Connection to Supabase using provided credentials
- Document and chunk storage in separate tables
- Vector similarity search using pgvector
- Document metadata management

## Configuration

The project uses environment variables for configuration, stored in the `.env` file:

- `OPENAI_API_KEY`: API key for OpenAI services
- `SUPABASE_URL`: URL for the Supabase project
- `SUPABASE_KEY`: API key for Supabase
- `CHUNK_SIZE`: Target size of each chunk in characters
- `CHUNK_OVERLAP`: Overlap between chunks in characters
- `EMBEDDING_MODEL`: OpenAI embedding model to use

## Testing

Each component has a corresponding test script:

- `test_extractor.py`: Tests the document extraction functionality
- `test_chunker.py`: Tests the semantic chunking functionality
- `test_embedder.py`: Tests the embedding generation
- `test_vector_store.py`: Tests the Supabase vector store implementation

## Vector Database Implementation

The project has migrated from ChromaDB to Supabase with pgvector:

### Supabase with pgvector

- PostgreSQL database with pgvector extension
- Managed cloud service for vector storage
- Tables for documents and chunks with metadata
- Vector similarity search using pgvector

### Benefits of Supabase Implementation:

1. **Scalability**
   - Cloud-hosted solution that can scale with the project
   - Handles the planned 10,000-15,000 documents efficiently

2. **Integration**
   - Works well with the web application
   - Provides authentication and other services needed for the full project

3. **Cost-Effectiveness**
   - Reasonable pricing for the scale of the project
   - Predictable cost structure

## Current Status

- ✅ Document extraction is implemented and fully tested for PDF files
- ✅ Semantic chunking is implemented and fully tested
- ✅ Embedding generation is implemented and fully tested
- ✅ Migration from ChromaDB to Supabase is complete
- ✅ Vector database implementation with Supabase is complete and tested
- ✅ End-to-end pipeline is functional and verified:
  - Successfully processed multiple documents through the entire pipeline
  - Extracted text from PDFs, generated semantic chunks, created embeddings, and stored in Supabase
- ✅ Enhanced metadata extraction implemented:
  - Extracts title, author, and other document properties from PDFs
  - Fallback to filename when metadata is missing
- ✅ Unified performance metrics system:
  - Comprehensive tracking across all pipeline phases
  - Performance analytics and visualization tools
  - Historical data persistence and comparison

## Documentation

- [Technical Documentation](TECHNICAL_DOCS.md) - Complete technical documentation of the system
- [Performance Metrics System](METRICS.md) - Detailed guide to the metrics framework
- [Development Log](DEVLOG.md) - Ongoing development notes and progress

## Running the Pipeline

To run the processing pipeline, use the following command:

```bash
source venv/bin/activate && python -m src.pipeline.rag_pipeline --subdir <subdir>
```

Replace `<subdir>` with the specific subdirectory you want to process from the `data/raw/` directory. This is the default and recommended way to run the pipeline. Additional parameters can be added as needed, but specifying the subdirectory is the only required change.
