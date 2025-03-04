# Awakened AI - Technical Documentation

## Project Overview
Awakened AI is a Retrieval-Augmented Generation (RAG) based knowledge system designed to process and synthesize information from a large collection of ebooks (10,000-15,000) in various formats (primarily PDFs). The system extracts text from these documents, processes it into semantic chunks, generates embeddings, stores them in a vector database, and provides a query interface for retrieving and synthesizing information.

## System Architecture

### 1. Data Processing Pipeline

The data processing pipeline consists of several stages:

```
Raw Documents → Text Extraction → Semantic Chunking → Embedding Generation → Vector Database Storage
```

#### Components:

1. **Document Extractor** (`src/extractor.py`)
   - Extracts text from various document formats (PDF, EPUB, etc.)
   - Creates JSON files with extracted text and metadata
   - Currently implemented for PDF files using PyPDF2

2. **Semantic Chunker** (`src/chunker.py`)
   - Splits documents into semantic chunks for better retrieval
   - Uses sentence boundaries to create coherent chunks
   - Maintains context with overlapping chunks
   - Preserves document metadata with each chunk

3. **Document Embedder** (`src/embedder.py`)
   - Generates vector embeddings for each chunk using OpenAI's embedding models
   - Handles API rate limiting and retries
   - Stores embeddings with chunk metadata

4. **Vector Database** (to be implemented)
   - Stores embeddings and metadata for efficient retrieval
   - Supports semantic search capabilities
   - Options include Chroma, Pinecone, Weaviate, or FAISS

### 2. RAG System (to be implemented)

The RAG system will provide the query interface and response generation:

```
User Query → Query Processing → Vector Search → Context Retrieval → Response Generation
```

## Directory Structure

```
AwakenedAI/
├── data/
│   ├── raw/                  # Raw document files (PDF, EPUB, etc.)
│   ├── processed/            # Extracted text and metadata (JSON)
│   ├── chunks/               # Semantic chunks (JSON)
│   └── embeddings/           # Generated embeddings (JSON)
├── src/
│   ├── extraction/           # Document extraction modules
│   ├── processing/           # Text processing modules
│   ├── embedding/            # Embedding generation modules
│   ├── retrieval/            # Vector search and retrieval modules
│   ├── extractor.py          # Document extraction implementation
│   ├── chunker.py            # Semantic chunking implementation
│   └── embedder.py           # Embedding generation implementation
├── test_extractor.py         # Test script for document extraction
├── test_chunker.py           # Test script for semantic chunking
├── requirements.txt          # Project dependencies
├── .env                      # Environment variables and configuration
├── DEVLOG.md                 # Development log
└── TECHNICAL_DOCS.md         # Technical documentation
```

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

4. **Vector Database Storage** (to be implemented):
   - Input: Embeddings and metadata from `data/embeddings/`
   - Process: Store in vector database with metadata
   - Output: Searchable vector database

5. **Query Processing** (to be implemented):
   - Input: User query
   - Process: Convert query to embedding, search vector database
   - Output: Relevant document chunks

6. **Response Generation** (to be implemented):
   - Input: Retrieved document chunks
   - Process: Synthesize information, generate response
   - Output: Comprehensive answer with source attribution

## Implementation Details

### Document Extractor (`src/extractor.py`)

The `DocumentExtractor` class handles the extraction of text from various document formats:

- Initialization with raw and processed directories
- Method to extract text from PDF files using PyPDF2
- Error handling for encrypted or problematic PDFs
- Metadata extraction (filename, file size, number of pages)
- JSON output with extracted text and metadata

### Semantic Chunker (`src/chunker.py`)

The `SemanticChunker` class handles the splitting of documents into semantic chunks:

- Initialization with processed and chunks directories
- Configuration for chunk size and overlap
- Sentence-based chunking to maintain semantic coherence
- Metadata preservation with each chunk
- JSON output with chunks and metadata

### Document Embedder (`src/embedder.py`)

The `DocumentEmbedder` class handles the generation of embeddings:

- Initialization with chunks and embeddings directories
- Configuration for embedding model and batch size
- API rate limiting and retry logic
- JSON output with embeddings and metadata

## Configuration

The project uses environment variables for configuration, stored in the `.env` file:

- `OPENAI_API_KEY`: API key for OpenAI services
- `COLLECTION_NAME`: Name of the vector database collection
- `CHUNK_SIZE`: Target size of each chunk in characters
- `CHUNK_OVERLAP`: Overlap between chunks in characters
- `EMBEDDING_MODEL`: OpenAI embedding model to use

## Testing

Each component has a corresponding test script:

- `test_extractor.py`: Tests the document extraction functionality
- `test_chunker.py`: Tests the semantic chunking functionality
- (To be implemented) `test_embedder.py`: Tests the embedding generation
- (To be implemented) `test_retrieval.py`: Tests the vector search and retrieval

## Vector Database Implementation

After evaluating various vector database options, the following decision has been made:

- **Production Target**: Pinecone (cloud-based managed service)
- **Development Environment**: Chroma (local, self-hosted)
- **Migration Strategy**: Abstraction layer to support both backends

#### Development Approach:

1. **Local Development with Chroma**
   - Free, open-source vector database for initial development
   - Similar API to cloud solutions for easier migration
   - Will be used during the development and testing phases

2. **Abstraction Layer**
   - Implementation of a database interface that works with multiple vector DB backends
   - Adapters for both Chroma and Pinecone
   - Ensures seamless migration from development to production

3. **Testing Strategy**
   - Initial testing with a subset of 100-500 books
   - Validation of core RAG functionality
   - Performance testing and parameter optimization

4. **Production Deployment**
   - Migration to Pinecone when ready for full-scale deployment
   - Strategic use of Pinecone trial during final testing phase
   - Export of vectors from development environment to production
   - Scaling to handle the full 10,000-15,000 book collection

#### Vector Database Interface Implementation (to be implemented)

The `VectorStore` interface will provide:
- Abstract methods for adding documents/embeddings
- Query methods for similarity search
- Metadata filtering capabilities
- Implementations for both Chroma and Pinecone

## Current Status

- ✅ Document extraction is implemented and fully tested for PDF files
- ✅ Semantic chunking is implemented and fully tested
- ✅ Embedding generation is implemented and fully tested
- ✅ Vector database implementation with ChromaDB is complete and tested
- ✅ End-to-end pipeline is functional and verified:
  - Successfully processed multiple documents through the entire pipeline
  - Extracted text from PDFs, generated semantic chunks, created embeddings, and stored in ChromaDB
  - Processed 3816 chunks from 5 test documents