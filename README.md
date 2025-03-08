# AwakenedAI-Processing
Processing repo, one of 2 repos for Awakened AI: A RAG-based knowledge system transforming a curated library of ebooks spanning mysticism, spirituality, history, psychology, alternative health, philosophy, and more, into a comprehensive AI knowledge base using vector embeddings and retrieval-augmented generation.

Site: https://awakened-ai-web.vercel.app/

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
   
2. **Web Application Repository** (Separate Repository)
   - Frontend interface built with Next.js
   - Query API
   - User authentication
   - Document management

3. **Database** (Supabase-hosted PostgreSQL with pgvector)
   - Vector storage
   - Document metadata
   - User data

## System Architecture

### 1. Data Processing Pipeline

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

### 2. RAG System

The RAG system provides the query interface and response generation:

```
User Query → Query Processing → Vector Search → Context Retrieval → LLM Response Generation
```

#### Components:

1. **Query Engine** (`src/interface/query.py`)
   - Processes user queries
   - Interfaces with vector database to retrieve relevant chunks
   - Integrates with LLM service for response generation
   - Formats responses with source attribution

2. **LLM Service** (`src/interface/llm.py`)
   - Connects to OpenAI API with GPT-4 Turbo by default
   - Handles prompt construction with retrieved context
   - Manages API interactions with retry logic
   - Returns structured responses with metadata

3. **Chat CLI** (`tools/chat_cli.py`)
   - Interactive command-line interface for querying the system
   - Rich text formatting for user experience
   - Displays responses with source attribution
   - Interactive session management

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
│   │   └── extractor.py      # Document extraction implementation
│   ├── processing/           # Text processing modules
│   │   └── chunker.py        # Semantic chunking implementation
│   ├── embedding/            # Embedding generation modules
│   │   └── embedder.py       # Embedding generation implementation
│   ├── storage/              # Vector database modules
│   │   └── vector_store.py   # Vector database implementation
│   ├── interface/            # User interface modules
│   │   ├── query.py          # Query engine implementation
│   │   └── llm.py            # LLM service implementation
│   ├── pipeline/             # Pipeline integration
│   │   └── rag_pipeline.py   # RAG pipeline implementation
│   └── retrieval/            # Vector search and retrieval modules
├── database/
│   └── supabase_adapter.py   # Supabase database adapter
├── tools/
│   ├── process_sample.py     # Tool for processing sample documents
│   ├── query_cli.py          # Basic query CLI tool
│   ├── chat_cli.py           # Interactive chat interface
│   ├── check_collection.py   # Tool for checking vector DB contents
│   └── show_sources.py       # Tool for displaying document sources
├── tests/
│   ├── test_extractor.py     # Tests for document extraction
│   ├── test_chunker.py       # Tests for semantic chunking
│   ├── test_embedder.py      # Tests for embedding generation
│   ├── test_vector_store.py  # Tests for vector store integration
│   └── test_llm_integration.py # Tests for LLM integration
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

4. **Vector Database Storage**:
   - Input: Embeddings and metadata from `data/embeddings/`
   - Process: Store in Supabase with pgvector extension
   - Output: Searchable vector database in Supabase

5. **Query Processing**:
   - Input: User query
   - Process: Convert query to embedding, search vector database
   - Output: Relevant document chunks with metadata

6. **Response Generation**:
   - Input: Retrieved document chunks and user query
   - Process: Send context and query to LLM for synthesis
   - Output: Comprehensive answer with source attribution

## Implementation Details

### Document Extractor (`src/extraction/extractor.py`)

The `DocumentExtractor` class handles the extraction of text from various document formats:

- Initialization with raw and processed directories
- Method to extract text from PDF files using PyPDF2
- Error handling for encrypted or problematic PDFs
- Rich metadata extraction (title, author, subject, creator)
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

### Query Engine (`src/interface/query.py`)

The `QueryEngine` class provides the interface for querying the system:

- Initialization with RAG pipeline and LLM service
- Basic retrieval methods returning raw results
- Enhanced methods for LLM-powered responses
- Formatting utilities for text output with attribution

### LLM Service (`src/interface/llm.py`)

The `LLMService` class handles interactions with the language model:

- Initialization with GPT-4 Turbo as the default model
- Response generation from context and query
- Retry logic for API resilience
- Structured response format with source attribution

## Configuration

The project uses environment variables for configuration, stored in the `.env` file:

- `OPENAI_API_KEY`: API key for OpenAI services
- `SUPABASE_URL`: URL for the Supabase project
- `SUPABASE_KEY`: API key for Supabase
- `CHUNK_SIZE`: Target size of each chunk in characters
- `CHUNK_OVERLAP`: Overlap between chunks in characters
- `EMBEDDING_MODEL`: OpenAI embedding model to use
- `LLM_MODEL`: OpenAI model for response generation (default: "gpt-4-turbo")
- `LLM_TEMPERATURE`: Temperature parameter for LLM responses

## Testing

Each component has a corresponding test script:

- `test_extractor.py`: Tests the document extraction functionality
- `test_chunker.py`: Tests the semantic chunking functionality
- `test_embedder.py`: Tests the embedding generation
- `test_vector_store.py`: Tests the Supabase vector store implementation
- `test_llm_integration.py`: Tests the LLM response generation

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
   - Works well with the planned web application
   - Provides authentication and other services needed for the full project

3. **Cost-Effectiveness**
   - Reasonable pricing for the scale of the project
   - Predictable cost structure

## LLM Integration

The LLM integration provides AI-generated responses based on retrieved context:

### Components:

1. **RAGResponse Data Structure**:
   - Contains the AI-generated answer
   - Includes source attribution information
   - Tracks metadata like model used and processing time
   - Provides access to the full context used

2. **Prompt Construction**:
   - Formats retrieved context for optimal LLM understanding
   - Creates system prompts with appropriate instructions
   - Structures user prompts with query and context

3. **Error Handling**:
   - Implements retry logic for API failures
   - Handles rate limiting gracefully
   - Provides informative error messages

4. **Source Attribution**:
   - Extracts metadata from retrieved chunks
   - Associates responses with source documents
   - Formats citations with available metadata (title, author, etc.)

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
- ✅ LLM integration is implemented and tested:
  - Upgraded to GPT-4 Turbo as the default model
  - Connected to OpenAI API for response generation
  - Created interactive chat interface
  - Implemented enhanced source attribution with document titles

## Scaling Considerations

- Implementation of EPUB support for remaining document types
- Parallel processing for improved throughput
- Incremental processing of the full document collection
- Integration with the web application (future repository)
- Monitoring and optimization of vector storage

## Web Application Future Development

The web application will be developed in a separate repository with:

- Next.js frontend with TypeScript and Tailwind CSS
- React components using Shadcn UI
- User authentication via Supabase Auth
- API routes for document querying and management
- Integration with this processing pipeline

## Documents Currently Processed:
1. Cure_Tooth_Decay_Remineralize_Cavities_and_Repair_Your_Teeth_Naturally.pdf
2. How_to_Get_Rich_-_Felix_Dennis.pdf
3. Niacin_The_Real_Story_Learn_about_the_Wonderful_Healing_Properties.pdf
4. The-Book_of_five_rings_-_Kenji_tokitsu-Japanese-strategy.pdf
5. _Dr_Mark_Sircus_Transdermal_Magnesium_Therapy_A_New_Modality_for.pdf