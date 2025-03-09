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
   - PDF processing implemented using PyPDF2
   - EPUB processing implemented using ebooklib and BeautifulSoup
   - Extracts rich metadata (title, author, etc.) from document properties
   - Handles different document structures (pages for PDFs, chapters for EPUBs)

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

## Pipeline Implementations and Tools

The project has two distinct implementations of the document processing pipeline, each serving different purposes:

### 1. RAGPipeline Class (`src/pipeline/rag_pipeline.py`)

**Purpose:** Core pipeline implementation designed for production-scale processing and querying.

**Features:**
- In-memory processing approach for efficiency
- Handles the entire document processing flow (extraction → chunking → embedding → storage)
- Provides a query interface for retrieving documents
- Optimized for processing large document collections (thousands of files)

**Usage:**
- Used by query tools (query_cli.py, chat_cli.py) for retrieving documents
- Should be used for full-scale processing of the entire document collection
- Not directly exposed through a CLI tool for processing

### 2. Process Sample Tool (`tools/process_sample.py`)

**Purpose:** Testing and development tool for processing small batches of documents.

**Features:**
- File-based approach with intermediate JSON files for easy inspection
- Command-line options for controlling processing (file count, specific files, extensions)
- Support for random sampling of documents
- Detailed logging and statistics
- Designed for smaller-scale testing and validation

**Usage:**
- Use during development and testing
- Process small document samples to verify pipeline functionality
- Test new document types or processing features
- NOT intended for processing the complete document collection

### When to Use Each Implementation:

1. **For testing with small batches (5-10 documents):**
   - Use `process_sample.py` with appropriate options
   - Example: `python tools/process_sample.py --count 5 --extensions pdf epub`

2. **For querying the system:**
   - Use `query_cli.py` or `chat_cli.py`
   - These tools use the RAGPipeline implementation for retrieval

3. **For full-scale processing (thousands of documents):**
   - Currently, there is no dedicated CLI tool for this
   - A new tool based on RAGPipeline.process_directory should be created for this purpose

### Future Development:

**Planned Tools:**
- Create `process_collection.py` that uses RAGPipeline.process_directory for full-scale processing
- This tool will leverage the more efficient in-memory processing approach of RAGPipeline for handling the entire document collection

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
│   │   └── rag_pipeline.py   # RAG pipeline implementation (core pipeline for production)
│   └── retrieval/            # Vector search and retrieval modules
├── database/
│   └── supabase_adapter.py   # Supabase database adapter
├── tools/
│   ├── process_sample.py     # Tool for processing sample documents (testing/development)
│   ├── query_cli.py          # Basic query CLI tool (uses RAGPipeline)
│   ├── chat_cli.py           # Interactive chat interface (uses RAGPipeline)
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
- Support for multiple document formats:
  - PDF extraction using PyPDF2
  - EPUB extraction using ebooklib and BeautifulSoup
- Format-specific processing techniques:
  - Page-based extraction for PDFs
  - Chapter-based extraction for EPUBs
- Rich metadata extraction:
  - PDF metadata: title, author, subject, creator
  - EPUB metadata: title, author, description, publisher, date, language
- Error handling for encrypted or problematic documents
- Fallback to filename as title when metadata is missing
- JSON output with extracted text and metadata
- Detailed processing statistics for each file format

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

### RAGPipeline (`src/pipeline/rag_pipeline.py`)

The `RAGPipeline` class provides an integrated pipeline implementation:

- Orchestrates all components (extraction, chunking, embedding, storage)
- In-memory processing approach for efficiency
- Methods:
  - `process_directory`: Process all documents in a directory
  - `query`: Search for relevant documents
- Used by query tools but not directly exposed for document processing
- Designed for production-scale processing of the full document collection

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

## Tools Usage Guide

### 1. Process Sample Documents (Testing/Development)

**Tool:** `tools/process_sample.py`

**Purpose:** Process a small batch of documents for testing and development.

**Key Options:**
- `--count`: Number of documents to randomly select (default: 5)
- `--extensions`: File extensions to include (default: pdf epub txt)
- `--specific_files`: Process specific files instead of random selection

**Example Usage:**
```bash
# Process 10 random documents
python tools/process_sample.py --count 10

# Process specific file types
python tools/process_sample.py --extensions pdf epub

# Process specific files
python tools/process_sample.py --specific_files data/raw/document1.pdf data/raw/document2.epub
```

**Notes:**
- Creates intermediate JSON files in processed/, chunks/, and embeddings/ directories
- Detailed logging for debugging
- NOT recommended for full-scale processing of thousands of documents

### 2. Query the System (Basic Interface)

**Tool:** `tools/query_cli.py`

**Purpose:** Simple command-line interface for querying the system.

**Example Usage:**
```bash
python tools/query_cli.py "What is the significance of magnesium in health?"
```

**Notes:**
- Uses RAGPipeline for document retrieval
- Returns raw chunks without LLM processing by default
- Add --ai flag to get LLM-generated responses

### 3. Interactive Chat (Advanced Interface)

**Tool:** `tools/chat_cli.py`

**Purpose:** Rich interactive chat interface with history and formatting.

**Example Usage:**
```bash
python tools/chat_cli.py
```

**Notes:**
- Uses RAGPipeline for document retrieval
- Maintains conversation history
- Includes source attribution
- Uses GPT-4 Turbo by default for responses

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

- `test_extractor.py`: Tests the document extraction functionality for PDF and EPUB
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
- ✅ Document extraction for EPUB files is implemented and fully tested
- ✅ Semantic chunking is implemented and fully tested
- ✅ Embedding generation is implemented and fully tested
- ✅ Migration from ChromaDB to Supabase is complete
- ✅ Vector database implementation with Supabase is complete and tested
- ✅ End-to-end pipeline is functional and verified:
  - Successfully processed multiple documents through the entire pipeline
  - Extracted text from PDFs and EPUBs, generated semantic chunks, created embeddings, and stored in Supabase
- ✅ Enhanced metadata extraction implemented:
  - Extracts title, author, and other document properties from PDFs and EPUBs
  - Fallback to filename when metadata is missing
- ✅ LLM integration is implemented and tested:
  - Upgraded to GPT-4 Turbo as the default model
  - Connected to OpenAI API for response generation
  - Created interactive chat interface
  - Implemented enhanced source attribution with document titles

## Scaling Considerations

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
6. Erikson, Thomas - Surrounded by Idiots (2019).epub
7. Hypnotically Annihilating Anxiety - Penetr - The Rogue Hypnotist.epub
8. WILBER, Ken - The Religion of Tomorrow.epub
9. the-symbolism-of-freemasons-9781594629136.epub
10. Paramahansa_Yogananda_How_You_Can_Talk_With_God_Yogoda_Satsanga_.epub
11. The Asshole Survival Guide--How to Deal with People Who Treat You Like Dirt - Robert I. Sutton.epub
12. Uncover_Your_Authentic_Self_Through_Shadow_Work_by_Kristina_Rosen.epub