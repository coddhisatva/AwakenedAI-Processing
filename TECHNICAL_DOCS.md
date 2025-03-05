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

1. **Document Extractor** (`src/extraction/extractor.py`)
   - Extracts text from various document formats (PDF, EPUB, etc.)
   - Creates JSON files with extracted text and metadata
   - Currently implemented for PDF files using PyPDF2

2. **Semantic Chunker** (`src/processing/chunker.py`)
   - Splits documents into semantic chunks for better retrieval
   - Uses sentence boundaries to create coherent chunks
   - Maintains context with overlapping chunks
   - Preserves document metadata with each chunk

3. **Document Embedder** (`src/embedding/embedder.py`)
   - Generates vector embeddings for each chunk using OpenAI's embedding models
   - Handles API rate limiting and retries
   - Stores embeddings with chunk metadata

4. **Vector Database** (`src/storage/vector_store.py`)
   - Stores embeddings and metadata for efficient retrieval
   - Supports semantic search capabilities
   - Currently implemented with ChromaDB
   - Abstract interface allows for future backends (e.g., Pinecone)

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
   - Connects to OpenAI API
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
│   ├── embeddings/           # Generated embeddings (JSON)
│   └── vector_db/            # Vector database storage
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
├── tools/
│   ├── process_sample.py     # Tool for processing sample documents
│   ├── query_cli.py          # Basic query CLI tool
│   ├── chat_cli.py           # Interactive chat interface
│   ├── check_collection.py   # Tool for checking vector DB contents
│   └── estimate_processing.py # Tool for estimating processing time
├── tests/
│   ├── test_extractor.py     # Tests for document extraction
│   ├── test_chunker.py       # Tests for semantic chunking
│   ├── test_embedder.py      # Tests for embedding generation
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
   - Process: Store in ChromaDB vector database with metadata
   - Output: Searchable vector database in `data/vector_db/`

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
- Metadata extraction (filename, file size, number of pages)
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
- JSON output with embeddings and metadata

### Vector Store (`src/storage/vector_store.py`)

The `ChromaVectorStore` class handles vector database operations:

- Initialization with persist directory and collection name
- Methods for adding documents with embeddings
- Search functionality with metadata filtering
- Implementation of abstract `VectorStoreBase` interface

### Query Engine (`src/interface/query.py`)

The `QueryEngine` class provides the interface for querying the system:

- Initialization with RAG pipeline and LLM service
- Basic retrieval methods returning raw results
- Enhanced methods for LLM-powered responses
- Formatting utilities for text output with attribution

### LLM Service (`src/interface/llm.py`)

The `LLMService` class handles interactions with the language model:

- Initialization with model parameters
- Response generation from context and query
- Retry logic for API resilience
- Structured response format with source attribution

## Configuration

The project uses environment variables for configuration, stored in the `.env` file:

- `OPENAI_API_KEY`: API key for OpenAI services
- `COLLECTION_NAME`: Name of the vector database collection
- `CHUNK_SIZE`: Target size of each chunk in characters
- `CHUNK_OVERLAP`: Overlap between chunks in characters
- `EMBEDDING_MODEL`: OpenAI embedding model to use
- `LLM_MODEL`: OpenAI model for response generation (e.g., "gpt-3.5-turbo")
- `LLM_TEMPERATURE`: Temperature parameter for LLM responses

## Testing

Each component has a corresponding test script:

- `test_extractor.py`: Tests the document extraction functionality
- `test_chunker.py`: Tests the semantic chunking functionality
- `test_embedder.py`: Tests the embedding generation
- `test_llm_integration.py`: Tests the LLM response generation

## Vector Database Implementation

After evaluating various vector database options, the following decision has been made:

- **Production Target**: Pinecone (cloud-based managed service)
- **Development Environment**: Chroma (local, self-hosted)
- **Migration Strategy**: Abstraction layer to support both backends

### Development Approach:

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

### Vector Database Interface Implementation

The `VectorStoreBase` interface provides:
- Abstract methods for adding documents/embeddings
- Query methods for similarity search
- Metadata filtering capabilities
- Implementation for ChromaDB (Pinecone to be added)

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
   - Formats citations with available metadata

## Current Status

- ✅ Document extraction is implemented and fully tested for PDF files
- ✅ Semantic chunking is implemented and fully tested
- ✅ Embedding generation is implemented and fully tested
- ✅ Vector database implementation with ChromaDB is complete and tested
- ✅ End-to-end pipeline is functional and verified:
  - Successfully processed multiple documents through the entire pipeline
  - Extracted text from PDFs, generated semantic chunks, created embeddings, and stored in ChromaDB
  - Processed 3 documents into 1583 chunks
- ✅ LLM integration is implemented and tested:
  - Connected to OpenAI API for response generation
  - Created interactive chat interface
  - Implemented source attribution

## Scaling Considerations

- Current processing rate suggests ~13 hours to process remaining 30 documents (158.93 MB)
- 7 of 33 documents are EPUBs, which are not yet supported
- Need for parallel processing to improve throughput
- Implementation of progress tracking and resumable processing

## Documents Currently Processed:
1. P. D. Ouspensky - The Fourth Way.pdf - A philosophical work on Gurdjieff's Fourth Way teaching
2. The_Beginning_of_Infinity_-_David_Deutsch.pdf - A book on explanations and knowledge by physicist David Deutsch
3. _Dr_Mark_Sircus_Transdermal_Magnesium_Therapy_A_New_Modality_for.pdf - A book about magnesium therapy