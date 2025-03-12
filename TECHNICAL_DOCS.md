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
   - OCR capability for PDFs without extractable text via OCRmyPDF
   - Extracts rich metadata (title, author, etc.) from document properties
   - Handles different document structures (pages for PDFs, chapters for EPUBs)
   - Skips already processed documents to avoid redundant work

2. **Semantic Chunker** (`src/processing/chunker.py`)
   - Splits documents into semantic chunks for better retrieval
   - Uses sentence boundaries to create coherent chunks
   - Maintains context with overlapping chunks
   - Preserves document metadata with each chunk

3. **Document Embedder** (`src/embedding/embedder.py`)
   - Generates vector embeddings for each chunk using OpenAI's embedding models
   - Implements optimized batch processing with recommended batch size of 500 items
   - Advanced token management to stay within OpenAI limits
   - Rate limiting detection and self-throttling to avoid API rate limits
   - Automatic adjustment of batch sizes based on token limits
   - Sophisticated retry logic with exponential backoff and error handling
   - Detailed performance monitoring and statistics
   - Supports both file-based processing and in-memory processing

4. **Vector Database** (`src/storage/vector_store.py`)
   - Stores embeddings and metadata for efficient retrieval
   - Supports semantic search capabilities
   - Implemented with Supabase's pgvector extension
   - Uses batch insertion with optimal batch size of 200 items
   - Error resilience with retry logic and exponential backoff
   - Performance monitoring for insertion rates
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

## Production Processing Strategy

For processing the full collection of 10,000-15,000 documents, we've designed an optimized approach that efficiently handles both regular documents and those requiring OCR:

### Optimized Two-Phase Processing Approach

1. **Phase 1: Regular Document Processing**
   - Process all documents normally using PyPDF2 for PDFs and ebooklib for EPUBs
   - When a PDF has no extractable text (requires OCR), add it to an OCR manifest file
   - Continue processing other documents without waiting for OCR
   - This phase processes the majority of documents (70-90%) quickly

2. **Phase 2: OCR-Intensive Processing**
   - Process only the documents identified in the OCR manifest file
   - Apply OCR using OCRmyPDF to create searchable text layers
   - Extract text from the OCR'd documents
   - This separated approach allows for scheduling intensive OCR processing at appropriate times

### Efficient Document Processing Implementation

Our production processing implementation uses a multi-level filtering approach to minimize unnecessary work:

1. **First-Level Filtering: Local Manifest**
   - Check files against a local manifest file first (fastest check)
   - Skip files already in manifest to avoid redundant extraction and processing
   - The manifest tracks all files that have been processed or identified in the database

2. **Second-Level Filtering: Database Check**
   - For files not in the manifest, check against the database to detect duplicates
   - Files found in the database but not in the manifest are added to the manifest as "database_duplicates"
   - These files are tracked but not reprocessed, saving significant computational resources

3. **Processing: Only True New Files**
   - Process only files that are not in the manifest AND not in the database
   - This ensures we do minimal processing work while maintaining a complete knowledge base

4. **Final Safety Check: During Database Insertion**
   - The vector store implementation performs a final check during insertion
   - This catches any edge cases where another process may have added the same document
   - Ensures complete database integrity without redundant entries

### Benefits of the Optimized Approach

1. **Efficiency**: Multiple filtering layers minimize unnecessary processing
2. **Complete Tracking**: The manifest provides a comprehensive record of all available documents
3. **Resource Management**: OCR processing can be scheduled separately for resource-intensive files
4. **Progress Tracking**: Clearer visibility of overall progress and remaining work
5. **Incremental Processing**: Support for adding new documents to an existing collection
6. **Database Integrity**: Multiple checks ensure no duplicate entries in the database

### Implementation Tools

1. **RAGPipeline CLI** (`src/pipeline/rag_pipeline.py`)
   - Implements the optimized multi-level filtering approach
   - Processes the entire document collection efficiently
   - Maintains a comprehensive manifest of all documents
   - Handles both regular processing and OCR-specific processing
   - Command-line options for controlling processing behavior:
     * `--force_reprocess`: Process all files regardless of manifest or database status
     * `--update_manifest_only`: Only update the manifest with database files, skip processing
     * `--skip_ocr`: Skip files that need OCR processing
     * `--batch_size`: Control processing batch size for memory management
     * `--extensions`: Specify which file types to process

## Pipeline Implementations and Tools

The project has a consolidated implementation of the document processing pipeline:

### RAGPipeline Class (`src/pipeline/rag_pipeline.py`)

**Purpose:** Core pipeline implementation designed for production-scale processing and querying.

**Features:**
- In-memory processing approach for efficiency
- Handles the entire document processing flow (extraction → chunking → embedding → storage)
- Intelligent document filtering with manifest and database checks
- OCR file handling with skip option
- Performance statistics and timing information
- Provides a query interface for retrieving documents
- Includes command-line interface directly in the class
- Optimized for processing large document collections (thousands of files)

**Usage:**
- Can be run directly as a command-line tool:
  ```bash
  python -m src.pipeline.rag_pipeline --skip_ocr
  ```
- Used by query tools (query_cli.py, chat_cli.py) for retrieving documents
- Handles the full production-scale processing of the entire document collection

**CLI Options:**
- `--input_dir`: Directory containing documents to process (default: data/raw)
- `--batch_size`: Number of documents to process in each batch (default: 5)
- `--extensions`: File extensions to include (default: pdf epub txt)
- `--force_reprocess`: Process all files even if they are in the manifest or database
- `--update_manifest_only`: Only update the manifest with files found in the database, don't process files
- `--skip_ocr`: Skip files that need OCR processing
- `--manifest_path`: Path to the manifest file (default: data/processed_documents.json)

### When to Use RAGPipeline:

1. **For processing the entire document collection:**
   - Run directly: `python -m src.pipeline.rag_pipeline`
   - With OCR skipping: `python -m src.pipeline.rag_pipeline --skip_ocr`

2. **For querying the system:**
   - Use `query_cli.py` or `chat_cli.py` 
   - These tools use the RAGPipeline implementation for retrieval

### Optimized Two-Phase Processing Approach

Our RAGPipeline now supports a two-phase approach for handling both regular documents and those requiring OCR:

1. **Phase 1: Regular Document Processing**
   - Process all regular documents with `--skip_ocr` flag
   - Run: `python -m src.pipeline.rag_pipeline --skip_ocr`
   - This processes all documents except those identified as needing OCR

2. **Phase 2: OCR-Intensive Processing**
   - Process only OCR-requiring files as a separate run
   - The files previously skipped will be processed with OCR
   - This approach allows scheduling intensive OCR processing when resources are available

This consolidated architecture ensures a clean implementation with all functionality in a single class, preventing divergent code paths and potential bugs.

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
├── sql_queries/              # SQL queries for database operations and analysis
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
- OCR capability for PDFs without extractable text:
  - Uses OCRmyPDF to create searchable text layers
  - Applies image processing optimizations like deskewing
  - Extracts text from the OCR'd document
- Rich metadata extraction:
  - PDF metadata: title, author
  - EPUB metadata: title, author
- Error handling for encrypted or problematic documents
- Skip processing for already processed documents:
  - Checks if output file exists before processing
  - Maintains statistics for skipped files 
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
- ✅ OCR capability for PDFs without extractable text
- ✅ Skip processing for already processed documents
- ✅ Semantic chunking is implemented and fully tested
- ✅ Embedding generation is implemented and fully tested
  - ✅ Optimized batch processing with token limit management
  - ✅ Advanced rate limit handling with self-throttling
  - ✅ Performance monitoring with detailed statistics
- ✅ Vector database implementation with Supabase is complete and tested
  - ✅ Efficient batch insertion for improved performance
  - ✅ Retry logic with exponential backoff for resilience
  - ✅ Performance monitoring for insertion metrics
- ✅ Migration from ChromaDB to Supabase is complete
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
- ✅ Two-phase processing for production-scale document collection:
  - Implemented approach for efficiently handling regular and OCR-intensive documents
  - Consolidated all functionality into the RAGPipeline class
  - Added direct CLI functionality to the RAGPipeline
  - Added OCR skipping with `--skip_ocr` flag
- ✅ Performance optimizations for large-scale processing:
  - Implemented batched chunk insertion for Supabase
  - Implemented batched embedding generation
  - Added error resilience and retry mechanisms
  - Enhanced monitoring and metrics for performance tuning
- ✅ Unified metrics framework implemented:
  - Comprehensive performance tracking across all pipeline stages
  - Phase-specific timers for detailed performance analysis
  - Cross-phase analytics for understanding bottlenecks
  - Metrics persistence for historical comparisons
  - Dashboard tools for visualizing performance metrics

## Performance Metrics System

The Awakened AI Processing Pipeline includes a comprehensive performance metrics system designed to track, analyze, and visualize the efficiency of the pipeline across all processing stages. This system provides valuable insights for optimization, troubleshooting, and monitoring production performance.

### 1. Unified Metrics Framework (`src/pipeline/metrics.py`)

The `PipelineMetrics` class forms the core of our metrics system:

- **Centralized Metrics Collection**: Accumulates performance data from all pipeline components
- **Phase-Specific Tracking**: Organizes metrics by pipeline phase (extraction, chunking, embedding, storage)
- **Cross-Phase Analytics**: Analyzes relationships between different pipeline stages
- **Metrics Persistence**: Saves metrics to JSON files for historical analysis
- **Detailed Timings**: Captures fine-grained timing information for each operation

Key components of the metrics framework include:

```python
# Initialize metrics with optional persistence directory
metrics = PipelineMetrics(metrics_dir="data/metrics")

# Update phase-specific metrics
metrics.update("extraction", "files_processed", 10)

# Increment counters
metrics.increment("chunking", "chunks_created", 50)

# Add timing information
metrics.add_timing("PDF extraction", 1.23, phase="extraction")

# Calculate derived metrics (throughput, averages)
metrics.calculate_derived_metrics()

# Log comprehensive summary
metrics.log_summary()

# Save metrics to file
metrics.save()
```

### 2. Phase Timer System

The `PhaseTimer` class enables precise measurement of operation durations:

- **Context Manager Interface**: Simple usage with Python's `with` statement
- **Phase Association**: Links timings to specific pipeline phases
- **Hierarchical Timing**: Supports nested timers for detailed profiling
- **Automatic Metrics Integration**: Updates the unified metrics system

Example usage:

```python
# Time an operation and associate it with a phase
with PhaseTimer("Process document", metrics, phase="extraction"):
    # Processing code here
    extract_document()
```

### 3. Tracked Metrics

The framework tracks a comprehensive set of metrics:

#### Overall Pipeline Metrics:
- Total processing time
- Files processed (total, successful, failed)
- Total chunks created
- Database statistics

#### Extraction Phase Metrics:
- Time spent in extraction
- Files processed by format (PDF, EPUB, etc.)
- OCR statistics
- Throughput (files/second)

#### Chunking Phase Metrics:
- Time spent in chunking
- Documents processed
- Chunks created
- Average chunks per document
- Throughput (chunks/second)

#### Embedding Phase Metrics:
- Time spent generating embeddings
- Chunks processed
- Token statistics
- API calls and errors
- Rate limit statistics
- Batch processing statistics
- Throughput (chunks/second, tokens/second)

#### Storage Phase Metrics:
- Time spent in storage operations
- Chunks stored
- Batch insertion statistics
- Database performance metrics
- Throughput (chunks/second)

#### Cross-Phase Analytics:
- Time distribution by phase
- Performance correlations
- Processing efficiency metrics

### 4. Metrics Dashboard (`tools/metrics_dashboard.py`)

The metrics dashboard provides visualization and analysis of collected metrics:

- **Metrics Loading**: Loads and parses metrics files from the metrics directory
- **Summary Display**: Shows high-level statistics from the most recent pipeline run
- **Historical Analysis**: Compares performance across multiple runs
- **Performance Charts**: Generates visualizations of key metrics:
  - Phase time distribution
  - Throughput by phase
  - Total processing time trends

Example usage:

```bash
# Display summary of recent metrics
python tools/metrics_dashboard.py

# Generate charts and save to output directory
python tools/metrics_dashboard.py --generate_charts --output_dir data/metrics/charts

# Analyze a specific number of recent runs
python tools/metrics_dashboard.py --limit 5
```

### 5. Integration with Pipeline Components

All major pipeline components are integrated with the metrics system:

- **RAGPipeline**: Initializes the metrics system and coordinates metrics collection
- **DocumentExtractor**: Tracks file format statistics and extraction performance
- **SemanticChunker**: Monitors chunking efficiency and document-to-chunk ratios
- **DocumentEmbedder**: Records token usage, API statistics, and embedding throughput
- **SupabaseVectorStore**: Measures storage performance and batch insertion metrics

### 6. Benefits of the Metrics System

1. **Performance Optimization**: Identifies bottlenecks and optimization opportunities
2. **Resource Planning**: Provides data for capacity planning and resource allocation
3. **Cost Monitoring**: Tracks API usage and associated costs for budget management
4. **Quality Assurance**: Validates pipeline efficiency and reliability
5. **Troubleshooting**: Helps diagnose and resolve performance issues
6. **Progress Tracking**: Monitors progress during large-scale document processing

### 7. Using Metrics in Production

For production deployments, consider these best practices:

- **Regular Monitoring**: Review metrics dashboard outputs regularly
- **Baseline Comparison**: Establish performance baselines for detecting changes
- **Alert Thresholds**: Set up alerts for metrics falling outside expected ranges
- **Retention Policy**: Define how long to retain metrics history
- **Aggregation**: For very large processing jobs, consider aggregating metrics

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
13. Frank Rudolph Young - Cyclomancy - The Secret of Psychic Power Control.pdf (OCR processed)
14. frank-rudolph-young-yoga-for-men-only.pdf (OCR processed)