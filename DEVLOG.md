# Development Log

## Day 1 - Initial Setup
**Date:** [Current Date]

### Completed
- Set up project structure
- Broken PDF extraction
- Simple test implementation

### Issues
- Environment setup challenges
- Dependency management issues

## Day 2 - PDF Extraction Refinement
**Date:** [Current Date]

### Completed
- Fixed and Properly Tested PDF extraction pipeline
- Made Dev log, and Technical Doc
- Properly Fixed and Tested semantic Chunking implementation
- Properly Fixed and Tested Embedding generation process
- Implemented and verified Chroma Vector Database with abstraction layer for local development
- **MILESTONE**: Successfully implemented and tested the entire RAG pipeline from document extraction to vector storage. The complete workflow now functions properly:
  - PDF extraction → semantic chunking → embedding generation → vector database storage
  - Tested with multiple documents, processing 3816 chunks across 5 files into the ChromaDB vector store

## Day 3 - LLM Integration
**Date:** [Current Date]

### Completed
- Implemented OpenAI API integration
- Created query interface with LLM response generation
- Built interactive chat CLI
- Tested LLM integration functionality
- Analyzed scaling requirements for full document collection

## Day 4 - Project Restructuring and Database Migration
**Date:** [Current Date]

### Completed
- Designed restructuring of the project
  - Split into multi-repository architecture (Processing, Web Application, Database)
  - Processing repository focuses on document pipeline
  - Web Application repository will use Next.js, React, TypeScript
  - Centralized Supabase database for both repositories
- Switched from ChromaDB to pgVector with Supabase
- Enhanced metadata collection, including title, author, from PDF properties
- Improved source attribution in chat responses with proper document titles
- Upgraded default LLM from GPT-3.5 Turbo to GPT-4 Turbo
- Implemented single-text embedding functionality for improved query handling

## Day 7
- EPUB implementation
- PDFs that need OCR can be extracted and therefore processed
- skip already extracted documents when extracting
- automatic list of files that are in the db created upon adding files to the db (or not needing too bc it would be a dupe) ((validated))
- filenames in db/manifest checked before insertion to prevent dupes in db -- manifest tested
- processing process yields stats about timing and performance, other stats
- ran process, haven't investigated yet; stuff looks weird, but:

## Day 8
- Fixed some stuff; 31/33 documents (non-ocr) now succesfully in the db
  - Need to batch them at embedding and storage so it doesn't take hours