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

## Day 9
- Intended to implement embedding and storage stage batching, failed and refactored to try again next day

## Day 10
- Implemented embedding and storage stage batching
- Overhauled metrics
- last 2 of original 33 docs processed to test stage batching
- 33 docs now in db

## Day 11
- Remove extraneous metadata
 ( Waiting on more files to proceed with processing )

## Day 12
- 446 more files received
- group raw files into directories
  - group0A, first 33 files
  - sorted the 446 into 3 groups, 1A, 1B, and 1C, with 10, 100, 336 files each, to catch problems before expanding scope
- Rag pipeline runs from subdir, not just data/raw
- Process level batching to prevent system memory issues
- Try once on 1A (10 files)
  - 2 are already in db (dupes from 0A)
  - Other 8 all fail, but are still added to manifest
    - To debug next day

## Day 13
- Debug processing:
- Fix processing issue, introduced by subdir change, by switching to absolute paths
- Fix issue where failed-to-process docs still get added to manifest
- Now 5/8 non-dupes processed into db, 3 still fail
  - 38 files in db
- Implement backup pdfminer for cases MyPDF2 can't handle 
  - Fall back happens succesfully
- 2/3 remaining files still completely fail 
  - Turns out they are corrupted
    - (Magic Vol 1)
    - (Tobacco Addiction)
  - Still 38 files in db
- Attempt to fix issue where NullObject in metadata causes doc to fail
- Now the document gets added to the document table, but the actual chunks don't get added to the chunk table
  - This is a new problem in and of itself
  - Claims 39 docs in db, really 38

## Day 14
- Catch up on dev log
- Plan for proceeding:
  - Figure how to solve doc/chunk dislink
    - Can't leave db in bad state
    - Can't leave code where it will in future leave db in bad state
  - Implement md file which logs files that fail (likely corrupted), and group (subdir), pasing them over and marking them for later