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
- Made Dev log
- Properly Fixed and Tested Chunking
- Properly Fixed and Tested Embedding
- Decided Plan for Vector Database
  - Recommended Development Plan
    - Start with Chroma locally for initial development
      - Free, open-source vector DB for development phase
      - No immediate costs while building core functionality
      - Similar API interface to cloud solutions
    - Implement an abstraction layer in code
      - Design vector store interface that works with multiple backends
      - Create adapters for both Chroma and Pinecone
      - Ensure seamless future migration path
    - Test with a sample book collection
      - Use 100-500 books for development testing
      - Validate core RAG functionality
      - Optimize retrieval parameters and performance
    - Migrate to Pinecone when ready for production
      - Use Pinecone trial strategically during final testing
      - Export vectors from development environment
      - Scale up to full collection in production environment