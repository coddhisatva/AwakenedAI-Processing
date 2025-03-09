import logging
import os
from pathlib import Path
import json
from src.extraction.extractor import DocumentExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pdf_extraction():
    """Test PDF document extraction."""
    # Set up directories
    raw_dir = "data/raw"
    processed_dir = "data/processed/test_pdf"
    
    # Create test output directory
    Path(processed_dir).mkdir(exist_ok=True, parents=True)
    
    # Create extractor
    extractor = DocumentExtractor(raw_dir, processed_dir)
    
    # Find a PDF file to test
    pdf_files = list(Path(raw_dir).glob("**/*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in data/raw directory")
        return False
    
    # Process a single PDF file
    file_path = pdf_files[0]
    try:
        extractor._process_pdf(file_path)
        output_path = Path(processed_dir) / f"{file_path.stem}.json"
        if output_path.exists():
            logger.info(f"Successfully processed PDF: {file_path.name}")
            
            # Verify the JSON structure
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert "metadata" in data, "Metadata missing in output"
                assert "content" in data, "Content missing in output"
                assert data["metadata"]["file_type"] == "pdf", "Incorrect file type"
            return True
        else:
            logger.error(f"Failed to generate output file for {file_path.name}")
            return False
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        return False

def test_epub_extraction():
    """Test EPUB document extraction."""
    # Set up directories
    raw_dir = "data/raw"
    processed_dir = "data/processed/test_epub"
    
    # Create test output directory
    Path(processed_dir).mkdir(exist_ok=True, parents=True)
    
    # Create extractor
    extractor = DocumentExtractor(raw_dir, processed_dir)
    
    # Find an EPUB file to test
    epub_files = list(Path(raw_dir).glob("**/*.epub"))
    if not epub_files:
        logger.error("No EPUB files found in data/raw directory")
        return False
    
    # Process a single EPUB file
    file_path = epub_files[0]
    try:
        extractor._process_epub(file_path)
        output_path = Path(processed_dir) / f"{file_path.stem}.json"
        if output_path.exists():
            logger.info(f"Successfully processed EPUB: {file_path.name}")
            
            # Verify the JSON structure
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert "metadata" in data, "Metadata missing in output"
                assert "content" in data, "Content missing in output"
                assert data["metadata"]["file_type"] == "epub", "Incorrect file type"
            return True
        else:
            logger.error(f"Failed to generate output file for {file_path.name}")
            return False
    except Exception as e:
        logger.error(f"Error processing EPUB {file_path}: {str(e)}")
        return False

def main():
    """Test the document extractor for both PDF and EPUB files."""
    # Test PDF extraction
    pdf_result = test_pdf_extraction()
    logger.info(f"PDF extraction test {'passed' if pdf_result else 'failed'}")
    
    # Test EPUB extraction
    epub_result = test_epub_extraction()
    logger.info(f"EPUB extraction test {'passed' if epub_result else 'failed'}")
    
    # Test full extraction process
    raw_dir = "data/raw"
    processed_dir = "data/processed/test_all"
    
    # Create extractor
    extractor = DocumentExtractor(raw_dir, processed_dir)
    
    # Extract all documents
    stats = extractor.extract_all()
    
    # Print results
    logger.info(f"Extraction results: {stats}")

if __name__ == "__main__":
    main() 