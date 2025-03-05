import logging
from extraction.extractor import DocumentExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the document extractor."""
    # Set up directories
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    # Create extractor
    extractor = DocumentExtractor(raw_dir, processed_dir)
    
    # Extract all documents
    stats = extractor.extract_all()
    
    # Print results
    logger.info(f"Extraction results: {stats}")

if __name__ == "__main__":
    main() 