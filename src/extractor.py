import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import datetime

import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Simple document extractor for PDF files."""
    
    def __init__(self, raw_dir: str, processed_dir: str):
        """
        Initialize the document extractor.
        
        Args:
            raw_dir: Directory containing raw documents
            processed_dir: Directory to store processed text
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        
        # Track processing statistics
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0
        }
    
    def extract_all(self) -> Dict[str, Any]:
        """
        Process all PDF documents in the raw directory.
        
        Returns:
            Statistics about the extraction process
        """
        logger.info(f"Starting extraction of documents from {self.raw_dir}")
        
        # Find all PDF files
        pdf_files = list(self.raw_dir.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Process each file
        for file_path in pdf_files:
            self.stats["total"] += 1
            try:
                self._process_pdf(file_path)
                self.stats["successful"] += 1
                logger.info(f"Successfully processed {file_path.name}")
            except Exception as e:
                self.stats["failed"] += 1
                logger.error(f"Failed to process {file_path}: {str(e)}")
        
        logger.info(f"Extraction complete. Processed {self.stats['successful']} files successfully, {self.stats['failed']} failed.")
        return self.stats
    
    def _process_pdf(self, file_path: Path) -> None:
        """
        Process a single PDF file.
        
        Args:
            file_path: Path to the PDF file
        """
        # Extract text using PyPDF2
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Check if PDF is encrypted
            if reader.is_encrypted:
                try:
                    reader.decrypt('')  # Try empty password
                except:
                    logger.error(f"Cannot decrypt PDF: {file_path}")
                    raise ValueError(f"Encrypted PDF: {file_path}")
            
            # Extract text from all pages
            num_pages = len(reader.pages)
            text = ""
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            # Create metadata
            metadata = {
                "filename": file_path.name,
                "file_size": os.path.getsize(file_path),
                "num_pages": num_pages,
                "file_type": "pdf"
            }
            
            # Save the extracted content and metadata
            if text:
                output_path = self.processed_dir / f"{file_path.stem}.json"
                self._save_processed_document(output_path, text, metadata)
            else:
                logger.warning(f"No text extracted from {file_path}")
                raise ValueError(f"No text extracted from {file_path}")
    
    def _save_processed_document(self, output_path: Path, content: str, metadata: Dict[str, Any]) -> None:
        """
        Save the processed document content and metadata to a JSON file.
        
        Args:
            output_path: Path to save the processed document
            content: Extracted text content
            metadata: Document metadata
        """
        document = {
            "metadata": metadata,
            "content": content,
            "processing_info": {
                "timestamp": str(datetime.datetime.now()),
                "extractor_version": "1.0"
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved processed document to {output_path}") 