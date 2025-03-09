import os
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import datetime

import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Document extractor for PDF and EPUB files."""
    
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
            "failed": 0,
            "skipped": 0,
            "by_type": {
                "pdf": {"processed": 0, "failed": 0, "ocr_applied": 0, "skipped": 0},
                "epub": {"processed": 0, "failed": 0, "skipped": 0}
            }
        }
    
    def extract_all(self) -> Dict[str, Any]:
        """
        Process all PDF and EPUB documents in the raw directory.
        
        Returns:
            Statistics about the extraction process
        """
        logger.info(f"Starting extraction of documents from {self.raw_dir}")
        logger.info("Will skip already processed files")
        
        # Find all PDF files
        pdf_files = list(self.raw_dir.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Find all EPUB files
        epub_files = list(self.raw_dir.glob("**/*.epub"))
        logger.info(f"Found {len(epub_files)} EPUB files")
        
        # Process each PDF file
        for file_path in pdf_files:
            self.stats["total"] += 1
            try:
                self._process_pdf(file_path)
                # Successfully processed stat is incremented inside _process_pdf
            except Exception as e:
                self.stats["failed"] += 1
                self.stats["by_type"]["pdf"]["failed"] += 1
                logger.error(f"Failed to process PDF {file_path}: {str(e)}")
        
        # Process each EPUB file
        for file_path in epub_files:
            self.stats["total"] += 1
            try:
                self._process_epub(file_path)
                # Successfully processed stat is incremented inside _process_epub
            except Exception as e:
                self.stats["failed"] += 1
                self.stats["by_type"]["epub"]["failed"] += 1
                logger.error(f"Failed to process EPUB {file_path}: {str(e)}")
        
        logger.info(f"Extraction complete. Processed {self.stats['successful']} files successfully, {self.stats['failed']} failed, {self.stats['skipped']} skipped.")
        logger.info(f"PDF: {self.stats['by_type']['pdf']['processed']} processed, {self.stats['by_type']['pdf']['failed']} failed, {self.stats['by_type']['pdf']['ocr_applied']} required OCR, {self.stats['by_type']['pdf']['skipped']} skipped")
        logger.info(f"EPUB: {self.stats['by_type']['epub']['processed']} processed, {self.stats['by_type']['epub']['failed']} failed, {self.stats['by_type']['epub']['skipped']} skipped")
        return self.stats
    
    def _process_pdf(self, file_path: Path) -> None:
        """
        Process a single PDF file, using OCR if regular extraction fails.
        
        Args:
            file_path: Path to the PDF file
        """
        # Check if this file has already been processed
        output_path = self.processed_dir / f"{file_path.stem}.json"
        if output_path.exists():
            logger.info(f"Skipping already processed file: {file_path.name}")
            self.stats["skipped"] += 1
            self.stats["by_type"]["pdf"]["skipped"] += 1
            return
        
        # First try regular PDF extraction
        try:
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
                
                # Extract document information from PDF metadata
                pdf_info = reader.metadata
                
                # Create metadata with enhanced document information
                metadata = {
                    "filename": file_path.name,
                    "file_size": os.path.getsize(file_path),
                    "num_pages": num_pages,
                    "file_type": "pdf",
                    "ocr_applied": False
                }
                
                # Add PDF metadata if available
                if pdf_info:
                    if pdf_info.title:
                        metadata["title"] = pdf_info.title
                    if pdf_info.author:
                        metadata["author"] = pdf_info.author
                    if pdf_info.subject:
                        metadata["subject"] = pdf_info.subject
                    if pdf_info.creator:
                        metadata["creator"] = pdf_info.creator
                
                # If no title in metadata, use filename without extension as title
                if "title" not in metadata or not metadata["title"]:
                    metadata["title"] = file_path.stem
                
                # If text was extracted, save it
                if text.strip():
                    self._save_processed_document(output_path, text, metadata)
                    self.stats["successful"] += 1
                    self.stats["by_type"]["pdf"]["processed"] += 1
                    return
                
                logger.info(f"No text extracted with PyPDF2 for {file_path.name}, attempting OCR")
            
            # If no text was extracted, try OCR
            try:
                # Create a temporary file for OCR output
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_output_path = temp_file.name
                
                # Use OCRmyPDF to add text layer to the PDF
                try:
                    import ocrmypdf
                    ocrmypdf.ocr(
                        input_file=str(file_path),
                        output_file=temp_output_path,
                        skip_text=True,  # Skip pages that already have text
                        deskew=True,     # Straighten pages
                        force_ocr=False  # Only apply OCR where needed
                    )
                except ImportError:
                    # If ocrmypdf is not available as a module, try command line
                    cmd = ['ocrmypdf', '--skip-text', '--deskew', str(file_path), temp_output_path]
                    subprocess.run(cmd, check=True)
                
                # Extract text from the OCR'd PDF
                with open(temp_output_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                
                # Update metadata to indicate OCR was applied
                metadata["ocr_applied"] = True
                self.stats["by_type"]["pdf"]["ocr_applied"] += 1
                
                # Delete the temporary file
                os.unlink(temp_output_path)
                
                # Save the extracted content and metadata
                if text.strip():
                    self._save_processed_document(output_path, text, metadata)
                    self.stats["successful"] += 1
                    self.stats["by_type"]["pdf"]["processed"] += 1
                    logger.info(f"Successfully extracted text with OCR for {file_path.name}")
                    return
                else:
                    logger.warning(f"No text extracted from {file_path} even after OCR")
                    raise ValueError(f"No text extracted from {file_path} even after OCR")
                
            except Exception as ocr_e:
                logger.error(f"OCR processing failed for {file_path}: {str(ocr_e)}")
                raise ValueError(f"Failed to extract text from {file_path}, OCR error: {str(ocr_e)}")
        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
    
    def _process_epub(self, file_path: Path) -> None:
        """
        Process a single EPUB file.
        
        Args:
            file_path: Path to the EPUB file
        """
        # Check if this file has already been processed
        output_path = self.processed_dir / f"{file_path.stem}.json"
        if output_path.exists():
            logger.info(f"Skipping already processed file: {file_path.name}")
            self.stats["skipped"] += 1
            self.stats["by_type"]["epub"]["skipped"] += 1
            return
            
        # Load the EPUB file
        book = epub.read_epub(str(file_path))
        
        # Extract text from all HTML items
        text = ""
        chapter_count = 0
        
        # Helper function to extract text from HTML content
        def chapter_to_text(content):
            soup = BeautifulSoup(content, 'html.parser')
            return soup.get_text(' ', strip=True)
        
        # Process each document item
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                chapter_text = chapter_to_text(item.get_content())
                if chapter_text:
                    text += chapter_text + "\n\n"
                    chapter_count += 1
        
        # Create metadata with available EPUB information
        metadata = {
            "filename": file_path.name,
            "file_size": os.path.getsize(file_path),
            "num_chapters": chapter_count,
            "file_type": "epub"
        }
        
        # Extract metadata from the EPUB
        if book.get_metadata('DC', 'title'):
            metadata["title"] = book.get_metadata('DC', 'title')[0][0]
        if book.get_metadata('DC', 'creator'):
            metadata["author"] = book.get_metadata('DC', 'creator')[0][0]
        if book.get_metadata('DC', 'description'):
            metadata["description"] = book.get_metadata('DC', 'description')[0][0]
        if book.get_metadata('DC', 'publisher'):
            metadata["publisher"] = book.get_metadata('DC', 'publisher')[0][0]
        if book.get_metadata('DC', 'date'):
            metadata["date"] = book.get_metadata('DC', 'date')[0][0]
        if book.get_metadata('DC', 'language'):
            metadata["language"] = book.get_metadata('DC', 'language')[0][0]
        
        # If no title in metadata, use filename without extension as title
        if "title" not in metadata or not metadata["title"]:
            metadata["title"] = file_path.stem
        
        # Save the extracted content and metadata
        if text:
            self._save_processed_document(output_path, text, metadata)
            self.stats["successful"] += 1
            self.stats["by_type"]["epub"]["processed"] += 1
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