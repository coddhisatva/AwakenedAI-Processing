import os
import json
import logging
import tempfile
import subprocess
import io
from pathlib import Path
from typing import Dict, Any, List
import datetime

import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Import pdfminer.six libraries for fallback extraction
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.pdfparser import PDFSyntaxError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Document extractor for PDF and EPUB files."""
    
    def __init__(self, raw_dir: str, processed_dir: str, skip_ocr: bool = False):
        """
        Initialize the document extractor.
        
        Args:
            raw_dir: Directory containing raw documents
            processed_dir: Directory to store processed text
            skip_ocr: Whether to skip OCR processing for PDFs
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        self.skip_ocr = skip_ocr
        
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
                self.extract_pdf(file_path)
                # Successfully processed stat is incremented inside extract_pdf
            except Exception as e:
                self.stats["failed"] += 1
                self.stats["by_type"]["pdf"]["failed"] += 1
                logger.error(f"Failed to process PDF {file_path}: {str(e)}")
        
        # Process each EPUB file
        for file_path in epub_files:
            self.stats["total"] += 1
            try:
                self.extract_epub(file_path)
                # Successfully processed stat is incremented inside extract_epub
            except Exception as e:
                self.stats["failed"] += 1
                self.stats["by_type"]["epub"]["failed"] += 1
                logger.error(f"Failed to process EPUB {file_path}: {str(e)}")
        
        logger.info(f"Extraction complete. Processed {self.stats['successful']} files successfully, {self.stats['failed']} failed, {self.stats['skipped']} skipped.")
        logger.info(f"PDF: {self.stats['by_type']['pdf']['processed']} processed, {self.stats['by_type']['pdf']['failed']} failed, {self.stats['by_type']['pdf']['ocr_applied']} required OCR, {self.stats['by_type']['pdf']['skipped']} skipped")
        logger.info(f"EPUB: {self.stats['by_type']['epub']['processed']} processed, {self.stats['by_type']['epub']['failed']} failed, {self.stats['by_type']['epub']['skipped']} skipped")
        return self.stats
    
    def extract_pdf(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from a PDF file and save to the processed directory.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted text and metadata, or None if extraction fails
        """
        # Check if this file has already been extracted
        output_path = self.processed_dir / f"{file_path.stem}.json"
        if output_path.exists():
            logger.info(f"File already extracted, loading existing content: {file_path.name}")
            self.stats["skipped"] += 1
            self.stats["by_type"]["pdf"]["skipped"] += 1
            # Load and return the already extracted content
            try:
                with open(output_path, 'r') as f:
                    saved_doc = json.load(f)
                    # Convert to the format expected by the pipeline
                    return {
                        "text": saved_doc["content"],
                        "source": file_path.name,
                        "title": saved_doc["metadata"].get("title", file_path.stem),
                        "page": None,
                        "chapter": None
                    }
            except Exception as e:
                logger.error(f"Error loading previously extracted file {output_path}: {e}")
                return None
                
        # Create basic metadata structure
        metadata = {
            "filename": file_path.name,
            "file_size": os.path.getsize(file_path),
            "file_type": "pdf",
            "ocr_applied": False,
            "title": file_path.stem,  # Default title (will be overwritten if metadata is available)
            "author": "Unknown"       # Default author (will be overwritten if metadata is available)
        }
            
        # Step 1: Try regular PDF extraction with PyPDF2
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
                
                for i in range(num_pages):
                    try:
                        page = reader.pages[i]
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n\n"
                        except Exception as page_err:
                            logger.warning(f"Error extracting text from page {i+1} of {file_path.name}: {page_err}")
                    except Exception as page_access_err:
                        logger.warning(f"Error accessing page {i+1} of {file_path.name}: {page_access_err}")
                
                # Update metadata with page count
                metadata["num_pages"] = num_pages
                
                # Extract document information from PDF metadata
                pdf_info = reader.metadata
                
                # Try to add PDF metadata if available, with robust error handling
                if pdf_info:
                    try:
                        # Safely extract title
                        if hasattr(pdf_info, 'title') and pdf_info.title and pdf_info.title not in (None, ''):
                            metadata["title"] = str(pdf_info.title)
                    except Exception as metadata_err:
                        logger.warning(f"Error extracting title metadata from {file_path.name}: {metadata_err}")
                        
                    try:
                        # Safely extract author
                        if hasattr(pdf_info, 'author') and pdf_info.author and pdf_info.author not in (None, ''):
                            metadata["author"] = str(pdf_info.author)
                    except Exception as metadata_err:
                        logger.warning(f"Error extracting author metadata from {file_path.name}: {metadata_err}")
                
                # If text was extracted successfully, save it and return
                if text.strip():
                    self._save_processed_document(output_path, text, metadata)
                    self.stats["successful"] += 1
                    self.stats["by_type"]["pdf"]["processed"] += 1
                    # Return in the format expected by the pipeline
                    return {
                        "text": text,
                        "source": file_path.name,
                        "title": metadata.get("title", file_path.stem),
                        "page": None,
                        "chapter": None
                    }
                
                logger.info(f"No text extracted with PyPDF2 for {file_path.name}, trying pdfminer")
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed for {file_path.name}: {str(e)}")
            # Don't return or raise here, continue to pdfminer
        
        # Step 2: Try pdfminer.six as a fallback
        try:
            # Try to extract with pdfminer.six
            pdfminer_text = self._extract_pdf_with_pdfminer(file_path)
            
            if pdfminer_text.strip():
                # Save the extracted content
                self._save_processed_document(output_path, pdfminer_text, metadata)
                self.stats["successful"] += 1
                self.stats["by_type"]["pdf"]["processed"] += 1
                
                # Return in the format expected by the pipeline
                return {
                    "text": pdfminer_text,
                    "source": file_path.name,
                    "title": metadata.get("title", file_path.stem),
                    "page": None,
                    "chapter": None
                }
            
            logger.warning(f"No text extracted with pdfminer.six for {file_path.name}")
        except Exception as pdfminer_e:
            logger.warning(f"pdfminer.six extraction failed for {file_path.name}: {str(pdfminer_e)}")
            # Don't return or raise here, check if we should try OCR
        
        # Step 3: If both PyPDF2 and pdfminer failed, check if OCR should be attempted
        if self.skip_ocr:
            logger.info(f"Skipping OCR for {file_path.name} as per skip_ocr flag")
            # Not counting as failed, just returning None
            return None
            
        # Step 4: If OCR is allowed, try it as last resort
        logger.info(f"PyPDF2 and pdfminer.six failed to extract text from {file_path.name}, attempting OCR")
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
                num_pages = len(reader.pages)
                
                for i in range(num_pages):
                    try:
                        page = reader.pages[i]
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n\n"
                        except Exception as page_err:
                            logger.warning(f"Error extracting text from OCR'd page {i+1} of {file_path.name}: {page_err}")
                    except Exception as page_access_err:
                        logger.warning(f"Error accessing OCR'd page {i+1} of {file_path.name}: {page_access_err}")
            
            # Update metadata to indicate OCR was applied
            metadata["ocr_applied"] = True
            # Update the num_pages field in case it wasn't set earlier
            metadata["num_pages"] = num_pages
            self.stats["by_type"]["pdf"]["ocr_applied"] += 1
            
            # Delete the temporary file
            os.unlink(temp_output_path)
            
            # Save the extracted content and metadata
            if text.strip():
                self._save_processed_document(output_path, text, metadata)
                self.stats["successful"] += 1
                self.stats["by_type"]["pdf"]["processed"] += 1
                # Return in the format expected by the pipeline
                return {
                    "text": text,
                    "source": file_path.name,
                    "title": metadata.get("title", file_path.stem),
                    "page": None,
                    "chapter": None
                }
            else:
                logger.warning(f"No text extracted from {file_path} even after OCR")
                self.stats["by_type"]["pdf"]["failed"] += 1
                self.stats["failed"] += 1
                raise ValueError(f"No text extracted from {file_path} even after OCR")
            
        except Exception as ocr_e:
            logger.error(f"OCR processing failed for {file_path}: {str(ocr_e)}")
            self.stats["by_type"]["pdf"]["failed"] += 1
            self.stats["failed"] += 1
            raise ValueError(f"Failed to extract text from {file_path}, OCR error: {str(ocr_e)}")
    
    def extract_epub(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from an EPUB file and save to the processed directory.
        
        Args:
            file_path: Path to the EPUB file
            
        Returns:
            Dictionary with extracted text and metadata, or None if extraction fails
        """
        # Check if this file has already been extracted
        output_path = self.processed_dir / f"{file_path.stem}.json"
        if output_path.exists():
            logger.info(f"File already extracted, loading existing content: {file_path.name}")
            self.stats["skipped"] += 1
            self.stats["by_type"]["epub"]["skipped"] += 1
            # Load and return the already extracted content
            try:
                with open(output_path, 'r') as f:
                    saved_doc = json.load(f)
                    # Convert to the format expected by the pipeline
                    return {
                        "text": saved_doc["content"],
                        "source": file_path.name,
                        "title": saved_doc["metadata"].get("title", file_path.stem),
                        "page": None,
                        "chapter": None
                    }
            except Exception as e:
                logger.error(f"Error loading previously extracted file {output_path}: {e}")
                return None
            
        # Extract text from EPUB
        try:
            book = epub.read_epub(file_path)
            
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
                "file_type": "epub",
                "title": file_path.stem,  # Default title (will be overwritten if metadata is available)
                "author": "Unknown"       # Default author (will be overwritten if metadata is available)
            }
            
            # Extract metadata from the EPUB with robust error handling
            try:
                # Safely extract title
                title_data = book.get_metadata('DC', 'title')
                if title_data and len(title_data) > 0 and title_data[0][0]:
                    metadata["title"] = str(title_data[0][0])
            except Exception as metadata_err:
                logger.warning(f"Error extracting title metadata from {file_path.name}: {metadata_err}")
                
            try:
                # Safely extract author
                creator_data = book.get_metadata('DC', 'creator')
                if creator_data and len(creator_data) > 0 and creator_data[0][0]:
                    metadata["author"] = str(creator_data[0][0])
            except Exception as metadata_err:
                logger.warning(f"Error extracting author metadata from {file_path.name}: {metadata_err}")
            
            # Save the extracted content and metadata
            if text:
                self._save_processed_document(output_path, text, metadata)
                self.stats["successful"] += 1
                self.stats["by_type"]["epub"]["processed"] += 1
                # Return in the format expected by the pipeline
                return {
                    "text": text,
                    "source": file_path.name,
                    "title": metadata.get("title", file_path.stem),
                    "page": None,
                    "chapter": None
                }
            else:
                logger.warning(f"No text extracted from {file_path}")
                raise ValueError(f"No text extracted from {file_path}")
        
        except Exception as e:
            # Instead of logging the error here, include detailed context in the exception
            # This avoids duplicate logs but preserves all information
            detailed_message = f"Error processing EPUB {file_path}: {str(e)}"
            raise ValueError(detailed_message) from e
    
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
    
    def _extract_pdf_with_pdfminer(self, file_path: Path) -> str:
        """
        Extract text from a PDF file using pdfminer.six as a fallback when PyPDF2 fails.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        try:
            logger.info(f"Attempting PDF extraction with pdfminer.six for {file_path.name}")
            
            # Use pdfminer to extract text from the PDF
            text = pdfminer_extract_text(str(file_path))
            
            if text.strip():
                logger.info(f"Successfully extracted text with pdfminer.six from {file_path.name}")
                return text
            else:
                logger.warning(f"pdfminer.six extraction successful but no text found in {file_path.name}")
                return ""
        except PDFSyntaxError as e:
            logger.warning(f"pdfminer.six extraction failed due to PDF syntax error in {file_path.name}: {e}")
            return ""
        except Exception as e:
            logger.warning(f"pdfminer.six extraction failed for {file_path.name}: {e}")
            return "" 