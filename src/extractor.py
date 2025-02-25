import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pypdf2
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.epub import partition_epub
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Handles extraction of text from various document formats."""
    
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
            "by_format": {}
        }
    
    def extract_all(self) -> Dict[str, Any]:
        """
        Process all documents in the raw directory.
        
        Returns:
            Statistics about the extraction process
        """
        logger.info(f"Starting extraction of documents from {self.raw_dir}")
        
        for file_path in self.raw_dir.glob("**/*"):
            if file_path.is_file():
                self.stats["total"] += 1
                try:
                    self._process_file(file_path)
                    self.stats["successful"] += 1
                except Exception as e:
                    self.stats["failed"] += 1
                    logger.error(f"Failed to process {file_path}: {str(e)}")
        
        logger.info(f"Extraction complete. Processed {self.stats['successful']} files successfully, {self.stats['failed']} failed.")
        return self.stats
    
    def _process_file(self, file_path: Path) -> None:
        """
        Process a single file based on its extension.
        
        Args:
            file_path: Path to the file to process
        """
        file_format = file_path.suffix.lower()
        
        # Update format statistics
        if file_format not in self.stats["by_format"]:
            self.stats["by_format"][file_format] = {"total": 0, "successful": 0, "failed": 0}
        self.stats["by_format"][file_format]["total"] += 1
        
        try:
            # Choose extraction method based on file format
            if file_format == '.pdf':
                content, metadata = self._extract_from_pdf(file_path)
            elif file_format == '.epub':
                content, metadata = self._extract_from_epub(file_path)
            elif file_format in ['.txt', '.md']:
                content, metadata = self._extract_from_text(file_path)
            elif file_format in ['.docx', '.doc']:
                content, metadata = self._extract_from_doc(file_path)
            else:
                logger.warning(f"Unsupported format: {file_format} for {file_path}")
                self.stats["by_format"][file_format]["failed"] += 1
                return
            
            # Save the extracted content and metadata
            if content:
                output_path = self.processed_dir / f"{file_path.stem}.json"
                self._save_processed_document(output_path, content, metadata)
                self.stats["by_format"][file_format]["successful"] += 1
            else:
                logger.warning(f"No content extracted from {file_path}")
                self.stats["by_format"][file_format]["failed"] += 1
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            self.stats["by_format"][file_format]["failed"] += 1
            raise
    
    def _extract_from_pdf(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from PDF files.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (content, metadata)
        """
        logger.info(f"Extracting from PDF: {file_path}")
        
        # Try with unstructured first
        try:
            elements = partition_pdf(
                filename=str(file_path),
                extract_images=False,
                infer_table_structure=False
            )
            content = "\n\n".join([str(element) for element in elements])
            
            # If content is too short, try PyPDF2 as fallback
            if len(content) < 100:
                logger.info(f"Unstructured extracted limited content, trying PyPDF2 for {file_path}")
                content = self._extract_with_pypdf2(file_path)
            
            # Extract metadata using PyPDF2
            metadata = self._extract_pdf_metadata(file_path)
            return content, metadata
            
        except Exception as e:
            logger.warning(f"Unstructured PDF extraction failed for {file_path}: {str(e)}. Trying PyPDF2.")
            content = self._extract_with_pypdf2(file_path)
            metadata = self._extract_pdf_metadata(file_path)
            return content, metadata
    
    def _extract_with_pypdf2(self, file_path: Path) -> str:
        """Extract text from PDF using PyPDF2 as a fallback method."""
        try:
            with open(file_path, 'rb') as file:
                reader = pypdf2.PdfReader(file)
                
                # Check if PDF is encrypted
                if reader.is_encrypted:
                    try:
                        reader.decrypt('')  # Try empty password
                    except:
                        logger.error(f"Cannot decrypt PDF: {file_path}")
                        return ""
                
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
                return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {file_path}: {str(e)}")
            return ""
    
    def _extract_pdf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = pypdf2.PdfReader(file)
                if reader.metadata:
                    metadata = {
                        "title": reader.metadata.get('/Title', ''),
                        "author": reader.metadata.get('/Author', ''),
                        "subject": reader.metadata.get('/Subject', ''),
                        "creator": reader.metadata.get('/Creator', ''),
                        "producer": reader.metadata.get('/Producer', ''),
                        "page_count": len(reader.pages)
                    }
                else:
                    metadata = {"page_count": len(reader.pages)}
                
                # Add file metadata
                metadata.update({
                    "filename": file_path.name,
                    "file_size": os.path.getsize(file_path),
                    "file_type": "pdf"
                })
                
                return metadata
        except Exception as e:
            logger.error(f"Failed to extract PDF metadata from {file_path}: {str(e)}")
            return {
                "filename": file_path.name,
                "file_type": "pdf",
                "extraction_error": str(e)
            }
    
    def _extract_from_epub(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from EPUB files.
        
        Args:
            file_path: Path to the EPUB file
            
        Returns:
            Tuple of (content, metadata)
        """
        logger.info(f"Extracting from EPUB: {file_path}")
        
        # Try with unstructured first
        try:
            elements = partition_epub(filename=str(file_path))
            content = "\n\n".join([str(element) for element in elements])
            
            # If content is too short, try ebooklib as fallback
            if len(content) < 100:
                logger.info(f"Unstructured extracted limited content, trying ebooklib for {file_path}")
                content, metadata = self._extract_with_ebooklib(file_path)
                return content, metadata
            
            # Extract metadata using ebooklib
            _, metadata = self._extract_with_ebooklib(file_path)
            return content, metadata
            
        except Exception as e:
            logger.warning(f"Unstructured EPUB extraction failed for {file_path}: {str(e)}. Trying ebooklib.")
            return self._extract_with_ebooklib(file_path)
    
    def _extract_with_ebooklib(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from EPUB using ebooklib."""
        try:
            book = epub.read_epub(str(file_path))
            
            # Extract metadata
            metadata = {
                "title": book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else '',
                "creator": book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else '',
                "language": book.get_metadata('DC', 'language')[0][0] if book.get_metadata('DC', 'language') else '',
                "identifier": book.get_metadata('DC', 'identifier')[0][0] if book.get_metadata('DC', 'identifier') else '',
                "filename": file_path.name,
                "file_size": os.path.getsize(file_path),
                "file_type": "epub"
            }
            
            # Extract content
            content = ""
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text()
                    content += text + "\n\n"
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Ebooklib extraction failed for {file_path}: {str(e)}")
            return "", {
                "filename": file_path.name,
                "file_type": "epub",
                "extraction_error": str(e)
            }
    
    def _extract_from_text(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Extract content from plain text files.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Tuple of (content, metadata)
        """
        logger.info(f"Extracting from text file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read()
            
            metadata = {
                "filename": file_path.name,
                "file_size": os.path.getsize(file_path),
                "file_type": file_path.suffix[1:],  # Remove the dot
                "line_count": len(content.splitlines())
            }
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {str(e)}")
            return "", {
                "filename": file_path.name,
                "file_type": file_path.suffix[1:],
                "extraction_error": str(e)
            }
    
    def _extract_from_doc(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Extract content from Word documents.
        
        Args:
            file_path: Path to the Word document
            
        Returns:
            Tuple of (content, metadata)
        """
        logger.info(f"Extracting from Word document: {file_path}")
        
        try:
            # For now, we'll use unstructured if available
            # This is a placeholder - you may need to install additional dependencies
            # like python-docx or textract for better DOC/DOCX support
            logger.warning(f"Word document extraction not fully implemented for {file_path}")
            
            metadata = {
                "filename": file_path.name,
                "file_size": os.path.getsize(file_path),
                "file_type": file_path.suffix[1:]
            }
            
            # Return empty content for now
            return "", metadata
            
        except Exception as e:
            logger.error(f"Word document extraction failed for {file_path}: {str(e)}")
            return "", {
                "filename": file_path.name,
                "file_type": file_path.suffix[1:],
                "extraction_error": str(e)
            }
    
    def _save_processed_document(self, output_path: Path, content: str, metadata: Dict[str, Any]) -> None:
        """
        Save the processed document content and metadata to a JSON file.
        
        Args:
            output_path: Path to save the processed document
            content: Extracted text content
            metadata: Document metadata
        """
        import json
        
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


# Add this import at the top of the file
import datetime 