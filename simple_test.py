import os
import json
from pathlib import Path

# Create necessary directories
os.makedirs("data/processed", exist_ok=True)

# Test basic file operations
def test_basic_file_operations():
    print("Testing basic file operations...")
    
    # Find PDF files in the raw directory
    raw_dir = Path("data/raw")
    pdf_files = list(raw_dir.glob("**/*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in data/raw directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files: {[f.name for f in pdf_files]}")
    
    # Try to extract text from the first PDF using PyPDF2
    try:
        import PyPDF2
        
        with open(pdf_files[0], 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            # Extract text from first page
            text = reader.pages[0].extract_text()
            
            print(f"Successfully extracted text from {pdf_files[0].name}")
            print(f"Number of pages: {num_pages}")
            print(f"First page preview: {text[:100]}...")
            
            # Save to a simple JSON file
            output_path = Path("data/processed") / f"{pdf_files[0].stem}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "filename": pdf_files[0].name,
                    "num_pages": num_pages,
                    "first_page_text": text[:500]
                }, f, ensure_ascii=False, indent=2)
            
            print(f"Saved processed data to {output_path}")
            
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    test_basic_file_operations() 