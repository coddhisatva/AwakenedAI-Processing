#!/usr/bin/env python3
import os
from pathlib import Path
import pypdf
import pandas as pd
import argparse

def extract_from_pdf(file_path):
    """Extract text from PDF using PyPDF."""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text
    except Exception as e:
        print(f"Error extracting from PDF {file_path}: {e}")
        return None

def extract_text(file_path):
    """Extract text based on file type."""
    file_path = str(file_path)
    if file_path.lower().endswith('.pdf'):
        return extract_from_pdf(file_path)
    else:
        print(f"Unsupported format for {file_path}")
        return None

def process_books(metadata_file, output_dir, limit=None):
    """Process books based on metadata file."""
    df = pd.read_csv(metadata_file)
    
    if limit:
        df = df.head(int(limit))
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for idx, row in df.iterrows():
        file_path = row['path']
        print(f"Processing {idx+1}/{len(df)}: {file_path}")
        
        text = extract_text(file_path)
        
        if text:
            # Create output filename
            output_file = os.path.join(
                output_dir, 
                f"{Path(file_path).stem}.txt"
            )
            
            # Save extracted text
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            word_count = len(text.split())
            results.append({
                'path': file_path,
                'output_file': output_file,
                'success': True,
                'word_count': word_count
            })
        else:
            results.append({
                'path': file_path,
                'output_file': None,
                'success': False,
                'word_count': 0
            })
    
    # Save processing results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'extraction_results.csv'), index=False)
    
    # Print summary
    success_rate = results_df['success'].mean() * 100
    total_words = results_df['word_count'].sum()
    print(f"\nProcessed {len(df)} files")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total words extracted: {total_words}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract text from books')
    parser.add_argument('--metadata', required=True, help='Path to metadata CSV')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--limit', help='Limit number of books to process')
    
    args = parser.parse_args()
    process_books(args.metadata, args.output, args.limit)