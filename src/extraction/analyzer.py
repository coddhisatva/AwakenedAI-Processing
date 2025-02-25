#!/usr/bin/env python3
import os
import magic
from pathlib import Path
import pandas as pd
import argparse

def analyze_book_collection(directory):
    """Analyze a directory of books and return metadata."""
    results = []
    
    for file_path in Path(directory).glob('**/*'):
        if file_path.is_file():
            try:
                # Get file info
                size_kb = os.path.getsize(file_path) / 1024
                mime = magic.Magic(mime=True)
                file_type = mime.from_file(str(file_path))
                
                results.append({
                    'path': str(file_path),
                    'filename': file_path.name,
                    'size_kb': size_kb,
                    'type': file_type,
                    'extension': file_path.suffix
                })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze a collection of books')
    parser.add_argument('directory', help='Directory containing the books')
    parser.add_argument('--output', default='data/processed/book_metadata.csv', 
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    df = analyze_book_collection(args.directory)
    
    # Print summary
    print(f"Total files: {len(df)}")
    print("\nFile types:")
    print(df['type'].value_counts())
    print("\nExtensions:")
    print(df['extension'].value_counts())
    print("\nSize statistics (KB):")
    print(df['size_kb'].describe())
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nMetadata saved to {args.output}")