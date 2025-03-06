#!/usr/bin/env python3
"""
Set up Supabase tables and indexes for the Awakened AI project.
"""

import os
import sys
import logging
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Set up Supabase tables and indexes."""
    try:
        # Get environment variables
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
            return 1
        
        # Read the SQL file
        sql_path = Path(os.path.join(os.path.dirname(__file__), "../database/setup_supabase.sql"))
        with open(sql_path, "r") as f:
            sql_content = f.read()
        
        logger.info("SQL script loaded successfully")
        
        # Print the SQL statements for debugging
        logger.info("SQL to be executed:")
        print(sql_content)
        
        # Visit Supabase SQL Editor to run this script manually
        logger.info(f"Please go to {supabase_url}/project/sql and run the SQL statements above")
        logger.info("After running the SQL, test that the tables were created properly by running:")
        logger.info("python tests/test_vector_store.py")
        
        # Ask for confirmation
        response = input("Would you like to continue and run the tests now? (y/n): ")
        if response.lower() != 'y':
            logger.info("Setup paused. Run the tests after you've run the SQL in Supabase.")
            return 0
            
        # Import and run the tests
        logger.info("Running test_vector_store.py...")
        from tests.test_vector_store import main as test_vector_store
        test_vector_store()

    except Exception as e:
        logger.error(f"Error setting up Supabase: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())