#!/usr/bin/env python
"""
Script to set up Supabase tables and functions for Boga Chat.
"""
import os
import sys
import logging
from pathlib import Path

import supabase
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    sys.exit(1)


def get_supabase_client():
    """Create and return a Supabase client."""
    try:
        client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
        return client
    except Exception as e:
        logger.error(f"Error creating Supabase client: {e}")
        sys.exit(1)


def run_setup_sql():
    """Run the setup SQL script."""
    try:
        # Get the path to the SQL file
        sql_path = Path(__file__).parent / "supabase_setup.sql"
        
        if not sql_path.exists():
            logger.error(f"SQL file not found: {sql_path}")
            sys.exit(1)
        
        # Read the SQL file
        with open(sql_path, "r") as f:
            sql = f.read()
        
        # Split the SQL into individual statements
        statements = sql.split(";")
        
        # Get the Supabase client
        client = get_supabase_client()
        
        # First, check if we can connect to Supabase
        try:
            # Try a simple query to check connection
            response = client.table("document_embeddings").select("id").limit(1).execute()
            logger.info("Successfully connected to Supabase")
        except Exception as e:
            logger.warning(f"Could not query document_embeddings table: {e}")
            logger.info("This is expected if the table doesn't exist yet")
        
        # Execute each statement using the REST API
        for statement in statements:
            statement = statement.strip()
            if statement:
                try:
                    logger.info(f"Executing SQL: {statement[:50]}...")
                    
                    # For CREATE TABLE statements
                    if statement.upper().startswith("CREATE TABLE"):
                        table_name = statement.split("CREATE TABLE IF NOT EXISTS ")[1].split("(")[0].strip()
                        logger.info(f"Creating table: {table_name}")
                        # We can't directly create tables via REST API, but we can check if they exist
                        try:
                            client.table(table_name).select("*").limit(1).execute()
                            logger.info(f"Table {table_name} already exists")
                        except Exception:
                            logger.warning(f"Table {table_name} doesn't exist. Please create it manually in the Supabase dashboard.")
                    
                    # For CREATE INDEX statements
                    elif statement.upper().startswith("CREATE INDEX"):
                        logger.info("Creating index (this needs to be done manually in Supabase dashboard)")
                    
                    # For CREATE EXTENSION statements
                    elif statement.upper().startswith("CREATE EXTENSION"):
                        logger.info("Creating extension (this needs to be done manually in Supabase dashboard)")
                    
                    # For CREATE FUNCTION statements
                    elif statement.upper().startswith("CREATE OR REPLACE FUNCTION"):
                        logger.info("Creating function (this needs to be done manually in Supabase dashboard)")
                    
                    logger.info(f"SQL statement processed")
                    
                except Exception as e:
                    logger.error(f"Error executing SQL: {e}")
                    logger.error(f"Statement: {statement}")
                    # Continue with the next statement
        
        logger.info("Supabase setup completed")
        logger.info("NOTE: Some operations like creating extensions, functions, and indexes need to be done manually in the Supabase dashboard")
        logger.info("Please copy the SQL from supabase_setup.sql and execute it in the Supabase SQL editor")
        
    except Exception as e:
        logger.error(f"Error running setup SQL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logger.info("Setting up Supabase for Boga Chat...")
    run_setup_sql()
    logger.info("Setup complete!") 