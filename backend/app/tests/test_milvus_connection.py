#!/usr/bin/env python3
"""
Test script to verify connection to Milvus.
Run this inside the backend container:
docker exec -it basic-rag-new-backend-1 python test_milvus_connection.py
"""

import os
import sys
from pathlib import Path
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("milvus_test")

# Set environment variables
os.environ["MILVUS_HOST"] = "milvus"
os.environ["MILVUS_PORT"] = "19530"

logger.info("Attempting to connect to Milvus using PyMilvus...")

try:
    import pymilvus
    from pymilvus import connections, utility
    
    # Try to disconnect first (in case there's an existing connection)
    try:
        if pymilvus.connections.has_connection("default"):
            pymilvus.connections.disconnect("default")
            logger.info("Disconnected from previous Milvus connection")
    except Exception as e:
        logger.warning(f"Error disconnecting: {e}")
    
    # Connection parameters
    connection_params = {
        "alias": "default",
        "host": "milvus",
        "port": "19530"
    }
    
    logger.info(f"Connecting with parameters: {connection_params}")
    
    # Connect to Milvus
    pymilvus.connections.connect(**connection_params)
    
    # Wait for connection to establish
    time.sleep(1)
    
    # Verify connection
    if pymilvus.connections.has_connection("default"):
        logger.info("Connection established successfully")
        
        # Try to list collections
        try:
            collections = utility.list_collections()
            logger.info(f"Available collections: {collections}")
            logger.info("Milvus connection test passed!")
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
    else:
        logger.error("Failed to establish connection")
        
except ImportError:
    logger.error("Failed to import pymilvus. Make sure it's installed")
except Exception as e:
    logger.error(f"Error connecting to Milvus: {e}")

# Try using LangChain's Milvus wrapper
logger.info("\nAttempting to connect to Milvus using LangChain...")

try:
    from langchain_milvus import Milvus
    from langchain_core.documents import Document
    from langchain_core.embeddings import FakeEmbeddings
    
    # Create a simple document
    docs = [Document(page_content="This is a test document")]
    
    # Use fake embeddings for the test
    embeddings = FakeEmbeddings(size=1536)
    
    # Connection parameters
    connection_args = {
        "host": "milvus",
        "port": "19530"
    }
    
    logger.info(f"Creating Milvus store with parameters: {connection_args}")
    
    # Create a collection
    Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="test_collection",
        connection_args=connection_args
    )
    
    logger.info("Successfully created test collection")
    logger.info("LangChain Milvus connection test passed!")
    
except ImportError:
    logger.error("Failed to import langchain_milvus. Make sure it's installed")
except Exception as e:
    logger.error(f"Error using LangChain Milvus: {e}")