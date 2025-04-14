"""
Milvus connection override to ensure proper Docker networking.

This module must be imported before any pymilvus imports to ensure
that connections are established correctly in a Docker environment.
"""

import os
import sys
from pathlib import Path
import logging

# Set up logging for this override module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("milvus_override")

# Force Milvus host configuration through environment variables
# This must be set before pymilvus is imported anywhere
def setup_milvus_environment():
    """Configure Milvus environment variables for Docker networking."""
    # Set environment variables
    os.environ["MILVUS_HOST"] = "milvus"
    os.environ["MILVUS_PORT"] = "19530"
    
    # Set pymilvus default alias
    try:
        import pymilvus
        # Override default Connection class to ensure it always uses our host/port
        original_connect = pymilvus.connections.connect
        
        def patched_connect(*args, **kwargs):
            # Force host and port for all connections
            kwargs["host"] = "milvus"
            kwargs["port"] = "19530"
            return original_connect(*args, **kwargs)
        
        # Replace the connect function
        pymilvus.connections.connect = patched_connect
        logger.info("Successfully patched pymilvus.connections.connect")
    except ImportError:
        logger.info("pymilvus not yet imported, environment variables should be sufficient")
    
    # Log what we're doing
    logger.info(f"Forcing Milvus connection to milvus:19530")
    logger.info(f"MILVUS_HOST={os.environ.get('MILVUS_HOST')}")
    logger.info(f"MILVUS_PORT={os.environ.get('MILVUS_PORT')}")
    
    # Monkey patch the pymilvus connection if it's already loaded
    try:
        if "pymilvus" in sys.modules:
            logger.warning("pymilvus already imported before override! Some settings may not take effect.")
    except Exception as e:
        logger.error(f"Error checking pymilvus import status: {e}")

# Call setup on import
setup_milvus_environment()