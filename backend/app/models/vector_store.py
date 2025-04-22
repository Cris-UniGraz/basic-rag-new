# IMPORTANT: Set environment variables BEFORE importing any modules
import os
import time

# Force Milvus host configuration through environment variables
# These must be set before importing pymilvus
os.environ["MILVUS_HOST"] = os.environ.get("MILVUS_HOST", "milvus")
os.environ["MILVUS_PORT"] = os.environ.get("MILVUS_PORT", "19530")

from typing import List, Dict, Any, Optional, Union
from loguru import logger
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Import pymilvus after environment variables are set
import pymilvus
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from langchain_milvus import Milvus as BaseMilvus

# Custom Milvus wrapper to ensure correct connection parameters
class Milvus(BaseMilvus):
    """
    Custom Milvus wrapper that ensures connection parameters are correctly set.
    """
    
    @classmethod
    def from_documents(
        cls,
        documents,
        embedding,
        collection_name="langchain",
        connection_args=None,
        **kwargs,
    ):
        """Override to ensure connection parameters are properly set."""
        # CRITICAL: Force set environment variables again
        os.environ["MILVUS_HOST"] = "milvus"
        os.environ["MILVUS_PORT"] = "19530"
        
        # Force connection args to use service name in Docker
        connection_args = {
            "host": "milvus", 
            "port": "19530",
            "uri": "http://milvus:19530"
        }
        logger.info(f"Overriding connection args to: {connection_args}")
        
        # Ensure there's a connection to Milvus before proceeding
        if not pymilvus.connections.has_connection("default"):
            logger.info("No existing connection. Connecting to Milvus before creating collection.")
            try:
                # Connect to Milvus with explicit parameters
                pymilvus.connections.connect(**connection_args)
                time.sleep(1)  # Wait for connection to establish
            except Exception as e:
                logger.error(f"Error connecting to Milvus before creating collection: {e}")
        
        return super().from_documents(
            documents=documents,
            embedding=embedding,
            collection_name=collection_name,
            connection_args=connection_args,
            **kwargs,
        )
        
    def __init__(
        self,
        embedding_function,
        collection_name: str = "langchain",
        text_field: str = "text",
        connection_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize with connection args override."""
        # Force connection args to use service name in Docker
        connection_args = {
            "host": "milvus", 
            "port": "19530",
            "uri": "http://milvus:19530"
        }
        logger.info(f"Overriding connection args in __init__ to: {connection_args}")
        
        super().__init__(
            embedding_function=embedding_function,
            collection_name=collection_name,
            text_field=text_field,
            connection_args=connection_args,
            **kwargs,
        )
from pymilvus import connections, utility
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import gc

from app.core.config import settings
from app.core.metrics import measure_time, EMBEDDING_RETRIEVAL_DURATION, ERROR_COUNTER


class VectorStoreManager:
    """
    Manages vector stores for document embeddings.
    
    Features:
    - Batch processing for efficient document indexing
    - Connection management
    - Performance metrics collection
    - Memory management
    """
    
    def __init__(self):
        """Initialize the vector store manager."""
        self._stores: Dict[str, VectorStore] = {}
        self._connected = False
        
    def connect(self) -> None:
        """Establish connection to Milvus server."""
        if not self._connected:
            try:
                # Close any existing connection first
                try:
                    if pymilvus.connections.has_connection("default"):
                        pymilvus.connections.disconnect("default")
                        logger.info("Disconnected from previous Milvus connection")
                except Exception as disconnect_error:
                    logger.warning(f"Error disconnecting from Milvus: {disconnect_error}")
                
                # Log the connection attempt
                logger.info(f"Attempting to connect to Milvus at milvus:19530")
                
                # CRITICAL: Force set environment variables again
                os.environ["MILVUS_HOST"] = "milvus"
                os.environ["MILVUS_PORT"] = "19530"
                
                # Explicitly set connection parameters
                connection_params = {
                    "alias": "default",
                    "host": "milvus",  # Use Docker service name
                    "port": "19530",
                    # Added uri parameter as a backup
                    "uri": "http://milvus:19530"
                }
                logger.info(f"Connection parameters: {connection_params}")
                
                # Attempt connection with retries
                max_retries = 5
                retry_count = 0
                last_error = None
                
                while retry_count < max_retries:
                    try:
                        # Connect to Milvus server with explicit parameters
                        pymilvus.connections.connect(**connection_params)
                        
                        # Wait for connection to establish
                        time.sleep(1)
                        
                        # Verify the connection
                        if not pymilvus.connections.has_connection("default"):
                            raise RuntimeError("Connection not established")
                            
                        # Test the connection with a simple operation
                        collections = utility.list_collections()
                        logger.info(f"Successfully connected to Milvus. Available collections: {collections}")
                        self._connected = True
                        break
                        
                    except Exception as connect_error:
                        last_error = connect_error
                        retry_count += 1
                        logger.warning(f"Connection attempt {retry_count}/{max_retries} failed: {connect_error}")
                        
                        # Wait before retrying
                        time.sleep(2)
                
                if not self._connected:
                    logger.error(f"Failed to connect to Milvus after {max_retries} attempts: {last_error}")
                    raise RuntimeError(f"Milvus connection failed: {str(last_error)}")
                
                logger.info("Successfully connected to Milvus server at milvus:19530")
                
            except Exception as e:
                ERROR_COUNTER.labels(
                    error_type="ConnectionError",
                    component="milvus"
                ).inc()
                logger.error(f"Failed to connect to Milvus: {e}")
                raise RuntimeError(f"Failed to connect to Milvus: {str(e)}")
    
    def disconnect(self) -> None:
        """Disconnect from Milvus server."""
        if self._connected:
            try:
                # Intentar desconectar de todas las conexiones
                if pymilvus.connections.has_connection("default"):
                    connections.disconnect("default")
                
                # Reiniciar el estado de conexiones
                self._connected = False
                logger.info("Disconnected from Milvus server")
            except Exception as e:
                logger.warning(f"Error disconnecting from Milvus: {e}")
                
    def get_collection(
        self, 
        collection_name: str, 
        embedding_model: Embeddings
    ) -> VectorStore:
        """
        Get a vector store collection, creating it if it doesn't exist.
        
        Args:
            collection_name: Name of the collection
            embedding_model: Embedding model to use
            
        Returns:
            Milvus vector store
        """
        # Check if we already have this store cached
        cache_key = collection_name
        if cache_key in self._stores:
            return self._stores[cache_key]
        
        # Connect to Milvus if not already connected
        self.connect()
        
        # Check if collection exists
        try:
            if utility.has_collection(collection_name):
                logger.info(f"Loading existing Milvus collection: '{collection_name}'")
                
                # Create fresh connection to ensure we're connecting to the right instance
                if pymilvus.connections.has_connection("default"):
                    pymilvus.connections.disconnect("default")
                    logger.info("Disconnected from previous Milvus connection")
                    
                # Explicitly set connection parameters
                connection_params = {
                    "alias": "default",
                    "host": "milvus", 
                    "port": "19530"
                }
                logger.info(f"Creating new connection with parameters: {connection_params}")
                pymilvus.connections.connect(**connection_params)
                
                # Verify the connection
                if not pymilvus.connections.has_connection("default"):
                    raise RuntimeError("Failed to establish connection to Milvus before accessing collection")
                
                # Create and cache vector store with explicit connection parameters
                connection_args = {"host": "milvus", "port": "19530"}
                logger.info(f"Accessing collection with connection args: {connection_args}")
                
                vector_store = Milvus(
                    collection_name=collection_name,
                    embedding_function=embedding_model,
                    auto_id=True,
                    connection_args=connection_args
                )
                
                self._stores[cache_key] = vector_store
            else:
                logger.warning(f"Collection '{collection_name}' does not exist in Milvus")
                vector_store = None
                
            return vector_store
            
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type="MilvusError",
                component="get_collection"
            ).inc()
            logger.error(f"Error accessing Milvus collection '{collection_name}': {e}")
            raise RuntimeError(f"Failed to access Milvus collection: {str(e)}")
    
    @measure_time(EMBEDDING_RETRIEVAL_DURATION, {"collection": "default"})
    def create_collection(
        self,
        documents: List[Document],
        embedding_model: Embeddings,
        collection_name: str,
        batch_size: int = 100,
        force_recreate: bool = None
    ) -> VectorStore:
        """
        Create a new vector store collection with the provided documents.
        
        Args:
            documents: Documents to insert into the collection
            embedding_model: Embedding model to use
            collection_name: Name of the collection
            batch_size: Number of documents to process in each batch
            force_recreate: If True, will drop any existing collection with the same name.
                           If False, will keep existing collections and return them.
                           If None, will use the value from DONT_KEEP_COLLECTIONS in .env
            
        Returns:
            Milvus vector store
        """
        # Connect to Milvus if not already connected
        self.connect()
        
        try:
            # Use the .env variable if force_recreate is not provided
            from app.core.config import settings
            if force_recreate is None:
                force_recreate = settings.DONT_KEEP_COLLECTIONS
                logger.info(f"Using DONT_KEEP_COLLECTIONS={force_recreate} from environment")
            
            # Check if collection already exists
            if utility.has_collection(collection_name):
                if force_recreate:
                    logger.warning(f"Collection '{collection_name}' already exists and force_recreate=True, dropping it")
                    utility.drop_collection(collection_name)
                else:
                    logger.info(f"Collection '{collection_name}' already exists and force_recreate=False, reusing it")
                    # Return the existing collection instead of recreating it
                    return self.get_collection(collection_name, embedding_model)
            
            logger.info(f"Creating new Milvus collection '{collection_name}' with {len(documents)} documents")
            
            # Process documents in batches
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for batch_idx in tqdm(range(total_batches), desc="Inserting document batches"):
                # Extract batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(documents))
                batch = documents[start_idx:end_idx]
                
                # Create vector store from batch
                if batch_idx == 0:
                    # Create fresh connection to ensure we're connecting to the right instance
                    if pymilvus.connections.has_connection("default"):
                        pymilvus.connections.disconnect("default")
                        logger.info("Disconnected from previous Milvus connection")
                        
                    # Explicitly set connection parameters
                    connection_params = {
                        "alias": "default",
                        "host": "milvus", 
                        "port": "19530"
                    }
                    logger.info(f"Creating new connection with parameters: {connection_params}")
                    pymilvus.connections.connect(**connection_params)
                    
                    # Verify the connection before creating the collection
                    if not pymilvus.connections.has_connection("default"):
                        raise RuntimeError("Failed to establish connection to Milvus before creating collection")
                    
                    # Create the collection with explicit connection parameters
                    connection_args = {"host": "milvus", "port": "19530"}
                    logger.info(f"Creating collection with connection args: {connection_args}")
                    
                    # Try to create the collection with retries
                    max_retries = 3
                    retry_count = 0
                    last_error = None
                    
                    while retry_count < max_retries:
                        try:
                            logger.info(f"Attempt {retry_count+1}/{max_retries} to create collection")
                            vector_store = Milvus.from_documents(
                                documents=batch,
                                embedding=embedding_model,
                                collection_name=collection_name,
                                auto_id=True,
                                connection_args=connection_args
                            )
                            logger.info(f"Successfully created collection on attempt {retry_count+1}")
                            break
                        except Exception as e:
                            last_error = e
                            logger.warning(f"Attempt {retry_count+1}/{max_retries} failed: {e}")
                            retry_count += 1
                            
                            if retry_count < max_retries:
                                # Try to reconnect before the next attempt
                                try:
                                    if pymilvus.connections.has_connection("default"):
                                        pymilvus.connections.disconnect("default")
                                    logger.info("Reconnecting to Milvus before retry...")
                                    pymilvus.connections.connect(**connection_params)
                                    time.sleep(2)  # Give it time to establish connection
                                except Exception as reconnect_error:
                                    logger.warning(f"Error reconnecting: {reconnect_error}")
                    
                    # If all retries failed, raise the last error
                    if last_error is not None and retry_count == max_retries:
                        raise last_error
                    
                    # Check if vector_store was successfully created
                    if 'vector_store' not in locals():
                        raise RuntimeError("Failed to create vector store - variable not initialized")
                    
                    # Cache the vector store
                    self._stores[collection_name] = vector_store
                else:
                    # Subsequent batches add to existing collection
                    vector_store.add_documents(batch)
                
                # Log progress for large collections
                if batch_idx > 0 and batch_idx % 10 == 0:
                    logger.info(f"Processed {batch_idx * batch_size}/{len(documents)} documents")
                    
                    # Force garbage collection to free memory
                    if batch_idx % 50 == 0:
                        gc.collect()
            
            logger.info(f"Created Milvus collection '{collection_name}' with {len(documents)} documents")
            return vector_store
            
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type="MilvusError",
                component="create_collection"
            ).inc()
            logger.error(f"Error creating Milvus collection '{collection_name}': {e}")
            raise RuntimeError(f"Failed to create Milvus collection: {str(e)}")
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[Document],
        embedding_model: Embeddings,
        batch_size: int = 100,
        force_recreate: bool = None
    ) -> None:
        """
        Add documents to an existing collection.
        
        Args:
            collection_name: Name of the collection
            documents: Documents to add
            embedding_model: Embedding model to use
            batch_size: Number of documents to process in each batch
            force_recreate: If True and collection exists, will recreate it instead of adding to it.
                            If None, will use the value from DONT_KEEP_COLLECTIONS in .env
        """
        # Use the .env variable if force_recreate is not provided
        from app.core.config import settings
        if force_recreate is None:
            force_recreate = settings.DONT_KEEP_COLLECTIONS
            logger.info(f"Using DONT_KEEP_COLLECTIONS={force_recreate} from environment")
            
        # Get the vector store
        vector_store = self.get_collection(collection_name, embedding_model)
        
        if not vector_store or force_recreate:
            if not vector_store:
                logger.warning(f"Collection '{collection_name}' does not exist, creating it")
            elif force_recreate:
                logger.warning(f"Collection '{collection_name}' exists but force_recreate=True, recreating it")
            self.create_collection(documents, embedding_model, collection_name, batch_size, force_recreate)
            return
        
        try:
            logger.info(f"Adding {len(documents)} documents to collection '{collection_name}'")
            
            # Process documents in batches
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for batch_idx in tqdm(range(total_batches), desc="Adding document batches"):
                # Extract batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(documents))
                batch = documents[start_idx:end_idx]
                
                # Add batch to collection
                vector_store.add_documents(batch)
                
                # Log progress for large collections
                if batch_idx > 0 and batch_idx % 10 == 0:
                    logger.info(f"Added {batch_idx * batch_size}/{len(documents)} documents")
                    
                    # Force garbage collection to free memory
                    if batch_idx % 50 == 0:
                        gc.collect()
            
            logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")
            
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type="MilvusError",
                component="add_documents"
            ).inc()
            logger.error(f"Error adding documents to collection '{collection_name}': {e}")
            raise RuntimeError(f"Failed to add documents to collection: {str(e)}")
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if the collection was deleted successfully, False otherwise
        """
        # Connect to Milvus if not already connected
        self.connect()
        
        try:
            # Check if collection exists
            if utility.has_collection(collection_name):
                # Drop the collection
                utility.drop_collection(collection_name)
                logger.info(f"Deleted collection '{collection_name}'")
                
                # Remove from cache
                if collection_name in self._stores:
                    del self._stores[collection_name]
                
                # Also delete parent collection if it exists
                parent_collection_name = f"{collection_name}_parents"
                try:
                    from app.models.document_store import document_store_manager
                    document_store_manager.delete_collection(parent_collection_name)
                    logger.info(f"Deleted parent collection '{parent_collection_name}'")
                except Exception as parent_err:
                    logger.warning(f"Failed to delete parent collection '{parent_collection_name}': {parent_err}")
                
                return True
            else:
                logger.warning(f"Collection '{collection_name}' does not exist")
                return False
                
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type="MilvusError",
                component="delete_collection"
            ).inc()
            logger.error(f"Error deleting collection '{collection_name}': {e}")
            raise RuntimeError(f"Failed to delete collection: {str(e)}")
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection statistics
        """
        # Connect to Milvus if not already connected
        self.connect()
        
        try:
            # Check if collection exists
            if not utility.has_collection(collection_name):
                logger.warning(f"Collection '{collection_name}' does not exist")
                return {"exists": False}
            
            # Get entity count - handle pymilvus API differences
            try:
                count = utility.get_entity_num(collection_name)
            except AttributeError:
                # Newer pymilvus versions may use different methods
                try:
                    # Try to get collection and then get num_entities
                    coll = Collection(collection_name)
                    coll.load()
                    count = coll.num_entities
                except Exception as e2:
                    logger.warning(f"Could not get entity count for '{collection_name}': {e2}")
                    count = 0
            
            # Get collection info - handle pymilvus API differences
            try:
                info = utility.get_collection_stats(collection_name)
            except AttributeError:
                try:
                    # Try to get collection and extract stats
                    coll = Collection(collection_name)
                    schema = coll.schema
                    info = {"schema": str(schema)}
                except Exception as e2:
                    logger.warning(f"Could not get collection stats for '{collection_name}': {e2}")
                    info = {}
            
            # Return stats
            return {
                "exists": True,
                "count": count,
                "stats": info,
            }
            
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type="MilvusError",
                component="get_collection_stats"
            ).inc()
            logger.error(f"Error getting stats for collection '{collection_name}': {e}")
            return {"exists": False, "error": str(e)}
    
    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        # Connect to Milvus if not already connected
        self.connect()
        
        try:
            return utility.list_collections()
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type="MilvusError",
                component="list_collections"
            ).inc()
            logger.error(f"Error listing collections: {e}")
            return []


# Global instance
vector_store_manager = VectorStoreManager()