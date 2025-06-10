# IMPORTANT: Set environment variables BEFORE importing any modules
import os
import time
import asyncio
from datetime import datetime

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
        
        # Ensure all documents have normalized metadata to prevent Milvus errors
        # Esto evita los errores: "Insert missed an field" en la colección
        normalized_documents = normalize_document_metadata(documents)
        logger.info(f"Normalized metadata for {len(normalized_documents)} documents before sending to Milvus")
        
        return super().from_documents(
            documents=normalized_documents,
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
        
    def add_documents(self, documents: List[Document], **kwargs):
        """
        Sobrescribe el método add_documents para normalizar los metadatos antes de agregarlo a Milvus.
        Evita errores de 'Insert missed an field'
        """
        # Normalizar metadatos antes de agregar documentos
        normalized_documents = normalize_document_metadata(documents)
        logger.info(f"Normalized metadata for {len(normalized_documents)} documents in add_documents")
        
        # Llamar al método original con los documentos normalizados
        return super().add_documents(normalized_documents, **kwargs)
from pymilvus import connections, utility
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import gc

from app.core.config import settings
from app.core.metrics import measure_time, EMBEDDING_RETRIEVAL_DURATION, ERROR_COUNTER

def normalize_document_metadata(documents):
    """
    Normaliza los metadatos de los documentos para asegurar que todos los campos requeridos estén presentes.
    Esto previene errores de 'Insert missed an field' de Milvus cuando faltan campos.
    """
    # Lista de todos los campos que pueden ser requeridos por Milvus
    required_fields = ["source", "file_type", "page_number", "total_pages", "sheet_name", 
                      "sheet_index", "total_sheets", "width", "height"]
    
    # Valores por defecto para campos faltantes
    default_values = {
        "source": "unknown",
        "file_type": "text",
        "page_number": 1,
        "total_pages": 1,
        "sheet_name": "",
        "sheet_index": 0,
        "total_sheets": 1,
        "width": 612,
        "height": 864
    }
    
    for doc in documents:
        if not hasattr(doc, 'metadata') or doc.metadata is None:
            doc.metadata = {}
            
        # Asegurar que todos los campos requeridos estén presentes
        for field, default_value in default_values.items():
            if field not in doc.metadata:
                doc.metadata[field] = default_value
            elif doc.metadata[field] is None:
                # Si el campo existe pero es None, asignarle el valor por defecto
                doc.metadata[field] = default_value
                
        # Convertir valores numéricos a int/float según sea necesario
        numeric_fields = ["page_number", "total_pages", "sheet_index", "total_sheets", "width", "height"]
        for field in numeric_fields:
            if field in doc.metadata:
                try:
                    # Intentar convertir a entero primero (para los campos que deberían ser enteros)
                    if field in ["page_number", "total_pages", "sheet_index", "total_sheets"]:
                        doc.metadata[field] = int(doc.metadata[field])
                    # Para dimensiones, convertir a float
                    else:
                        doc.metadata[field] = float(doc.metadata[field])
                except (ValueError, TypeError):
                    # Si falla la conversión, usar el valor por defecto
                    doc.metadata[field] = default_values[field]
    
    return documents


class MilvusConnectionPool:
    """Connection pool for Milvus database connections."""
    
    def __init__(self, max_connections: int = 10, timeout: int = 30):
        self.max_connections = max_connections
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_connections)
        self.connections = []
        self.connection_params = {}
        self._lock = asyncio.Lock()
        self._health_check_interval = 60  # seconds
        self._last_health_check = time.time()
    
    async def initialize(self, connection_params: Dict[str, Any]):
        """Initialize connection pool with parameters."""
        self.connection_params = connection_params
        logger.info(f"Initializing Milvus connection pool with {self.max_connections} connections")
        
        # Create initial connections
        for i in range(min(3, self.max_connections)):  # Start with 3 connections
            try:
                await self._create_connection(f"pool_connection_{i}")
            except Exception as e:
                logger.warning(f"Failed to create initial connection {i}: {e}")
    
    async def _create_connection(self, alias: str) -> bool:
        """Create a new connection."""
        try:
            # Disconnect if already exists
            if pymilvus.connections.has_connection(alias):
                pymilvus.connections.disconnect(alias)
            
            # Create new connection
            pymilvus.connections.connect(alias=alias, **self.connection_params)
            
            # Test connection
            collections = utility.list_collections(using=alias)
            logger.debug(f"Created connection {alias} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create connection {alias}: {e}")
            return False
    
    async def get_connection(self):
        """Get a connection from the pool."""
        await self.semaphore.acquire()
        
        try:
            # Health check if needed
            now = time.time()
            if now - self._last_health_check > self._health_check_interval:
                await self._health_check()
                self._last_health_check = now
            
            # Return default connection for now
            # In a full implementation, we'd rotate through pool connections
            return "default"
            
        except Exception as e:
            self.semaphore.release()
            raise e
    
    def release_connection(self, connection_alias: str):
        """Release a connection back to the pool."""
        self.semaphore.release()
    
    async def _health_check(self):
        """Perform health check on connections."""
        logger.debug("Performing Milvus connection pool health check")
        
        try:
            # Test default connection
            if pymilvus.connections.has_connection("default"):
                collections = utility.list_collections()
                logger.debug(f"Connection pool health check passed. Collections: {len(collections)}")
            else:
                # Recreate default connection
                await self._create_connection("default")
                
        except Exception as e:
            logger.warning(f"Connection pool health check failed: {e}")
            # Try to recreate default connection
            try:
                await self._create_connection("default")
            except Exception as e2:
                logger.error(f"Failed to recreate connection: {e2}")
    
    async def cleanup(self):
        """Clean up all connections."""
        logger.info("Cleaning up Milvus connection pool...")
        
        try:
            # Close all named connections
            for alias in self.connections:
                try:
                    if pymilvus.connections.has_connection(alias):
                        pymilvus.connections.disconnect(alias)
                except Exception as e:
                    logger.warning(f"Error disconnecting {alias}: {e}")
            
            # Close default connection
            if pymilvus.connections.has_connection("default"):
                pymilvus.connections.disconnect("default")
                
        except Exception as e:
            logger.error(f"Error during connection pool cleanup: {e}")


class VectorStoreManager:
    """
    Enhanced vector store manager for production use with connection pooling.
    
    Features:
    - Connection pooling for efficient resource usage
    - Health monitoring and automatic recovery
    - Circuit breaker pattern for resilience
    - Background maintenance tasks
    - Performance metrics and monitoring
    - Thread-safe operations
    """
    
    def __init__(self):
        """Initialize the enhanced vector store manager."""
        self._stores: Dict[str, VectorStore] = {}
        self._connection_pool: Optional[MilvusConnectionPool] = None
        self._connected = False
        self._initialization_lock = asyncio.Lock()
        
        # Health monitoring
        self._health_status = {
            "is_healthy": False,
            "last_check": None,
            "error_count": 0,
            "last_error": None
        }
        
        # Circuit breaker
        self._circuit_breaker = {
            "state": "closed",  # closed, open, half-open
            "failures": 0,
            "last_failure": None,
            "failure_threshold": 5,
            "recovery_timeout": 300  # 5 minutes
        }
        
        # Performance metrics
        self._metrics = {
            "connection_attempts": 0,
            "successful_connections": 0,
            "failed_connections": 0,
            "collections_accessed": 0,
            "documents_processed": 0
        }
        
    async def initialize_pools(self) -> None:
        """Initialize connection pools during startup."""
        async with self._initialization_lock:
            if self._connection_pool:
                logger.info("Connection pool already initialized")
                return
            
            logger.info("Initializing Milvus connection pools...")
            
            try:
                # CRITICAL: Force set environment variables
                os.environ["MILVUS_HOST"] = "milvus"
                os.environ["MILVUS_PORT"] = "19530"
                
                # Initialize connection pool
                self._connection_pool = MilvusConnectionPool(
                    max_connections=settings.MILVUS_MAX_CONNECTIONS,
                    timeout=settings.MILVUS_CONNECTION_TIMEOUT
                )
                
                # Connection parameters
                connection_params = {
                    "host": "milvus",
                    "port": "19530",
                    "uri": "http://milvus:19530"
                }
                
                await self._connection_pool.initialize(connection_params)
                
                # Test connection
                await self._test_connection()
                
                self._connected = True
                self._health_status["is_healthy"] = True
                self._health_status["last_check"] = datetime.now()
                self._metrics["successful_connections"] += 1
                
                logger.info("Milvus connection pools initialized successfully")
                
            except Exception as e:
                self._metrics["failed_connections"] += 1
                self._record_circuit_breaker_failure()
                logger.error(f"Failed to initialize connection pools: {e}")
                raise RuntimeError(f"Connection pool initialization failed: {str(e)}")
    
    async def _test_connection(self) -> None:
        """Test the connection pool."""
        if self._connection_pool:
            connection_alias = await self._connection_pool.get_connection()
            try:
                collections = utility.list_collections(using=connection_alias)
                logger.info(f"Connection test successful. Collections: {collections}")
            finally:
                self._connection_pool.release_connection(connection_alias)
        else:
            # Fallback to direct connection test
            collections = utility.list_collections()
            logger.info(f"Direct connection test successful. Collections: {collections}")
    
    def connect(self) -> None:
        """Establish connection to Milvus server (legacy sync method)."""
        if not self._connected:
            self._metrics["connection_attempts"] += 1
            
            # Check circuit breaker
            if self._is_circuit_breaker_open():
                logger.warning("Circuit breaker open for Milvus connections")
                raise RuntimeError("Milvus connections temporarily disabled due to repeated failures")
            
            try:
                # Close any existing connection first
                try:
                    if pymilvus.connections.has_connection("default"):
                        pymilvus.connections.disconnect("default")
                        logger.info("Disconnected from previous Milvus connection")
                except Exception as disconnect_error:
                    logger.warning(f"Error disconnecting from Milvus: {disconnect_error}")
                
                logger.info(f"Attempting to connect to Milvus at milvus:19530")
                
                # CRITICAL: Force set environment variables again
                os.environ["MILVUS_HOST"] = "milvus"
                os.environ["MILVUS_PORT"] = "19530"
                
                # Explicitly set connection parameters
                connection_params = {
                    "alias": "default",
                    "host": "milvus",
                    "port": "19530",
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
                        
                        # Update health and metrics
                        self._health_status["is_healthy"] = True
                        self._health_status["last_check"] = datetime.now()
                        self._health_status["error_count"] = 0
                        self._metrics["successful_connections"] += 1
                        self._reset_circuit_breaker()
                        
                        break
                        
                    except Exception as connect_error:
                        last_error = connect_error
                        retry_count += 1
                        logger.warning(f"Connection attempt {retry_count}/{max_retries} failed: {connect_error}")
                        
                        # Wait before retrying
                        time.sleep(2)
                
                if not self._connected:
                    self._metrics["failed_connections"] += 1
                    self._record_circuit_breaker_failure()
                    logger.error(f"Failed to connect to Milvus after {max_retries} attempts: {last_error}")
                    raise RuntimeError(f"Milvus connection failed: {str(last_error)}")
                
                logger.info("Successfully connected to Milvus server at milvus:19530")
                
            except Exception as e:
                self._metrics["failed_connections"] += 1
                self._record_circuit_breaker_failure()
                self._health_status["error_count"] += 1
                self._health_status["last_error"] = str(e)
                self._health_status["is_healthy"] = False
                
                ERROR_COUNTER.labels(
                    error_type="ConnectionError",
                    component="milvus"
                ).inc()
                logger.error(f"Failed to connect to Milvus: {e}")
                raise RuntimeError(f"Failed to connect to Milvus: {str(e)}")
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_breaker["state"] == "open":
            # Check if enough time has passed to try half-open
            last_failure = self._circuit_breaker["last_failure"]
            if last_failure and (time.time() - last_failure) > self._circuit_breaker["recovery_timeout"]:
                self._circuit_breaker["state"] = "half-open"
                logger.info("Milvus circuit breaker moving to half-open state")
                return False
            return True
        return False
    
    def _record_circuit_breaker_failure(self):
        """Record failure in circuit breaker."""
        self._circuit_breaker["failures"] += 1
        self._circuit_breaker["last_failure"] = time.time()
        
        # Open circuit breaker if too many failures
        if self._circuit_breaker["failures"] >= self._circuit_breaker["failure_threshold"]:
            self._circuit_breaker["state"] = "open"
            logger.warning("Milvus circuit breaker opened due to repeated failures")
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker on success."""
        self._circuit_breaker["failures"] = 0
        self._circuit_breaker["state"] = "closed"
        self._circuit_breaker["last_failure"] = None
    
    async def cleanup(self) -> None:
        """Clean up connection pools and resources."""
        logger.info("Cleaning up VectorStoreManager...")
        
        if self._connection_pool:
            await self._connection_pool.cleanup()
            self._connection_pool = None
        
        # Clear store cache
        self._stores.clear()
        
        self._connected = False
        logger.info("VectorStoreManager cleanup completed")
    
    def disconnect(self) -> None:
        """Disconnect from Milvus server (legacy sync method)."""
        if self._connected:
            try:
                # If we have a connection pool, use async cleanup
                if self._connection_pool:
                    logger.info("Use cleanup() method for proper async disconnection")
                
                # Fallback to direct disconnection
                if pymilvus.connections.has_connection("default"):
                    connections.disconnect("default")
                
                # Reset state
                self._connected = False
                self._health_status["is_healthy"] = False
                logger.info("Disconnected from Milvus server")
            except Exception as e:
                logger.warning(f"Error disconnecting from Milvus: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        # Perform quick health check if needed
        now = datetime.now()
        if (not self._health_status["last_check"] or 
            (now - self._health_status["last_check"]).seconds > 60):
            await self._perform_health_check()
        
        return {
            "connected": self._connected,
            "health_status": self._health_status,
            "circuit_breaker": self._circuit_breaker,
            "metrics": self._metrics,
            "connection_pool": {
                "initialized": self._connection_pool is not None,
                "max_connections": self._connection_pool.max_connections if self._connection_pool else 0
            },
            "cached_stores": len(self._stores)
        }
    
    async def _perform_health_check(self) -> None:
        """Perform health check on Milvus connection."""
        try:
            start_time = time.time()
            
            if self._connection_pool:
                # Use connection pool for health check
                connection_alias = await self._connection_pool.get_connection()
                try:
                    collections = utility.list_collections(using=connection_alias)
                    response_time = time.time() - start_time
                    
                    # Update health status
                    self._health_status.update({
                        "is_healthy": True,
                        "last_check": datetime.now(),
                        "error_count": 0,
                        "last_error": None,
                        "response_time": response_time,
                        "collections_count": len(collections)
                    })
                    
                    logger.debug(f"Health check passed in {response_time:.3f}s, {len(collections)} collections")
                    
                finally:
                    self._connection_pool.release_connection(connection_alias)
            else:
                # Direct health check
                collections = utility.list_collections()
                response_time = time.time() - start_time
                
                self._health_status.update({
                    "is_healthy": True,
                    "last_check": datetime.now(),
                    "error_count": 0,
                    "last_error": None,
                    "response_time": response_time,
                    "collections_count": len(collections)
                })
                
        except Exception as e:
            self._health_status.update({
                "is_healthy": False,
                "last_check": datetime.now(),
                "error_count": self._health_status.get("error_count", 0) + 1,
                "last_error": str(e)
            })
            logger.warning(f"Health check failed: {e}")
                
    def get_collection(
        self, 
        collection_name: str, 
        embedding_model: Embeddings
    ) -> VectorStore:
        """
        Get a vector store collection with enhanced error handling and caching.
        
        Args:
            collection_name: Name of the collection
            embedding_model: Embedding model to use
            
        Returns:
            Milvus vector store or None if not found
        """
        # Check if we already have this store cached
        cache_key = collection_name
        if cache_key in self._stores:
            logger.debug(f"Retrieved cached vector store for: {collection_name}")
            return self._stores[cache_key]
        
        # Update metrics
        self._metrics["collections_accessed"] += 1
        
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            logger.warning("Circuit breaker open for Milvus operations")
            return None
        
        try:
            # Connect to Milvus if not already connected
            if not self._connected:
                self.connect()
            
            # Check if collection exists
            if utility.has_collection(collection_name):
                logger.info(f"Loading existing Milvus collection: '{collection_name}'")
                
                # Use connection pool if available
                if self._connection_pool:
                    # For now, still use default connection but with pool management
                    # In future versions, we could implement per-collection connections
                    connection_args = {"host": "milvus", "port": "19530"}
                else:
                    # Ensure connection for legacy mode
                    if not pymilvus.connections.has_connection("default"):
                        connection_params = {
                            "alias": "default",
                            "host": "milvus", 
                            "port": "19530"
                        }
                        logger.info(f"Creating connection with parameters: {connection_params}")
                        pymilvus.connections.connect(**connection_params)
                    
                    connection_args = {"host": "milvus", "port": "19530"}
                
                logger.info(f"Accessing collection with connection args: {connection_args}")
                
                # Create vector store
                vector_store = Milvus(
                    collection_name=collection_name,
                    embedding_function=embedding_model,
                    auto_id=True,
                    connection_args=connection_args
                )
                
                # Cache the vector store
                self._stores[cache_key] = vector_store
                
                # Reset circuit breaker on success
                self._reset_circuit_breaker()
                
                logger.info(f"Successfully loaded collection: {collection_name}")
                return vector_store
            else:
                logger.warning(f"Collection '{collection_name}' does not exist in Milvus")
                return None
                
        except Exception as e:
            # Record failure
            self._record_circuit_breaker_failure()
            self._health_status["error_count"] += 1
            self._health_status["last_error"] = str(e)
            
            ERROR_COUNTER.labels(
                error_type="MilvusError",
                component="get_collection"
            ).inc()
            logger.error(f"Error accessing Milvus collection '{collection_name}': {e}")
            
            # Don't raise exception, return None for graceful degradation
            return None
    
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
                            
                            # Normalizar metadatos para prevenir errores de 'Insert missed an field'
                            normalized_batch = normalize_document_metadata(batch)
                            logger.info(f"Normalized metadata for {len(normalized_batch)} documents")
                            
                            vector_store = Milvus.from_documents(
                                documents=normalized_batch,
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
                
                # Normalizar metadatos para prevenir errores de 'Insert missed an field'
                normalized_batch = normalize_document_metadata(batch)
                logger.info(f"Normalized metadata for batch {batch_idx+1}/{total_batches} ({len(normalized_batch)} documents)")
                
                # Add batch to collection
                vector_store.add_documents(normalized_batch)
                
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