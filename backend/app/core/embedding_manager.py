import os
import asyncio
import time
from datetime import datetime, timedelta
# Temporarily comment out torch until we can install it
# import torch
from typing import Dict, Optional, Any, List, Union
import numpy as np
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import gc
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .config import settings
from .metrics import measure_time, EMBEDDING_CREATION_DURATION
from .cache import cache_result


class AzureOpenAIConnectionPool:
    """Connection pool for Azure OpenAI API calls to manage rate limits and connections."""
    
    def __init__(self, max_connections: int = 10, rate_limit_per_minute: int = 1000):
        self.max_connections = max_connections
        self.rate_limit_per_minute = rate_limit_per_minute
        self.executor = ThreadPoolExecutor(max_workers=max_connections)
        self.semaphore = asyncio.Semaphore(max_connections)
        self.request_times = []
        self.lock = asyncio.Lock()
    
    async def rate_limit_check(self):
        """Check and enforce rate limits."""
        async with self.lock:
            now = datetime.now()
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if now - t < timedelta(minutes=1)]
            
            # Check if we're within rate limit
            if len(self.request_times) >= self.rate_limit_per_minute:
                # Calculate wait time
                oldest_in_window = min(self.request_times)
                wait_time = 60 - (now - oldest_in_window).seconds
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time} seconds")
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self.request_times.append(now)
    
    async def execute_with_rate_limit(self, func, *args, **kwargs):
        """Execute function with rate limiting and connection pooling."""
        await self.rate_limit_check()
        
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    def cleanup(self):
        """Clean up the thread pool."""
        self.executor.shutdown(wait=True)


class EmbeddingManager:
    """
    Enhanced Manages embedding models for production use with thread-safety and connection pooling.
    
    Features:
    - Singleton pattern with thread-safety
    - Pre-loading of models during startup
    - Connection pooling for Azure OpenAI
    - Circuit breaker pattern for resilience
    - Memory management and cleanup
    - Metrics collection and monitoring
    - Automatic retries and failover
    - Rate limiting for API calls
    """
    
    _instance = None
    _creation_lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._creation_lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            logger.info("Initializing Enhanced EmbeddingManager")
            
            # Thread-safe data structures
            self._models: Dict[str, Embeddings] = {}
            self._device_map: Dict[str, str] = {}
            self._model_locks: Dict[str, asyncio.Lock] = {}
            self._connection_pools: Dict[str, AzureOpenAIConnectionPool] = {}
            
            # Circuit breaker state
            self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
            
            # Health monitoring
            self._model_health: Dict[str, Dict[str, Any]] = {}
            self._last_health_check = datetime.now()
            
            # Performance metrics
            self._embedding_cache_hits = 0
            self._embedding_cache_misses = 0
            self._total_embeddings_generated = 0
            
            # Initialization state
            self._startup_completed = False
            self._initialization_lock = asyncio.Lock()
            
            self._init_device_map()
            self._initialized = True
    
    def _init_device_map(self) -> None:
        """Initialize device mappings for models."""
        # Use CPU only
        default_device = "cpu"
        
        logger.info(f"Default device for embeddings: {default_device}")
        
        # Assign default device to the unified model
        self._device_map = {
            settings.EMBEDDING_MODEL_NAME: default_device,
        }
    
    async def startup_initialize(self) -> None:
        """
        Complete initialization during application startup.
        Pre-loads models and sets up connection pools.
        """
        async with self._initialization_lock:
            if self._startup_completed:
                logger.info("EmbeddingManager startup already completed")
                return
            
            logger.info("Starting EmbeddingManager startup initialization...")
            start_time = time.time()
            
            try:
                # 1. Initialize connection pools
                await self._initialize_connection_pools()
                
                # 2. Pre-load main embedding model
                await self._preload_models()
                
                # 3. Initialize circuit breakers
                self._initialize_circuit_breakers()
                
                # 4. Perform initial health check
                await self._perform_health_check()
                
                self._startup_completed = True
                initialization_time = time.time() - start_time
                
                logger.info(f"EmbeddingManager startup completed in {initialization_time:.2f}s")
                
            except Exception as e:
                logger.error(f"EmbeddingManager startup failed: {e}")
                raise RuntimeError(f"EmbeddingManager initialization failed: {str(e)}")
    
    async def _initialize_connection_pools(self) -> None:
        """Initialize connection pools for API services."""
        logger.info("Initializing connection pools...")
        
        # Azure OpenAI connection pool
        azure_pool = AzureOpenAIConnectionPool(
            max_connections=settings.AZURE_OPENAI_MAX_CONNECTIONS,
            rate_limit_per_minute=settings.AZURE_OPENAI_RATE_LIMIT
        )
        self._connection_pools["azure_openai"] = azure_pool
        
        logger.info("Connection pools initialized")
    
    async def _preload_models(self) -> None:
        """Pre-load embedding models during startup."""
        logger.info("Pre-loading embedding models...")
        
        # Pre-load main model
        main_model_name = settings.EMBEDDING_MODEL_NAME
        try:
            await self._load_model_async(main_model_name)
            logger.info(f"Pre-loaded main embedding model: {main_model_name}")
        except Exception as e:
            logger.error(f"Failed to pre-load main model {main_model_name}: {e}")
            raise
    
    def initialize_model(self, model_name: Optional[str] = None) -> None:
        """
        Initialize the unified embedding model (legacy sync method).
        
        Args:
            model_name: Name of the embedding model
        """
        model_name = model_name or settings.EMBEDDING_MODEL_NAME
        
        # Load model if it's not already loaded
        if model_name not in self._models:
            self._models[model_name] = self._load_embedding_model(model_name)
            logger.info(f"Loaded unified embedding model: {model_name}")
    
    async def _load_model_async(self, model_name: str) -> Embeddings:
        """
        Async version of model loading with proper lock management.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded embedding model
        """
        # Get or create lock for this model
        if model_name not in self._model_locks:
            self._model_locks[model_name] = asyncio.Lock()
        
        async with self._model_locks[model_name]:
            # Double-check if model was loaded while waiting for lock
            if model_name in self._models:
                return self._models[model_name]
            
            # Load model
            logger.info(f"Loading embedding model: {model_name}")
            model = await self._load_embedding_model_with_pool(model_name)
            
            if model:
                self._models[model_name] = model
                # Initialize health tracking
                self._model_health[model_name] = {
                    "is_healthy": True,
                    "last_check": datetime.now(),
                    "error_count": 0,
                    "success_count": 0,
                    "avg_response_time": 0.0
                }
                return model
            else:
                raise RuntimeError(f"Failed to load model: {model_name}")
    
    async def _load_embedding_model_with_pool(self, model_name: str) -> Optional[Embeddings]:
        """
        Load embedding model using connection pool.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded embedding model or None if failed
        """
        try:
            # Check circuit breaker
            if self._is_circuit_breaker_open(model_name):
                logger.warning(f"Circuit breaker open for model: {model_name}")
                return None
            
            # Try to free memory before loading
            self._manage_memory()
            
            if model_name == "azure_openai" or model_name == settings.EMBEDDING_MODEL_NAME:
                # Use connection pool for Azure OpenAI
                pool = self._connection_pools.get("azure_openai")
                if pool:
                    embedding_model = await pool.execute_with_rate_limit(
                        self._create_azure_openai_model
                    )
                else:
                    # Fallback to direct creation
                    embedding_model = self._create_azure_openai_model()
                    
                logger.info(f"Loaded Azure OpenAI embedding model: {settings.AZURE_OPENAI_EMBEDDING_MODEL}")
                
                # Reset circuit breaker on success
                self._reset_circuit_breaker(model_name)
                
                return embedding_model
            else:
                # For now, redirect to Azure OpenAI
                logger.warning(f"HuggingFace models disabled, using Azure OpenAI for {model_name}")
                return await self._load_embedding_model_with_pool("azure_openai")
                
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {e}")
            self._record_circuit_breaker_failure(model_name)
            
            # Try memory cleanup and raise
            self._manage_memory(force=True)
            return None
    
    def _create_azure_openai_model(self) -> AzureOpenAIEmbeddings:
        """Create Azure OpenAI embedding model."""
        return AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            model=settings.AZURE_OPENAI_EMBEDDING_MODEL
        )
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for models."""
        for model_name in [settings.EMBEDDING_MODEL_NAME, "azure_openai"]:
            self._circuit_breakers[model_name] = {
                "failures": 0,
                "last_failure": None,
                "state": "closed"  # closed, open, half-open
            }
        logger.info("Circuit breakers initialized for embedding models")
    
    def _is_circuit_breaker_open(self, model_name: str) -> bool:
        """Check if circuit breaker is open for a model."""
        breaker = self._circuit_breakers.get(model_name, {})
        
        if breaker.get("state") == "open":
            # Check if enough time has passed to try half-open
            last_failure = breaker.get("last_failure")
            if last_failure and (time.time() - last_failure) > settings.CIRCUIT_BREAKER_TIMEOUT:
                breaker["state"] = "half-open"
                logger.info(f"Circuit breaker for {model_name} moving to half-open state")
                return False
            return True
        
        return False
    
    def _record_circuit_breaker_failure(self, model_name: str):
        """Record failure in circuit breaker."""
        breaker = self._circuit_breakers.get(model_name, {})
        breaker["failures"] = breaker.get("failures", 0) + 1
        breaker["last_failure"] = time.time()
        
        # Open circuit breaker if too many failures
        if breaker["failures"] >= settings.CIRCUIT_BREAKER_THRESHOLD:
            breaker["state"] = "open"
            logger.warning(f"Circuit breaker opened for model {model_name}")
    
    def _reset_circuit_breaker(self, model_name: str):
        """Reset circuit breaker on success."""
        breaker = self._circuit_breakers.get(model_name, {})
        breaker["failures"] = 0
        breaker["state"] = "closed"
        breaker["last_failure"] = None
    
    async def _perform_health_check(self) -> None:
        """Perform health check on all loaded models."""
        logger.info("Performing health check on embedding models...")
        
        for model_name, model in self._models.items():
            try:
                start_time = time.time()
                
                # Test embedding generation
                test_text = "health check test"
                embedding = await self._generate_embedding_with_pool(model, test_text, model_name)
                
                response_time = time.time() - start_time
                
                # Update health status
                health = self._model_health.get(model_name, {})
                health["is_healthy"] = True
                health["last_check"] = datetime.now()
                health["success_count"] = health.get("success_count", 0) + 1
                health["avg_response_time"] = response_time
                health["error_count"] = 0
                
                logger.info(f"Health check passed for {model_name}: {response_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Health check failed for {model_name}: {e}")
                
                # Update health status
                health = self._model_health.get(model_name, {})
                health["is_healthy"] = False
                health["last_check"] = datetime.now()
                health["error_count"] = health.get("error_count", 0) + 1
        
        self._last_health_check = datetime.now()
    
    async def _generate_embedding_with_pool(self, model: Embeddings, text: str, model_name: str) -> List[float]:
        """Generate embedding using connection pool."""
        pool = self._connection_pools.get("azure_openai")
        
        if pool and "azure" in model_name.lower():
            return await pool.execute_with_rate_limit(model.embed_query, text)
        else:
            # Direct call for non-Azure models
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, model.embed_query, text)
    
    @property
    def model(self) -> Embeddings:
        """Get the unified embedding model, loading it if necessary."""
        model_name = settings.EMBEDDING_MODEL_NAME
        if model_name not in self._models:
            # Check if we're in startup mode
            if self._startup_completed:
                logger.warning(f"Model {model_name} not pre-loaded during startup, loading now...")
            
            self._models[model_name] = self._load_embedding_model(model_name)
            logger.info(f"Loaded unified embedding model: {model_name}")
        return self._models[model_name]
    
    async def get_model_async(self, model_name: str) -> Embeddings:
        """
        Async version to get model with proper connection pooling.
        
        Args:
            model_name: Name of the embedding model to get
            
        Returns:
            The requested embedding model
        """
        if model_name not in self._models:
            return await self._load_model_async(model_name)
        return self._models[model_name]
    
    def get_model(self, model_name: str) -> Embeddings:
        """
        Get a specified embedding model, loading it if necessary.
        
        Args:
            model_name: Name of the embedding model to get
            
        Returns:
            The requested embedding model
            
        Raises:
            ValueError: If the model name is invalid
        """
        if model_name not in self._models:
            self._models[model_name] = self._load_embedding_model(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        return self._models[model_name]
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _load_embedding_model(self, model_name: str) -> Embeddings:
        """
        Load an embedding model with retry logic.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            An instance of the embedding model
            
        Raises:
            ValueError: If the specified device is not supported
            RuntimeError: If the model cannot be loaded
        """
        try:
            # Try to free memory before loading a new model
            self._manage_memory()
            
            if model_name == "azure_openai":
                # Azure OpenAI embeddings
                embedding_model = AzureOpenAIEmbeddings(
                    azure_deployment=settings.AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    model=settings.AZURE_OPENAI_EMBEDDING_MODEL
                )
                logger.info(f"Loaded Azure OpenAI embedding model: {settings.AZURE_OPENAI_EMBEDDING_MODEL}")
            else:
                # For now, just return Azure OpenAI embeddings for any model
                logger.warning(f"HuggingFace models are disabled, using Azure OpenAI instead for {model_name}")
                embedding_model = AzureOpenAIEmbeddings(
                    azure_deployment=settings.AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    model=settings.AZURE_OPENAI_EMBEDDING_MODEL
                )
                
                # Original HuggingFace code - temporarily disabled
                # Get device from map or use default
                # device = self._device_map.get(model_name, "cpu")
                # 
                # # Hugging Face embeddings
                # model_kwargs = {"device": device}
                # encode_kwargs = {"normalize_embeddings": True}  # For cosine similarity
                # 
                # embedding_model = HuggingFaceBgeEmbeddings(
                #     model_name=model_name,
                #     model_kwargs=model_kwargs,
                #     encode_kwargs=encode_kwargs,
                # )
                # logger.info(f"Loaded Hugging Face embedding model: {model_name} on {device}")
            
            return embedding_model
            
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {e}")
            # Try to free memory and reload
            self._manage_memory(force=True)
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")
    
    def _manage_memory(self, force: bool = False) -> None:
        """
        Manage memory to prevent OOM errors.
        
        Args:
            force: Whether to force memory cleanup regardless of usage
        """
        # Simplified memory management without torch
        if force:
            logger.info("Cleaning up memory")
            
            # Run garbage collection
            gc.collect()
    
    async def embed_texts_async(
        self, 
        texts: List[str], 
        model_name: Optional[str] = None,
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with async batching and connection pooling.
        
        Args:
            texts: List of texts to embed
            model_name: Name of the model to use
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Get the specified model or default to the unified model
        if not model_name:
            model_name = settings.EMBEDDING_MODEL_NAME
        
        model = await self.get_model_async(model_name)
        
        # Track metrics
        start_time = time.time()
        self._total_embeddings_generated += len(texts)
        
        try:
            # Use connection pooling for batched processing
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Use connection pool for embedding generation
                batch_embeddings = await self._generate_embeddings_batch_with_pool(
                    model, batch_texts, model_name
                )
                all_embeddings.extend(batch_embeddings)
                
                # Log progress for large batches
                if len(texts) > batch_size and i % (batch_size * 10) == 0 and i > 0:
                    logger.debug(f"Embedded {i}/{len(texts)} texts")
                
                # Update health metrics
                if model_name in self._model_health:
                    health = self._model_health[model_name]
                    health["success_count"] = health.get("success_count", 0) + len(batch_texts)
            
            # Record success metrics
            processing_time = time.time() - start_time
            logger.debug(f"Embedded {len(texts)} texts in {processing_time:.2f}s")
            
            return all_embeddings
            
        except Exception as e:
            # Record error metrics
            if model_name in self._model_health:
                health = self._model_health[model_name]
                health["error_count"] = health.get("error_count", 0) + 1
                health["is_healthy"] = False
            
            logger.error(f"Error in async embedding generation: {e}")
            raise
    
    async def _generate_embeddings_batch_with_pool(
        self, 
        model: Embeddings, 
        texts: List[str], 
        model_name: str
    ) -> List[List[float]]:
        """Generate embeddings for a batch using connection pool."""
        pool = self._connection_pools.get("azure_openai")
        
        if pool and "azure" in model_name.lower():
            return await pool.execute_with_rate_limit(model.embed_documents, texts)
        else:
            # Direct call for non-Azure models
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, model.embed_documents, texts)
    
    @cache_result(prefix="embed_texts")
    @measure_time(EMBEDDING_CREATION_DURATION, {"model": "default"})
    def embed_texts(
        self, 
        texts: List[str], 
        model_name: Optional[str] = None,
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with batching (legacy sync method).
        
        Args:
            texts: List of texts to embed
            model_name: Name of the model to use (defaults to German model)
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Check if we have startup completed - prefer async method
        if self._startup_completed:
            # Run async version in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in async context, create new task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.embed_texts_async(texts, model_name, batch_size))
                        return future.result()
                else:
                    return loop.run_until_complete(self.embed_texts_async(texts, model_name, batch_size))
            except Exception as e:
                logger.warning(f"Failed to use async embedding method, falling back to sync: {e}")
        
        # Fallback to original sync method
        # Get the specified model or default to the unified model
        if not model_name:
            model = self.model
            model_name = settings.EMBEDDING_MODEL_NAME
        else:
            model = self.get_model(model_name)
        
        # Use batching to avoid OOM errors with large input
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Log progress for large batches
            if len(texts) > batch_size and i % (batch_size * 10) == 0 and i > 0:
                logger.debug(f"Embedded {i}/{len(texts)} texts")
        
        return all_embeddings
    
    async def embed_query_async(
        self, 
        query: str, 
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for a single query with connection pooling (async version).
        
        Args:
            query: Text to embed
            model_name: Name of the model to use
            
        Returns:
            Embedding vector as a numpy array
        """
        # Get the specified model or default to the unified model
        if not model_name:
            model_name = settings.EMBEDDING_MODEL_NAME
        
        model = await self.get_model_async(model_name)
        
        # Track metrics
        start_time = time.time()
        self._total_embeddings_generated += 1
        
        try:
            # Generate embedding with connection pool
            embedding = await self._generate_embedding_with_pool(model, query, model_name)
            
            # Update health metrics
            processing_time = time.time() - start_time
            if model_name in self._model_health:
                health = self._model_health[model_name]
                health["success_count"] = health.get("success_count", 0) + 1
                health["avg_response_time"] = (
                    health.get("avg_response_time", 0) * 0.9 + processing_time * 0.1
                )
            
            # Convert to numpy array if it's a list
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            return embedding
            
        except Exception as e:
            # Record error metrics
            if model_name in self._model_health:
                health = self._model_health[model_name]
                health["error_count"] = health.get("error_count", 0) + 1
                health["is_healthy"] = False
            
            logger.error(f"Error in async query embedding: {e}")
            raise

    @cache_result(prefix="embed_query")
    def embed_query(
        self, 
        query: str, 
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for a single query (legacy sync method).
        
        Args:
            query: Text to embed
            model_name: Name of the model to use (defaults to German model)
            
        Returns:
            Embedding vector as a numpy array
        """
        # Check if we have startup completed - prefer async method
        if self._startup_completed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in async context, create new task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.embed_query_async(query, model_name))
                        return future.result()
                else:
                    return loop.run_until_complete(self.embed_query_async(query, model_name))
            except Exception as e:
                logger.warning(f"Failed to use async query embedding, falling back to sync: {e}")
        
        # Fallback to original sync method
        # Get the specified model or default to the unified model
        if not model_name:
            model = self.model
            model_name = settings.EMBEDDING_MODEL_NAME
        else:
            model = self.get_model(model_name)
        
        # Create metrics label
        labels = {"model": model_name}
        
        # Generate and time the embedding
        with measure_time(EMBEDDING_CREATION_DURATION, labels):
            embedding = model.embed_query(query)
        
        # Convertir a numpy array si es una lista
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        return embedding
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of embedding models."""
        return {
            "startup_completed": self._startup_completed,
            "models_loaded": list(self._models.keys()),
            "connection_pools": {
                name: {
                    "max_connections": pool.max_connections,
                    "rate_limit": pool.rate_limit_per_minute,
                    "active_requests": len(pool.request_times)
                }
                for name, pool in self._connection_pools.items()
            },
            "model_health": self._model_health,
            "circuit_breakers": self._circuit_breakers,
            "performance_metrics": {
                "cache_hits": self._embedding_cache_hits,
                "cache_misses": self._embedding_cache_misses,
                "total_embeddings": self._total_embeddings_generated,
                "last_health_check": self._last_health_check.isoformat()
            }
        }
    
    async def cleanup(self) -> None:
        """Clean up resources during shutdown."""
        logger.info("Cleaning up EmbeddingManager...")
        
        # Clean up connection pools
        for name, pool in self._connection_pools.items():
            try:
                pool.cleanup()
                logger.info(f"Cleaned up connection pool: {name}")
            except Exception as e:
                logger.warning(f"Error cleaning up pool {name}: {e}")
        
        # Clear models and memory
        self._models.clear()
        self._manage_memory(force=True)
        
        logger.info("EmbeddingManager cleanup completed")
    
    def clear_models(self) -> None:
        """Unload all models and clear the cache to free memory (legacy method)."""
        self._models.clear()
        self._manage_memory(force=True)
        logger.info("Cleared all embedding models")


# Global instance
embedding_manager = EmbeddingManager()