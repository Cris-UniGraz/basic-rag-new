import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pymilvus import utility
import weakref
from datetime import datetime, timedelta

from app.core.config import settings
from app.core.embedding_manager import embedding_manager
from app.core.metrics import measure_time, EMBEDDING_RETRIEVAL_DURATION, ERROR_COUNTER
from app.core.async_metadata_processor import async_metadata_processor, MetadataType
from app.models.vector_store import vector_store_manager
from app.models.document_store import document_store_manager


class RetrieverHealthStatus:
    """Tracks health status of a retriever."""
    
    def __init__(self):
        self.is_healthy = True
        self.last_check = datetime.now()
        self.error_count = 0
        self.last_error = None
        self.response_times = []
        self.last_success = datetime.now()
    
    def record_success(self, response_time: float):
        """Record a successful operation."""
        self.is_healthy = True
        self.error_count = 0
        self.last_success = datetime.now()
        self.response_times.append(response_time)
        
        # Keep only last 10 response times
        if len(self.response_times) > 10:
            self.response_times.pop(0)
    
    def record_error(self, error: Exception):
        """Record an error."""
        self.error_count += 1
        self.last_error = str(error)
        
        # Mark as unhealthy if too many errors
        if self.error_count >= settings.MAX_RETRIEVER_ERRORS:
            self.is_healthy = False
    
    def get_avg_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def should_recover(self) -> bool:
        """Check if retriever should attempt recovery."""
        # Try recovery after 5 minutes of being unhealthy
        return not self.is_healthy and (datetime.now() - self.last_check).seconds > 300


class RetrieverCache:
    """Cache for retrievers with TTL and health monitoring."""
    
    def __init__(self, ttl_seconds: int = 3600):  # 1 hour TTL
        self._cache: Dict[str, Any] = {}
        self._health: Dict[str, RetrieverHealthStatus] = {}
        self._access_times: Dict[str, datetime] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._ttl = ttl_seconds
        self._global_lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get retriever from cache if healthy and not expired."""
        async with self._global_lock:
            # Check if key exists and is not expired
            if key not in self._cache:
                return None
                
            # Check TTL
            if key in self._access_times:
                age = (datetime.now() - self._access_times[key]).seconds
                if age > self._ttl:
                    await self._remove(key)
                    return None
            
            # Check health
            if key in self._health and not self._health[key].is_healthy:
                # Try recovery if enough time has passed
                if self._health[key].should_recover():
                    logger.info(f"Attempting recovery for retriever: {key}")
                    await self._remove(key)
                    return None
                else:
                    return None
            
            # Update access time
            self._access_times[key] = datetime.now()
            return self._cache[key]
    
    async def set(self, key: str, retriever: Any) -> None:
        """Set retriever in cache."""
        async with self._global_lock:
            self._cache[key] = retriever
            self._access_times[key] = datetime.now()
            self._locks[key] = asyncio.Lock()
            
            # Initialize health status if not exists
            if key not in self._health:
                self._health[key] = RetrieverHealthStatus()
    
    async def get_lock(self, key: str) -> asyncio.Lock:
        """Get lock for specific retriever."""
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]
    
    async def record_success(self, key: str, response_time: float):
        """Record successful operation."""
        if key in self._health:
            self._health[key].record_success(response_time)
    
    async def record_error(self, key: str, error: Exception):
        """Record error for retriever."""
        if key in self._health:
            self._health[key].record_error(error)
    
    async def get_health_status(self, key: str) -> Optional[RetrieverHealthStatus]:
        """Get health status for retriever."""
        return self._health.get(key)
    
    async def _remove(self, key: str):
        """Remove retriever from cache."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._locks.pop(key, None)
        # Keep health status for recovery tracking
    
    async def clear(self):
        """Clear all cached retrievers."""
        async with self._global_lock:
            self._cache.clear()
            self._access_times.clear()
            self._locks.clear()
            # Keep health status for metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_retrievers": len(self._cache),
            "healthy_retrievers": sum(1 for h in self._health.values() if h.is_healthy),
            "unhealthy_retrievers": sum(1 for h in self._health.values() if not h.is_healthy),
            "cache_keys": list(self._cache.keys()),
            "health_summary": {
                key: {
                    "is_healthy": health.is_healthy,
                    "error_count": health.error_count,
                    "avg_response_time": health.get_avg_response_time(),
                    "last_success": health.last_success.isoformat()
                }
                for key, health in self._health.items()
            }
        }


class PersistentRAGService:
    """
    Persistent RAG Service that maintains retrievers in memory for production use.
    
    Features:
    - Singleton pattern with thread-safety
    - Persistent retriever cache with health monitoring
    - Graceful degradation when retrievers fail
    - Async initialization during application startup
    - Circuit breaker pattern for resilience
    - Background health monitoring
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls, llm_provider: Any = None):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(PersistentRAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, llm_provider: Any = None):
        """Initialize the persistent RAG service."""
        if not getattr(self, '_initialized', False):
            self.llm_provider = llm_provider
            self._retriever_cache = RetrieverCache(ttl_seconds=settings.RETRIEVER_CACHE_TTL)
            self._startup_completed = False
            self._initialization_lock = asyncio.Lock()
            self._background_tasks = set()
            self._circuit_breakers = {}
            self._degradation_mode = "full"  # full, intermediate, basic
            
            # Initialize RetrieverManager integration
            self._retriever_manager = None
            self._retriever_pool_manager = None
            self._use_advanced_management = True
            
            self._initialized = True
            
            logger.info("PersistentRAGService instance created")
    
    async def startup_initialization(self) -> None:
        """
        Complete initialization during application startup.
        Pre-loads retrievers for existing collections.
        """
        async with self._initialization_lock:
            if self._startup_completed:
                logger.info("Startup initialization already completed")
                return
            
            logger.info("Starting PersistentRAGService initialization...")
            start_time = time.time()
            
            try:
                # 1. Ensure core services are initialized
                await self._initialize_core_services()
                
                # 2. Initialize advanced retriever management
                await self._initialize_retriever_management()
                
                # 3. Pre-load retrievers for existing collections (non-blocking)
                try:
                    await self._preload_retrievers()
                except Exception as e:
                    logger.warning(f"Retriever pre-loading failed during startup: {e}")
                    logger.info("Continuing without pre-loading - retrievers will be created on demand")
                
                # 4. Start background health monitoring
                await self._start_health_monitoring()
                
                # 5. Initialize circuit breakers
                self._initialize_circuit_breakers()
                
                self._startup_completed = True
                initialization_time = time.time() - start_time
                
                logger.info(f"PersistentRAGService initialization completed in {initialization_time:.2f}s")
                
                # Log async initialization metrics
                async_metadata_processor.log_async("INFO", 
                    "PersistentRAGService startup completed",
                    {
                        "initialization_time": initialization_time,
                        "preloaded_retrievers": len(self._retriever_cache._cache),
                        "degradation_mode": self._degradation_mode
                    })
                
            except Exception as e:
                logger.error(f"Failed to initialize PersistentRAGService: {e}")
                # Enable degradation mode
                self._degradation_mode = "basic"
                
                async_metadata_processor.log_async("ERROR", 
                    "PersistentRAGService startup failed - enabling degradation mode",
                    {
                        "error": str(e),
                        "degradation_mode": self._degradation_mode
                    }, priority=3)
                
                raise RuntimeError(f"PersistentRAGService initialization failed: {str(e)}")
    
    async def _initialize_core_services(self) -> None:
        """Initialize core services required for operation."""
        logger.info("Initializing core services...")
        
        # Initialize embedding manager with pre-loading
        embedding_manager.initialize_model(settings.EMBEDDING_MODEL_NAME)
        
        # Ensure vector store manager is connected
        vector_store_manager.connect()
        
        # Verify connections
        collections = vector_store_manager.list_collections()
        logger.info(f"Vector store connected. Available collections: {collections}")
    
    async def _initialize_retriever_management(self) -> None:
        """Initialize advanced retriever management components."""
        logger.info("Initializing advanced retriever management...")
        
        try:
            # Import here to avoid circular imports
            from app.core.retriever_manager import RetrieverManager
            from app.core.retriever_pool import retriever_pool_manager
            
            # Initialize RetrieverManager with dependency injection
            self._retriever_manager = RetrieverManager(rag_service_instance=self)
            await self._retriever_manager.initialize()
            
            # Initialize RetrieverPoolManager
            self._retriever_pool_manager = retriever_pool_manager
            
            logger.info("Advanced retriever management initialized successfully")
            
            # Log async initialization
            async_metadata_processor.log_async("INFO", 
                "Advanced retriever management initialized",
                {
                    "retriever_manager_active": True,
                    "pool_manager_active": True,
                    "management_mode": "advanced"
                })
                
        except Exception as e:
            logger.warning(f"Failed to initialize advanced retriever management: {e}")
            logger.info("Falling back to basic retriever management")
            
            # Fallback to basic management
            self._retriever_manager = None
            self._retriever_pool_manager = None
            self._use_advanced_management = False
            
            async_metadata_processor.log_async("WARNING", 
                "Advanced retriever management failed - using fallback",
                {
                    "error": str(e),
                    "management_mode": "basic"
                })
    
    async def _preload_retrievers(self) -> None:
        """Pre-load retrievers for existing collections."""
        logger.info("Pre-loading retrievers for existing collections...")
        
        try:
            # Get list of available collections
            collections = vector_store_manager.list_collections()
            
            if not collections:
                logger.info("No collections found - skipping retriever pre-loading")
                return
            
            # Limit pre-loading to avoid timeout during startup
            max_preload_collections = 3  # Limit to 3 collections for faster startup
            collections_to_preload = collections[:max_preload_collections]
            
            if len(collections) > max_preload_collections:
                logger.info(f"Found {len(collections)} collections, pre-loading first {max_preload_collections} for faster startup")
            
            # Pre-load retrievers for main collections only
            preload_tasks = []
            for collection_name in collections_to_preload:
                # Skip system collections
                if collection_name.startswith('_') or collection_name.endswith('_parents'):
                    continue
                
                logger.info(f"Pre-loading retriever for collection: {collection_name}")
                task = asyncio.create_task(
                    self._preload_single_retriever(collection_name)
                )
                preload_tasks.append(task)
            
            # Wait for all pre-loading tasks with shorter timeout for startup
            if preload_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*preload_tasks, return_exceptions=True),
                        timeout=30  # Reduced timeout for faster startup
                    )
                except asyncio.TimeoutError:
                    logger.warning("Some retrievers failed to pre-load within startup timeout - continuing")
                    # Cancel remaining tasks
                    for task in preload_tasks:
                        if not task.done():
                            task.cancel()
            
            # Get final cache stats
            cache_stats = self._retriever_cache.get_stats()
            logger.info(f"Pre-loading completed. Cache stats: {cache_stats}")
            
            # Schedule background preloading for remaining collections if any
            remaining_collections = collections[max_preload_collections:]
            if remaining_collections:
                logger.info(f"Scheduling background pre-loading for {len(remaining_collections)} remaining collections")
                asyncio.create_task(self._background_preload_remaining(remaining_collections))
            
        except Exception as e:
            logger.error(f"Error during retriever pre-loading: {e}")
            # Don't raise - allow service to continue in degradation mode
    
    async def _preload_single_retriever(self, collection_name: str) -> None:
        """Pre-load a single retriever."""
        try:
            retriever = await self.get_persistent_retriever(
                collection_name, 
                settings.MAX_CHUNKS_CONSIDERED
            )
            
            if retriever:
                logger.info(f"Successfully pre-loaded retriever for: {collection_name}")
            else:
                logger.warning(f"Failed to pre-load retriever for: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error pre-loading retriever for {collection_name}: {e}")
    
    async def _background_preload_remaining(self, collections: List[str]) -> None:
        """Pre-load remaining collections in background after startup."""
        try:
            logger.info(f"Starting background pre-loading for {len(collections)} collections")
            await asyncio.sleep(5)  # Wait a bit after startup
            
            for collection_name in collections:
                # Skip system collections
                if collection_name.startswith('_') or collection_name.endswith('_parents'):
                    continue
                
                try:
                    logger.info(f"Background pre-loading retriever for collection: {collection_name}")
                    await self._preload_single_retriever(collection_name)
                    await asyncio.sleep(2)  # Small delay between background loads
                except Exception as e:
                    logger.warning(f"Background pre-loading failed for {collection_name}: {e}")
                    continue
            
            logger.info("Background pre-loading completed")
            
        except Exception as e:
            logger.error(f"Error in background pre-loading: {e}")
    
    async def _start_health_monitoring(self) -> None:
        """Start background health monitoring tasks."""
        logger.info("Starting background health monitoring...")
        
        # Create background task for health checks
        health_task = asyncio.create_task(self._health_monitor_loop())
        self._background_tasks.add(health_task)
        health_task.add_done_callback(self._background_tasks.discard)
        
        # Create background task for cache maintenance
        cache_maintenance_task = asyncio.create_task(self._cache_maintenance_loop())
        self._background_tasks.add(cache_maintenance_task)
        cache_maintenance_task.add_done_callback(self._background_tasks.discard)
        
        # Create background task for metrics collection
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self._background_tasks.add(metrics_task)
        metrics_task.add_done_callback(self._background_tasks.discard)
        
        # Create background task for performance optimization
        optimization_task = asyncio.create_task(self._performance_optimization_loop())
        self._background_tasks.add(optimization_task)
        optimization_task.add_done_callback(self._background_tasks.discard)
    
    async def _health_monitor_loop(self) -> None:
        """Background loop for health monitoring."""
        while True:
            try:
                await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL)
                await self._perform_health_checks()
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all cached retrievers."""
        cache_stats = self._retriever_cache.get_stats()
        
        # Log health summary
        async_metadata_processor.log_async("DEBUG", 
            "Retriever health check summary",
            {
                "total_retrievers": cache_stats["cached_retrievers"],
                "healthy_retrievers": cache_stats["healthy_retrievers"],
                "unhealthy_retrievers": cache_stats["unhealthy_retrievers"]
            })
        
        # Check if we need to adjust degradation mode
        if cache_stats["cached_retrievers"] > 0:
            health_ratio = cache_stats["healthy_retrievers"] / cache_stats["cached_retrievers"]
            
            if health_ratio < 0.3:
                self._degradation_mode = "basic"
                logger.warning("Switching to basic degradation mode due to low retriever health")
            elif health_ratio < 0.7:
                self._degradation_mode = "intermediate"
                logger.info("Switching to intermediate degradation mode")
            else:
                self._degradation_mode = "full"
    
    async def _cache_maintenance_loop(self) -> None:
        """Background loop for cache maintenance and cleanup."""
        logger.info("Starting cache maintenance loop")
        
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                # Perform cache maintenance
                await self._perform_cache_maintenance()
                
            except Exception as e:
                logger.error(f"Error in cache maintenance loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _perform_cache_maintenance(self) -> None:
        """Perform cache maintenance operations."""
        try:
            # Get cache stats before maintenance
            cache_stats_before = self._retriever_cache.get_stats()
            
            # Clean expired entries from basic cache
            expired_keys = []
            for key, access_time in self._retriever_cache._access_times.items():
                age = (datetime.now() - access_time).seconds
                if age > self._retriever_cache._ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                await self._retriever_cache._remove(key)
            
            # Clean unhealthy retrievers
            unhealthy_keys = []
            for key, health in self._retriever_cache._health.items():
                if not health.is_healthy and not health.should_recover():
                    unhealthy_keys.append(key)
            
            for key in unhealthy_keys:
                await self._retriever_cache._remove(key)
                logger.info(f"Removed unhealthy retriever from cache: {key}")
            
            # Get cache stats after maintenance
            cache_stats_after = self._retriever_cache.get_stats()
            
            # Log maintenance summary
            cleaned_count = len(expired_keys) + len(unhealthy_keys)
            if cleaned_count > 0:
                logger.info(f"Cache maintenance completed. Cleaned {cleaned_count} retrievers")
                
                async_metadata_processor.log_async("INFO", 
                    "Cache maintenance completed",
                    {
                        "expired_cleaned": len(expired_keys),
                        "unhealthy_cleaned": len(unhealthy_keys),
                        "cache_size_before": cache_stats_before["cached_retrievers"],
                        "cache_size_after": cache_stats_after["cached_retrievers"]
                    })
            
        except Exception as e:
            logger.error(f"Error performing cache maintenance: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Background loop for metrics collection and reporting."""
        logger.info("Starting metrics collection loop")
        
        while True:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                
                # Collect and report metrics
                await self._collect_and_report_metrics()
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _collect_and_report_metrics(self) -> None:
        """Collect and report comprehensive metrics."""
        try:
            # Get health status with all metrics
            health_status = await self.get_health_status()
            
            # Calculate key performance indicators
            cache_stats = health_status["cache_stats"]
            cache_utilization = 0
            
            if "max_size" in str(cache_stats):  # Check if cache has max size
                cache_utilization = cache_stats["cached_retrievers"] / 100  # Using default max size
            
            # Log comprehensive metrics
            async_metadata_processor.log_async("DEBUG", 
                "Periodic metrics collection",
                {
                    "service_status": health_status["service_status"],
                    "degradation_mode": health_status["degradation_mode"],
                    "management_mode": health_status["management_mode"],
                    "cache_utilization": cache_utilization,
                    "healthy_retrievers": cache_stats["healthy_retrievers"],
                    "unhealthy_retrievers": cache_stats["unhealthy_retrievers"],
                    "background_tasks_count": health_status["background_tasks"],
                    "circuit_breakers_status": health_status["circuit_breakers"]
                })
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    async def _performance_optimization_loop(self) -> None:
        """Background loop for performance optimization."""
        logger.info("Starting performance optimization loop")
        
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Perform performance optimizations
                await self._perform_performance_optimizations()
                
            except Exception as e:
                logger.error(f"Error in performance optimization loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _perform_performance_optimizations(self) -> None:
        """Perform performance optimizations based on current metrics."""
        try:
            # Get current health status
            health_status = await self.get_health_status()
            cache_stats = health_status["cache_stats"]
            
            # Analyze performance patterns
            optimization_actions = []
            
            # Check if we need to adjust degradation mode
            if cache_stats["unhealthy_retrievers"] > cache_stats["healthy_retrievers"]:
                if self._degradation_mode != "basic":
                    self._degradation_mode = "basic"
                    optimization_actions.append("degradation_mode_to_basic")
                    logger.warning("Switched to basic degradation mode due to high unhealthy retriever ratio")
            
            elif cache_stats["healthy_retrievers"] > 5 and self._degradation_mode != "full":
                self._degradation_mode = "full"
                optimization_actions.append("degradation_mode_to_full")
                logger.info("Switched to full degradation mode - sufficient healthy retrievers")
            
            # Check circuit breaker health
            circuit_breaker_issues = []
            for name, breaker in self._circuit_breakers.items():
                if breaker.get("state") == "open":
                    circuit_breaker_issues.append(name)
            
            if circuit_breaker_issues:
                optimization_actions.append(f"circuit_breakers_open_{len(circuit_breaker_issues)}")
            
            # Preload popular retrievers if needed (basic proactive optimization)
            if self._use_advanced_management and self._retriever_manager:
                try:
                    # Try to preload one retriever to test if system is healthy
                    collections = vector_store_manager.list_collections()
                    if collections:
                        test_collection = next((c for c in collections if not c.startswith('_')), None)
                        if test_collection:
                            test_retriever = await self._retriever_manager.get_retriever(
                                test_collection, 
                                settings.MAX_CHUNKS_CONSIDERED, 
                                preload=True
                            )
                            if test_retriever:
                                optimization_actions.append("preload_test_successful")
                except Exception as e:
                    logger.debug(f"Preload test failed during optimization: {e}")
                    optimization_actions.append("preload_test_failed")
            
            # Log optimization summary
            if optimization_actions:
                async_metadata_processor.log_async("INFO", 
                    "Performance optimization completed",
                    {
                        "actions_taken": optimization_actions,
                        "current_degradation_mode": self._degradation_mode,
                        "healthy_retrievers": cache_stats["healthy_retrievers"],
                        "unhealthy_retrievers": cache_stats["unhealthy_retrievers"],
                        "circuit_breaker_issues": circuit_breaker_issues
                    })
            
        except Exception as e:
            logger.error(f"Error performing performance optimizations: {e}")
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for resilience."""
        self._circuit_breakers = {
            "retriever_creation": {"failures": 0, "last_failure": None, "state": "closed"},
            "query_processing": {"failures": 0, "last_failure": None, "state": "closed"},
            "embedding_generation": {"failures": 0, "last_failure": None, "state": "closed"}
        }
        logger.info("Initialized circuit breakers")
    
    async def get_persistent_retriever(
        self,
        collection_name: str,
        top_k: int = 3,
        force_refresh: bool = False,
        priority: int = 1
    ) -> Optional[Any]:
        """
        Get a persistent retriever from cache or create new one using advanced management.
        
        Args:
            collection_name: Name of the collection
            top_k: Number of documents to retrieve
            force_refresh: Force recreation of retriever
            priority: Priority level (1=high, 2=medium, 3=low)
            
        Returns:
            Persistent retriever or None if failed
        """
        # Use advanced management if available
        if self._use_advanced_management and self._retriever_manager:
            try:
                logger.debug(f"Using advanced retriever management for: {collection_name}")
                
                # Force refresh if requested
                if force_refresh:
                    await self._retriever_manager.refresh_retriever(collection_name, top_k)
                
                # Get retriever from RetrieverManager
                retriever = await self._retriever_manager.get_retriever(
                    collection_name, 
                    top_k, 
                    priority=priority
                )
                
                if retriever:
                    logger.debug(f"Retrieved from RetrieverManager: {collection_name}_{top_k}")
                    return retriever
                else:
                    logger.warning(f"RetrieverManager failed to provide retriever for: {collection_name}")
                    # Fall back to basic management
                    
            except Exception as e:
                logger.error(f"Error using RetrieverManager: {e}")
                # Fall back to basic management
        
        # Fallback to basic cache-based management (basic_management)
        return await self._get_basic_persistent_retriever(collection_name, top_k, force_refresh)
    
    async def _get_basic_persistent_retriever(
        self,
        collection_name: str,
        top_k: int = 3,
        force_refresh: bool = False
    ) -> Optional[Any]:
        """
        Fallback method for basic persistent retriever management.
        """
        cache_key = f"{collection_name}_{top_k}"
        
        # Try to get from cache first
        if not force_refresh:
            cached_retriever = await self._retriever_cache.get(cache_key)
            if cached_retriever:
                logger.debug(f"Retrieved cached retriever for: {cache_key}")
                return cached_retriever
        
        # Get lock for this specific retriever
        retriever_lock = await self._retriever_cache.get_lock(cache_key)
        
        async with retriever_lock:
            # Double-check cache after acquiring lock
            if not force_refresh:
                cached_retriever = await self._retriever_cache.get(cache_key)
                if cached_retriever:
                    return cached_retriever
            
            # Create new retriever
            logger.info(f"Creating new persistent retriever for collection '{collection_name}' with top_k={top_k} (cache_key: {cache_key})")
            start_time = time.time()
            
            try:
                # Check circuit breaker
                if self._is_circuit_breaker_open("retriever_creation"):
                    logger.warning("Circuit breaker open for retriever creation")
                    return None
                
                # Create retriever using original RAG service logic
                retriever = await self._create_ensemble_retriever(collection_name, top_k)
                
                if retriever:
                    # Cache the retriever
                    await self._retriever_cache.set(cache_key, retriever)
                    
                    # Record success
                    creation_time = time.time() - start_time
                    await self._retriever_cache.record_success(cache_key, creation_time)
                    
                    # Reset circuit breaker
                    self._reset_circuit_breaker("retriever_creation")
                    
                    logger.info(f"Created and cached retriever for {cache_key} in {creation_time:.2f}s")
                    
                    # Log async metrics
                    async_metadata_processor.record_performance_async(
                        "persistent_retriever_creation",
                        creation_time,
                        True,
                        {
                            "collection": collection_name,
                            "top_k": top_k,
                            "degradation_mode": self._degradation_mode,
                            "management_type": "basic"
                        }
                    )
                    
                    return retriever
                else:
                    logger.warning(f"Failed to create retriever for: {cache_key}")
                    return None
                    
            except Exception as e:
                # Record error
                await self._retriever_cache.record_error(cache_key, e)
                self._record_circuit_breaker_failure("retriever_creation")
                
                logger.error(f"Error creating retriever for {cache_key}: {e}")
                
                # Log async error
                async_metadata_processor.log_async("ERROR", 
                    f"Failed to create persistent retriever",
                    {
                        "collection": collection_name,
                        "error": str(e),
                        "cache_key": cache_key
                    }, priority=3)
                
                return None
    
    async def get_pooled_retriever(
        self,
        collection_name: str,
        top_k: int = 3,
        timeout: float = 30.0,
        priority: int = 1
    ) -> Optional[Any]:
        """
        Get retriever from pool for high-concurrency scenarios.
        
        Args:
            collection_name: Name of the collection
            top_k: Number of documents to retrieve
            timeout: Maximum time to wait for available retriever
            priority: Priority level (1=high, 2=medium, 3=low)
            
        Returns:
            Pooled retriever or None if failed
        """
        # Use RetrieverPoolManager if available
        if self._use_advanced_management and self._retriever_pool_manager:
            try:
                logger.debug(f"Using pool management for: {collection_name}")
                
                # Get or create pool for this configuration
                pool = await self._retriever_pool_manager.get_or_create_pool(
                    collection_name=collection_name,
                    top_k=top_k,
                    min_size=1,
                    max_size=settings.RETRIEVER_POOL_MAX_SIZE if hasattr(settings, 'RETRIEVER_POOL_MAX_SIZE') else 3
                )
                
                if pool:
                    # Get retriever from pool
                    pooled_retriever = await pool.get_retriever(timeout=timeout)
                    
                    if pooled_retriever:
                        logger.debug(f"Retrieved from pool: {collection_name}_{top_k}")
                        return pooled_retriever.retriever_instance
                    else:
                        logger.warning(f"Pool failed to provide retriever for: {collection_name}")
                        
            except Exception as e:
                logger.error(f"Error using RetrieverPoolManager: {e}")
        
        # Fallback to regular persistent retriever
        return await self.get_persistent_retriever(collection_name, top_k, priority=priority)
    
    async def return_pooled_retriever(
        self,
        collection_name: str,
        top_k: int,
        retriever: Any,
        success: bool = True,
        response_time: float = 0.0
    ) -> None:
        """
        Return a pooled retriever after use.
        
        Args:
            collection_name: Name of the collection
            top_k: Number of documents retrieved
            retriever: The retriever instance to return
            success: Whether the operation was successful
            response_time: Time taken for the operation
        """
        if self._use_advanced_management and self._retriever_pool_manager:
            try:
                # Find the pool for this configuration
                pool_id = f"{collection_name}_{top_k}"
                
                # Get pool stats to find the retriever
                pool_stats = await self._retriever_pool_manager.get_global_stats()
                
                if pool_id in pool_stats.get("pool_stats", {}):
                    pool = self._retriever_pool_manager._pools.get(pool_id)
                    if pool:
                        # Find the pooled retriever instance
                        for instance in pool._instances.values():
                            if instance.retriever_instance == retriever:
                                await pool.return_retriever(instance, success, response_time)
                                logger.debug(f"Returned retriever to pool: {pool_id}")
                                return
                                
            except Exception as e:
                logger.error(f"Error returning pooled retriever: {e}")
        
        # If not pooled or error, nothing to do for regular persistent retrievers
    
    async def _create_ensemble_retriever(self, collection_name: str, top_k: int) -> Optional[Any]:
        """Create ensemble retriever using degradation mode."""
        try:
            # Import here to avoid circular imports
            from app.services.rag_service import RAGService
            
            # Create temporary RAG service for retriever creation
            temp_rag_service = RAGService(self.llm_provider)
            await temp_rag_service.ensure_initialized()
            
            # Create retriever based on degradation mode
            if self._degradation_mode == "basic":
                # Basic mode: only vector search
                return await self._create_basic_retriever(collection_name, top_k)
            elif self._degradation_mode == "intermediate":
                # Intermediate mode: vector + parent retriever
                return await temp_rag_service.get_retriever(collection_name, top_k, max_concurrency=2)
            else:
                # Full mode: complete ensemble
                return await temp_rag_service.get_retriever(collection_name, top_k)
                
        except Exception as e:
            logger.error(f"Error in _create_ensemble_retriever: {e}")
            return None
    
    async def _create_basic_retriever(self, collection_name: str, top_k: int) -> Optional[Any]:
        """Create basic vector-only retriever for degradation mode."""
        try:
            # Get embedding model
            embedding_model = embedding_manager.model
            
            # Get vector store
            vector_store = vector_store_manager.get_collection(collection_name, embedding_model)
            
            if vector_store:
                # Return simple vector retriever
                return vector_store.as_retriever(search_kwargs={"k": top_k})
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error creating basic retriever: {e}")
            return None
    
    def _is_circuit_breaker_open(self, breaker_name: str) -> bool:
        """Check if circuit breaker is open."""
        breaker = self._circuit_breakers.get(breaker_name, {})
        
        if breaker.get("state") == "open":
            # Check if enough time has passed to try half-open
            last_failure = breaker.get("last_failure")
            if last_failure and (time.time() - last_failure) > settings.CIRCUIT_BREAKER_TIMEOUT:
                breaker["state"] = "half-open"
                logger.info(f"Circuit breaker {breaker_name} moving to half-open state")
                return False
            return True
        
        return False
    
    def _record_circuit_breaker_failure(self, breaker_name: str):
        """Record failure in circuit breaker."""
        breaker = self._circuit_breakers.get(breaker_name, {})
        breaker["failures"] = breaker.get("failures", 0) + 1
        breaker["last_failure"] = time.time()
        
        # Open circuit breaker if too many failures
        if breaker["failures"] >= settings.CIRCUIT_BREAKER_THRESHOLD:
            breaker["state"] = "open"
            logger.warning(f"Circuit breaker {breaker_name} opened due to repeated failures")
    
    def _reset_circuit_breaker(self, breaker_name: str):
        """Reset circuit breaker on success."""
        breaker = self._circuit_breakers.get(breaker_name, {})
        breaker["failures"] = 0
        breaker["state"] = "closed"
        breaker["last_failure"] = None
    
    async def process_query_with_persistent_retrievers(
        self,
        query: str,
        collection_name: str,
        chat_history: List[Tuple[str, str]] = [],
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Process query using persistent retrievers.
        
        Args:
            query: User query
            collection_name: Collection to search
            chat_history: Chat history
            top_k: Number of documents to retrieve
            
        Returns:
            Query processing result
        """
        top_k = top_k or settings.MAX_CHUNKS_CONSIDERED
        
        try:
            # Check circuit breaker
            if self._is_circuit_breaker_open("query_processing"):
                logger.warning("Circuit breaker open for query processing")
                return {
                    'response': "Lo siento, el servicio está temporalmente no disponible. Por favor, inténtelo de nuevo en unos minutos.",
                    'sources': [],
                    'from_cache': False,
                    'processing_time': 0.0,
                    'error': 'circuit_breaker_open'
                }
            
            # Get persistent retriever
            retriever = await self.get_persistent_retriever(collection_name, top_k)
            
            if not retriever:
                logger.error(f"No retriever available for collection: {collection_name}")
                return {
                    'response': "Lo siento, no puedo acceder a la base de datos en este momento.",
                    'sources': [],
                    'from_cache': False,
                    'processing_time': 0.0,
                    'error': 'no_retriever'
                }
            
            # Process query using original RAG service
            from app.services.rag_service import RAGService
            temp_rag_service = RAGService(self.llm_provider)
            await temp_rag_service.ensure_initialized()
            
            start_time = time.time()
            result = await temp_rag_service.process_query(query, retriever, chat_history)
            processing_time = time.time() - start_time
            
            # Record success
            cache_key = f"{collection_name}_{top_k}"
            await self._retriever_cache.record_success(cache_key, processing_time)
            self._reset_circuit_breaker("query_processing")
            
            return result
            
        except Exception as e:
            # Record error
            cache_key = f"{collection_name}_{top_k}"
            await self._retriever_cache.record_error(cache_key, e)
            self._record_circuit_breaker_failure("query_processing")
            
            logger.error(f"Error processing query with persistent retrievers: {e}")
            
            return {
                'response': "Lo siento, ha ocurrido un error al procesar su consulta. Por favor, inténtelo de nuevo.",
                'sources': [],
                'from_cache': False,
                'processing_time': 0.0,
                'error': str(e)
            }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        cache_stats = self._retriever_cache.get_stats()
        
        # Get advanced management stats
        retriever_manager_stats = None
        pool_manager_stats = None
        
        if self._retriever_manager:
            try:
                retriever_manager_stats = await self._retriever_manager.get_stats()
            except Exception as e:
                logger.error(f"Error getting RetrieverManager stats: {e}")
        
        if self._retriever_pool_manager:
            try:
                pool_manager_stats = await self._retriever_pool_manager.get_global_stats()
            except Exception as e:
                logger.error(f"Error getting RetrieverPoolManager stats: {e}")
        
        return {
            "service_status": "healthy" if self._startup_completed else "initializing",
            "degradation_mode": self._degradation_mode,
            "management_mode": "advanced" if self._use_advanced_management else "basic",
            "cache_stats": cache_stats,
            "circuit_breakers": self._circuit_breakers,
            "background_tasks": len(self._background_tasks),
            "startup_completed": self._startup_completed,
            "advanced_management": {
                "retriever_manager_active": self._retriever_manager is not None,
                "pool_manager_active": self._retriever_pool_manager is not None,
                "retriever_manager_stats": retriever_manager_stats,
                "pool_manager_stats": pool_manager_stats
            }
        }
    
    async def force_refresh_retriever(self, collection_name: str, top_k: int = None) -> bool:
        """Force refresh of a specific retriever."""
        top_k = top_k or settings.MAX_CHUNKS_CONSIDERED
        
        try:
            retriever = await self.get_persistent_retriever(
                collection_name, 
                top_k, 
                force_refresh=True
            )
            return retriever is not None
        except Exception as e:
            logger.error(f"Error force refreshing retriever {collection_name}: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up resources during shutdown."""
        logger.info("Cleaning up PersistentRAGService...")
        
        try:
            # Clean up advanced retriever management
            if self._retriever_manager:
                await self._retriever_manager.cleanup()
                logger.info("RetrieverManager cleanup completed")
            
            if self._retriever_pool_manager:
                await self._retriever_pool_manager.cleanup()
                logger.info("RetrieverPoolManager cleanup completed")
            
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Clear cache
            await self._retriever_cache.clear()
            
            logger.info("PersistentRAGService cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during PersistentRAGService cleanup: {e}")


# Factory function for creating persistent RAG service
def create_persistent_rag_service(llm_provider: Any) -> PersistentRAGService:
    """
    Create and return persistent RAG service instance.
    
    Args:
        llm_provider: LLM service or callable
        
    Returns:
        PersistentRAGService instance
    """
    return PersistentRAGService(llm_provider)