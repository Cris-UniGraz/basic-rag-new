import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple
from collections import defaultdict, OrderedDict
from loguru import logger
import weakref
from dataclasses import dataclass, field
from enum import Enum
import json

from app.core.config import settings
from app.core.async_metadata_processor import async_metadata_processor


class RetrieverStatus(Enum):
    """Status of a retriever."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    COOLING_DOWN = "cooling_down"
    DEPRECATED = "deprecated"


@dataclass
class RetrieverMetrics:
    """Metrics for a retriever."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_used: Optional[datetime] = None
    creation_time: datetime = field(default_factory=datetime.now)
    total_uptime: float = 0.0
    error_rate: float = 0.0
    throughput_per_minute: float = 0.0
    
    def update_success(self, response_time: float):
        """Update metrics for a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.last_used = datetime.now()
        
        # Update average response time with exponential moving average
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * 0.9) + (response_time * 0.1)
        
        # Update error rate
        self.error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0
    
    def update_failure(self):
        """Update metrics for a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_used = datetime.now()
        self.error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0
    
    def get_uptime(self) -> float:
        """Get current uptime in seconds."""
        return (datetime.now() - self.creation_time).total_seconds()
    
    def calculate_throughput(self) -> float:
        """Calculate throughput per minute."""
        uptime_minutes = self.get_uptime() / 60
        if uptime_minutes > 0:
            self.throughput_per_minute = self.successful_requests / uptime_minutes
        return self.throughput_per_minute


@dataclass
class RetrieverInfo:
    """Information about a retriever instance."""
    retriever_id: str
    collection_name: str
    top_k: int
    retriever_instance: Any
    status: RetrieverStatus = RetrieverStatus.INITIALIZING
    metrics: RetrieverMetrics = field(default_factory=RetrieverMetrics)
    last_health_check: Optional[datetime] = None
    config_hash: str = ""
    tags: Set[str] = field(default_factory=set)
    priority: int = 1  # 1=high, 2=medium, 3=low
    
    def __post_init__(self):
        """Generate config hash after initialization."""
        config_data = {
            "collection_name": self.collection_name,
            "top_k": self.top_k,
            "timestamp": self.metrics.creation_time.isoformat()
        }
        self.config_hash = str(hash(json.dumps(config_data, sort_keys=True)))
    
    def is_healthy(self) -> bool:
        """Check if retriever is healthy."""
        return (
            self.status in [RetrieverStatus.READY, RetrieverStatus.BUSY] and
            self.metrics.error_rate < settings.MAX_RETRIEVER_ERROR_RATE and
            self.retriever_instance is not None
        )
    
    def should_refresh(self) -> bool:
        """Check if retriever should be refreshed."""
        age = (datetime.now() - self.metrics.creation_time).total_seconds()
        return (
            age > settings.RETRIEVER_MAX_AGE or
            self.metrics.error_rate > settings.RETRIEVER_ERROR_THRESHOLD or
            self.status == RetrieverStatus.ERROR
        )


class LRUCache:
    """LRU Cache with size limit and TTL."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        async with self._lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if key in self.timestamps:
                age = time.time() - self.timestamps[key]
                if age > self.ttl_seconds:
                    del self.cache[key]
                    del self.timestamps[key]
                    return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    async def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        async with self._lock:
            # Remove oldest items if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.timestamps.pop(oldest_key, None)
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    async def remove(self, key: str) -> bool:
        """Remove item from cache."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                self.timestamps.pop(key, None)
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        expired_count = sum(
            1 for timestamp in self.timestamps.values()
            if current_time - timestamp > self.ttl_seconds
        )
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "expired_items": expired_count,
            "ttl_seconds": self.ttl_seconds
        }


class RetrieverManager:
    """
    Advanced manager for retriever lifecycle with load balancing and optimization.
    
    Features:
    - LRU cache with TTL for retriever instances
    - Background initialization of popular retrievers
    - Automatic refresh of obsolete retrievers
    - Load balancing between multiple retriever instances
    - Comprehensive metrics and monitoring
    - Intelligent preloading based on usage patterns
    - Circuit breaker integration
    - Performance-based routing
    - Resource optimization and cleanup
    """
    
    def __init__(self, rag_service_instance=None):
        """Initialize the retriever manager."""
        self._retrievers: Dict[str, RetrieverInfo] = {}
        self._usage_stats: Dict[str, List[datetime]] = defaultdict(list)
        self._retriever_cache = LRUCache(
            max_size=settings.RETRIEVER_CACHE_MAX_SIZE,
            ttl_seconds=settings.RETRIEVER_CACHE_TTL
        )
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Load balancing
        self._load_balancer_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self._round_robin_counters: Dict[str, int] = defaultdict(int)
        
        # Locks for thread safety
        self._manager_lock = asyncio.Lock()
        self._retriever_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Injected RAG service instance to avoid circular initialization
        self._rag_service_instance = rag_service_instance
        
        # Metrics and circuit_breaker integration
        self._global_metrics = {
            "total_retrievers_created": 0,
            "total_requests_served": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "preload_hits": 0,
            "background_refreshes": 0
        }
        
        logger.info("RetrieverManager initialized")
    
    async def initialize(self) -> None:
        """Initialize the retriever manager and start background tasks."""
        logger.info("Starting RetrieverManager initialization...")
        
        try:
            # Start background tasks
            await self._start_background_tasks()
            
            # Preload popular retrievers if any exist
            await self._initial_preload()
            
            logger.info("RetrieverManager initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize RetrieverManager: {e}")
            raise RuntimeError(f"RetrieverManager initialization failed: {str(e)}")
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        logger.info("Starting background tasks for RetrieverManager")
        
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitor_loop())
        self._background_tasks.add(health_task)
        health_task.add_done_callback(self._background_tasks.discard)
        
        # Usage analytics task
        analytics_task = asyncio.create_task(self._usage_analytics_loop())
        self._background_tasks.add(analytics_task)
        analytics_task.add_done_callback(self._background_tasks.discard)
        
        # Preloading task
        preload_task = asyncio.create_task(self._preloading_loop())
        self._background_tasks.add(preload_task)
        preload_task.add_done_callback(self._background_tasks.discard)
        
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._background_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._background_tasks.discard)
    
    async def _initial_preload(self) -> None:
        """Perform initial preloading of retrievers."""
        logger.info("Skipping initial retriever preloading during startup to avoid circular dependencies")
        
        # Schedule preloading as background task after startup
        if self._rag_service_instance:
            asyncio.create_task(self._delayed_preload())
        
    async def _delayed_preload(self) -> None:
        """Perform delayed preloading after startup is complete."""
        try:
            # Wait a bit to ensure startup is complete
            await asyncio.sleep(5)
            logger.info("Starting delayed retriever preloading...")
            
            # Get available collections
            from app.models.vector_store import vector_store_manager
            collections = vector_store_manager.list_collections()
            
            if not collections:
                logger.info("No collections found for preloading")
                return
            
            # Preload retrievers for main collections (up to 3)
            preload_tasks = []
            for collection_name in collections[:3]:
                if not collection_name.startswith('_') and not collection_name.endswith('_parents'):
                    task = asyncio.create_task(
                        self.get_retriever(collection_name, settings.MAX_CHUNKS_CONSIDERED, preload=True)
                    )
                    preload_tasks.append(task)
            
            if preload_tasks:
                await asyncio.gather(*preload_tasks, return_exceptions=True)
                logger.info(f"Successfully preloaded {len(preload_tasks)} retrievers in background")
            
        except Exception as e:
            logger.error(f"Error during delayed preloading: {e}")
    
    async def get_retriever(
        self,
        collection_name: str,
        top_k: int,
        priority: int = 1,
        tags: Optional[Set[str]] = None,
        preload: bool = False
    ) -> Optional[Any]:
        """
        Get or create a retriever with intelligent caching and load balancing.
        
        Args:
            collection_name: Collection to retrieve from
            top_k: Number of documents to retrieve
            priority: Priority level (1=high, 2=medium, 3=low)
            tags: Optional tags for categorization
            preload: Whether this is a preload operation
            
        Returns:
            Retriever instance or None if failed
        """
        cache_key = f"{collection_name}_{top_k}"
        
        # Record usage statistics
        if not preload:
            self._usage_stats[cache_key].append(datetime.now())
            self._global_metrics["total_requests_served"] += 1
        
        # Try cache first
        cached_retriever = await self._retriever_cache.get(cache_key)
        if cached_retriever and cached_retriever.is_healthy():
            if not preload:
                self._global_metrics["cache_hits"] += 1
                cached_retriever.metrics.update_success(0.0)  # Cache hit has no retrieval time
            logger.debug(f"Cache hit for retriever: {cache_key}")
            return cached_retriever.retriever_instance
        
        if not preload:
            self._global_metrics["cache_misses"] += 1
        
        # Get retriever-specific lock to prevent duplicate creation
        async with self._retriever_locks[cache_key]:
            # Double-check cache after acquiring lock
            cached_retriever = await self._retriever_cache.get(cache_key)
            if cached_retriever and cached_retriever.is_healthy():
                if not preload:
                    self._global_metrics["cache_hits"] += 1
                    cached_retriever.metrics.update_success(0.0)
                return cached_retriever.retriever_instance
            
            # Create new retriever
            logger.info(f"Creating new retriever for collection '{collection_name}' with top_k={top_k} (cache_key: {cache_key})")
            start_time = time.time()
            
            try:
                # Create retriever using the factory method
                retriever_instance = await self._create_retriever_instance(collection_name, top_k)
                
                if retriever_instance:
                    # Create retriever info
                    retriever_info = RetrieverInfo(
                        retriever_id=cache_key,
                        collection_name=collection_name,
                        top_k=top_k,
                        retriever_instance=retriever_instance,
                        status=RetrieverStatus.READY,
                        priority=priority,
                        tags=tags or set()
                    )
                    
                    # Update metrics
                    creation_time = time.time() - start_time
                    retriever_info.metrics.update_success(creation_time)
                    
                    # Cache the retriever
                    await self._retriever_cache.set(cache_key, retriever_info)
                    
                    # Store in manager
                    async with self._manager_lock:
                        self._retrievers[cache_key] = retriever_info
                    
                    # Update global metrics
                    self._global_metrics["total_retrievers_created"] += 1
                    if preload:
                        self._global_metrics["preload_hits"] += 1
                    
                    logger.info(f"Created retriever for {cache_key} in {creation_time:.2f}s")
                    
                    # Log async metrics
                    async_metadata_processor.record_performance_async(
                        "retriever_creation",
                        creation_time,
                        True,
                        {
                            "collection": collection_name,
                            "top_k": top_k,
                            "priority": priority,
                            "preload": preload
                        }
                    )
                    
                    return retriever_instance
                else:
                    logger.warning(f"Failed to create retriever for: {cache_key}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error creating retriever for {cache_key}: {e}")
                
                # Log async error
                async_metadata_processor.log_async("ERROR", 
                    f"Failed to create retriever",
                    {
                        "collection": collection_name,
                        "error": str(e),
                        "cache_key": cache_key
                    }, priority=3)
                
                return None
    
    async def _create_retriever_instance(self, collection_name: str, top_k: int) -> Optional[Any]:
        """Create a retriever instance using the injected RAG service or fallback."""
        try:
            # Use injected RAG service instance if available
            if self._rag_service_instance:
                retriever = await self._rag_service_instance._get_basic_persistent_retriever(
                    collection_name, top_k
                )
                return retriever
            
            # Fallback: Create basic retriever directly using RAGService (avoid circular initialization)
            from app.services.rag_service import RAGService
            from app.services.llm_service import llm_service
            
            # Create temporary RAG service for basic retriever creation (no startup_initialization)
            temp_rag_service = RAGService(llm_service)
            await temp_rag_service.ensure_initialized()
            
            # Create basic retriever
            retriever = await temp_rag_service.get_retriever(collection_name, top_k)
            return retriever
            
        except Exception as e:
            logger.error(f"Error in _create_retriever_instance: {e}")
            return None
    
    async def refresh_retriever(self, collection_name: str, top_k: int) -> bool:
        """Force refresh a specific retriever."""
        cache_key = f"{collection_name}_{top_k}"
        
        try:
            async with self._retriever_locks[cache_key]:
                # Remove from cache
                await self._retriever_cache.remove(cache_key)
                
                # Remove from manager
                async with self._manager_lock:
                    self._retrievers.pop(cache_key, None)
                
                # Create new retriever
                new_retriever = await self.get_retriever(collection_name, top_k)
                
                if new_retriever:
                    self._global_metrics["background_refreshes"] += 1
                    logger.info(f"Successfully refreshed retriever: {cache_key}")
                    return True
                else:
                    logger.warning(f"Failed to refresh retriever: {cache_key}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error refreshing retriever {cache_key}: {e}")
            return False
    
    async def _health_monitor_loop(self) -> None:
        """Background loop for health monitoring."""
        logger.info("Starting health monitor loop")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL)
                
                # Get current retrievers
                async with self._manager_lock:
                    retriever_list = list(self._retrievers.items())
                
                # Check health of each retriever
                unhealthy_retrievers = []
                for cache_key, retriever_info in retriever_list:
                    try:
                        # Perform health check
                        is_healthy = await self._perform_health_check(retriever_info)
                        
                        if not is_healthy:
                            unhealthy_retrievers.append(cache_key)
                            logger.warning(f"Retriever {cache_key} is unhealthy")
                        
                    except Exception as e:
                        logger.error(f"Error checking health of {cache_key}: {e}")
                        unhealthy_retrievers.append(cache_key)
                
                # Refresh unhealthy retrievers
                for cache_key in unhealthy_retrievers:
                    collection_name, top_k = cache_key.rsplit('_', 1)
                    await self.refresh_retriever(collection_name, int(top_k))
                
                # Log health summary
                total_retrievers = len(retriever_list)
                healthy_retrievers = total_retrievers - len(unhealthy_retrievers)
                
                async_metadata_processor.log_async("DEBUG", 
                    "Retriever health check summary",
                    {
                        "total_retrievers": total_retrievers,
                        "healthy_retrievers": healthy_retrievers,
                        "unhealthy_retrievers": len(unhealthy_retrievers)
                    })
                
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _perform_health_check(self, retriever_info: RetrieverInfo) -> bool:
        """Perform health check on a specific retriever."""
        try:
            # Update last health check time
            retriever_info.last_health_check = datetime.now()
            
            # Check if retriever instance is still valid
            if retriever_info.retriever_instance is None:
                retriever_info.status = RetrieverStatus.ERROR
                return False
            
            # Check error rate
            if retriever_info.metrics.error_rate > settings.RETRIEVER_ERROR_THRESHOLD:
                retriever_info.status = RetrieverStatus.ERROR
                return False
            
            # Check if retriever is too old
            if retriever_info.should_refresh():
                retriever_info.status = RetrieverStatus.DEPRECATED
                return False
            
            # All checks passed
            if retriever_info.status == RetrieverStatus.ERROR:
                retriever_info.status = RetrieverStatus.READY
            
            return True
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            retriever_info.status = RetrieverStatus.ERROR
            return False
    
    async def _usage_analytics_loop(self) -> None:
        """Background loop for usage analytics."""
        logger.info("Starting usage analytics loop")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Analyze usage patterns
                await self._analyze_usage_patterns()
                
                # Update load balancer weights
                await self._update_load_balancer_weights()
                
            except Exception as e:
                logger.error(f"Error in usage analytics loop: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_usage_patterns(self) -> None:
        """Analyze usage patterns for optimization."""
        current_time = datetime.now()
        
        # Clean old usage data (keep last 24 hours)
        cutoff_time = current_time - timedelta(hours=24)
        
        for cache_key in list(self._usage_stats.keys()):
            # Filter recent usage
            recent_usage = [
                timestamp for timestamp in self._usage_stats[cache_key]
                if timestamp > cutoff_time
            ]
            self._usage_stats[cache_key] = recent_usage
            
            # Remove entries with no recent usage
            if not recent_usage:
                del self._usage_stats[cache_key]
    
    async def _update_load_balancer_weights(self) -> None:
        """Update load balancer weights based on performance."""
        async with self._manager_lock:
            for cache_key, retriever_info in self._retrievers.items():
                # Calculate weight based on performance metrics
                base_weight = 1.0
                
                # Adjust for error rate
                error_penalty = retriever_info.metrics.error_rate * 2
                
                # Adjust for response time
                response_time_penalty = min(retriever_info.metrics.avg_response_time / 10, 0.5)
                
                # Calculate final weight
                final_weight = max(0.1, base_weight - error_penalty - response_time_penalty)
                
                self._load_balancer_weights[cache_key] = final_weight
    
    async def _preloading_loop(self) -> None:
        """Background loop for intelligent preloading."""
        logger.info("Starting preloading loop")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                # Identify popular retrievers that should be preloaded
                await self._identify_and_preload_popular()
                
            except Exception as e:
                logger.error(f"Error in preloading loop: {e}")
                await asyncio.sleep(300)
    
    async def _identify_and_preload_popular(self) -> None:
        """Identify and preload popular retrievers."""
        current_time = datetime.now()
        
        # Analyze usage in last hour
        usage_counts = {}
        for cache_key, timestamps in self._usage_stats.items():
            recent_usage = [
                t for t in timestamps
                if (current_time - t).total_seconds() < 3600
            ]
            usage_counts[cache_key] = len(recent_usage)
        
        # Sort by usage count
        popular_retrievers = sorted(
            usage_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Preload top 5 popular retrievers that aren't already cached
        preload_tasks = []
        for cache_key, usage_count in popular_retrievers[:5]:
            if usage_count > 2:  # Only preload if used more than 2 times in last hour
                collection_name, top_k = cache_key.rsplit('_', 1)
                
                # Check if already cached
                cached = await self._retriever_cache.get(cache_key)
                if not cached or not cached.is_healthy():
                    task = asyncio.create_task(
                        self.get_retriever(collection_name, int(top_k), preload=True)
                    )
                    preload_tasks.append(task)
        
        if preload_tasks:
            await asyncio.gather(*preload_tasks, return_exceptions=True)
            logger.info(f"Preloaded {len(preload_tasks)} popular retrievers")
    
    async def _cleanup_loop(self) -> None:
        """Background loop for cleanup operations."""
        logger.info("Starting cleanup loop")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up expired retrievers
                await self._cleanup_expired_retrievers()
                
                # Clean up unused locks
                await self._cleanup_unused_locks()
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_expired_retrievers(self) -> None:
        """Clean up expired retrievers."""
        current_time = datetime.now()
        expired_keys = []
        
        async with self._manager_lock:
            for cache_key, retriever_info in list(self._retrievers.items()):
                # Check if retriever is too old or unused
                age = (current_time - retriever_info.metrics.creation_time).total_seconds()
                last_used_age = None
                
                if retriever_info.metrics.last_used:
                    last_used_age = (current_time - retriever_info.metrics.last_used).total_seconds()
                
                # Mark for cleanup if too old or unused
                if (age > settings.RETRIEVER_MAX_AGE or 
                    (last_used_age and last_used_age > settings.RETRIEVER_UNUSED_TIMEOUT)):
                    expired_keys.append(cache_key)
        
        # Remove expired retrievers
        for cache_key in expired_keys:
            await self._retriever_cache.remove(cache_key)
            async with self._manager_lock:
                self._retrievers.pop(cache_key, None)
            
            logger.info(f"Cleaned up expired retriever: {cache_key}")
    
    async def _cleanup_unused_locks(self) -> None:
        """Clean up unused locks to prevent memory leaks."""
        # Get active cache keys
        active_keys = set()
        async with self._manager_lock:
            active_keys = set(self._retrievers.keys())
        
        # Remove locks for keys that no longer exist
        unused_locks = set(self._retriever_locks.keys()) - active_keys
        for lock_key in unused_locks:
            del self._retriever_locks[lock_key]
        
        if unused_locks:
            logger.debug(f"Cleaned up {len(unused_locks)} unused locks")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        cache_stats = self._retriever_cache.get_stats()
        
        async with self._manager_lock:
            retriever_stats = {
                cache_key: {
                    "status": info.status.value,
                    "metrics": {
                        "total_requests": info.metrics.total_requests,
                        "success_rate": (info.metrics.successful_requests / info.metrics.total_requests 
                                       if info.metrics.total_requests > 0 else 0),
                        "avg_response_time": info.metrics.avg_response_time,
                        "error_rate": info.metrics.error_rate,
                        "uptime": info.metrics.get_uptime(),
                        "throughput": info.metrics.calculate_throughput()
                    },
                    "config": {
                        "collection": info.collection_name,
                        "top_k": info.top_k,
                        "priority": info.priority,
                        "tags": list(info.tags)
                    }
                }
                for cache_key, info in self._retrievers.items()
            }
        
        return {
            "global_metrics": self._global_metrics,
            "cache_stats": cache_stats,
            "retriever_stats": retriever_stats,
            "load_balancer_weights": dict(self._load_balancer_weights),
            "background_tasks": len(self._background_tasks),
            "usage_patterns": {
                key: len(timestamps) for key, timestamps in self._usage_stats.items()
            }
        }
    
    async def cleanup(self) -> None:
        """Clean up resources during shutdown."""
        logger.info("Cleaning up RetrieverManager...")
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Clear all caches and data
        async with self._manager_lock:
            self._retrievers.clear()
        
        self._usage_stats.clear()
        self._load_balancer_weights.clear()
        self._round_robin_counters.clear()
        self._retriever_locks.clear()
        
        logger.info("RetrieverManager cleanup completed")


# Global instance
# Global instance will be initialized with proper dependency injection
retriever_manager = None