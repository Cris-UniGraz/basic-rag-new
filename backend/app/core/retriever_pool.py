import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import weakref
from loguru import logger

from app.core.config import settings
from app.core.async_metadata_processor import async_metadata_processor


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for retriever pools."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME = "response_time"
    ADAPTIVE = "adaptive"


class PooledRetrieverStatus(Enum):
    """Status of a pooled retriever instance."""
    AVAILABLE = "available"
    BUSY = "busy"
    WARMING_UP = "warming_up"
    COOLING_DOWN = "cooling_down"
    FAILED = "failed"
    DRAINING = "draining"


@dataclass
class PooledRetriever:
    """A retriever instance within a pool."""
    instance_id: str
    retriever_instance: Any
    status: PooledRetrieverStatus = PooledRetrieverStatus.AVAILABLE
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    total_requests: int = 0
    active_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    circuit_breaker_failures: int = 0
    weight: float = 1.0
    
    def update_success(self, response_time: float):
        """Update metrics after successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.last_used = datetime.now()
        self.active_requests = max(0, self.active_requests - 1)
        
        # Update average response time
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * 0.9) + (response_time * 0.1)
        
        # Reset circuit breaker failures on success
        self.circuit_breaker_failures = 0
        
        # Update weight based on performance
        self._update_weight()
    
    def update_failure(self):
        """Update metrics after failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_used = datetime.now()
        self.active_requests = max(0, self.active_requests - 1)
        self.circuit_breaker_failures += 1
        
        # Update weight based on performance
        self._update_weight()
    
    def _update_weight(self):
        """Update weight based on performance metrics."""
        if self.total_requests == 0:
            self.weight = 1.0
            return
        
        # Base weight on success rate
        success_rate = self.successful_requests / self.total_requests
        
        # Penalty for high response time
        response_time_factor = max(0.1, 1.0 - (self.avg_response_time / 10.0))
        
        # Penalty for circuit breaker failures
        failure_factor = max(0.1, 1.0 - (self.circuit_breaker_failures * 0.2))
        
        self.weight = success_rate * response_time_factor * failure_factor
    
    def get_error_rate(self) -> float:
        """Get current error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def is_healthy(self) -> bool:
        """Check if retriever is healthy."""
        return (
            self.status not in [PooledRetrieverStatus.FAILED, PooledRetrieverStatus.DRAINING] and
            self.get_error_rate() < settings.RETRIEVER_ERROR_THRESHOLD and
            self.circuit_breaker_failures < settings.CIRCUIT_BREAKER_THRESHOLD
        )
    
    def can_handle_request(self) -> bool:
        """Check if retriever can handle a new request."""
        return (
            self.status == PooledRetrieverStatus.AVAILABLE and
            self.active_requests < settings.MAX_CONCURRENT_REQUESTS_PER_RETRIEVER and
            self.is_healthy()
        )


class RetrieverPool:
    """
    Thread-safe pool of retriever instances with advanced load balancing.
    
    Features:
    - Multiple load balancing strategies
    - Circuit breaker integration per instance
    - Automatic scaling based on demand
    - Health monitoring and auto-recovery
    - Performance-based routing
    - Intelligent load_balancing
    - Request queuing with priority
    - Graceful degradation
    """
    
    def __init__(
        self,
        pool_id: str,
        collection_name: str,
        top_k: int,
        min_size: int = 1,
        max_size: int = 5,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    ):
        """Initialize the retriever pool."""
        self.pool_id = pool_id
        self.collection_name = collection_name
        self.top_k = top_k
        self.min_size = min_size
        self.max_size = max_size
        self.strategy = strategy
        
        # Pool instances with request_queuing
        self._instances: Dict[str, PooledRetriever] = {}
        self._available_queue = asyncio.Queue()
        self._request_queue = asyncio.Queue()
        
        # Load balancing state
        self._round_robin_index = 0
        self._adaptive_weights: Dict[str, float] = {}
        
        # Pool state
        self._pool_lock = asyncio.Lock()
        self._scaling_lock = asyncio.Lock()
        self._current_size = 0
        self._target_size = min_size
        self._last_scale_event = datetime.now()
        
        # Metrics
        self._pool_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "queue_overflows": 0,
            "scaling_events": 0,
            "circuit_breaker_trips": 0
        }
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"RetrieverPool created: {pool_id} (min={min_size}, max={max_size})")
    
    async def initialize(self) -> None:
        """Initialize the pool with minimum instances."""
        logger.info(f"Initializing RetrieverPool: {self.pool_id}")
        
        try:
            # Create minimum instances
            await self._ensure_minimum_instances()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info(f"RetrieverPool initialized: {self.pool_id} with {len(self._instances)} instances")
            
        except Exception as e:
            logger.error(f"Failed to initialize RetrieverPool {self.pool_id}: {e}")
            raise RuntimeError(f"Pool initialization failed: {str(e)}")
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Auto-scaling task
        scaling_task = asyncio.create_task(self._auto_scaling_loop())
        self._background_tasks.add(scaling_task)
        scaling_task.add_done_callback(self._background_tasks.discard)
        
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self._background_tasks.add(health_task)
        health_task.add_done_callback(self._background_tasks.discard)
        
        # Request processing task
        processor_task = asyncio.create_task(self._request_processor_loop())
        self._background_tasks.add(processor_task)
        processor_task.add_done_callback(self._background_tasks.discard)
    
    async def get_retriever(self, timeout: float = 30.0) -> Optional[PooledRetriever]:
        """
        Get an available retriever from the pool with load balancing.
        
        Args:
            timeout: Maximum time to wait for an available retriever
            
        Returns:
            PooledRetriever instance or None if timeout
        """
        start_time = time.time()
        self._pool_metrics["total_requests"] += 1
        
        # Try to get retriever based on strategy
        retriever = await self._select_retriever_by_strategy()
        
        if retriever:
            # Mark as busy
            retriever.status = PooledRetrieverStatus.BUSY
            retriever.active_requests += 1
            self._pool_metrics["successful_requests"] += 1
            
            logger.debug(f"Retrieved instance {retriever.instance_id} from pool {self.pool_id}")
            return retriever
        
        # No immediate retriever available, try scaling or queue
        if await self._can_scale_up():
            # Try to scale up
            new_retriever = await self._create_instance()
            if new_retriever:
                new_retriever.status = PooledRetrieverStatus.BUSY
                new_retriever.active_requests += 1
                self._pool_metrics["successful_requests"] += 1
                self._pool_metrics["scaling_events"] += 1
                return new_retriever
        
        # Queue the request with timeout
        try:
            request_future = asyncio.Future()
            await asyncio.wait_for(
                self._request_queue.put((request_future, start_time)),
                timeout=1.0  # Quick timeout for queuing
            )
            
            # Wait for retriever to become available
            retriever = await asyncio.wait_for(request_future, timeout=timeout)
            
            if retriever:
                retriever.status = PooledRetrieverStatus.BUSY
                retriever.active_requests += 1
                self._pool_metrics["successful_requests"] += 1
                return retriever
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for retriever in pool {self.pool_id}")
        except asyncio.QueueFull:
            self._pool_metrics["queue_overflows"] += 1
            logger.warning(f"Request queue full for pool {self.pool_id}")
        
        self._pool_metrics["failed_requests"] += 1
        return None
    
    async def _select_retriever_by_strategy(self) -> Optional[PooledRetriever]:
        """Select retriever based on load balancing strategy."""
        async with self._pool_lock:
            available_instances = [
                instance for instance in self._instances.values()
                if instance.can_handle_request()
            ]
            
            if not available_instances:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(available_instances)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(available_instances)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(available_instances)
            elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
                return self._response_time_selection(available_instances)
            elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
                return self._adaptive_selection(available_instances)
            else:
                return available_instances[0]  # Fallback
    
    def _round_robin_selection(self, instances: List[PooledRetriever]) -> PooledRetriever:
        """Round-robin selection."""
        if not instances:
            return None
        
        self._round_robin_index = (self._round_robin_index + 1) % len(instances)
        return instances[self._round_robin_index]
    
    def _weighted_round_robin_selection(self, instances: List[PooledRetriever]) -> PooledRetriever:
        """Weighted round-robin selection based on instance weights."""
        if not instances:
            return None
        
        # Calculate cumulative weights
        total_weight = sum(instance.weight for instance in instances)
        if total_weight == 0:
            return self._round_robin_selection(instances)
        
        # Select based on weight
        target = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for instance in instances:
            cumulative_weight += instance.weight
            if cumulative_weight >= target:
                return instance
        
        return instances[-1]  # Fallback
    
    def _least_connections_selection(self, instances: List[PooledRetriever]) -> PooledRetriever:
        """Select instance with least active connections."""
        return min(instances, key=lambda x: x.active_requests)
    
    def _response_time_selection(self, instances: List[PooledRetriever]) -> PooledRetriever:
        """Select instance with best average response time."""
        return min(instances, key=lambda x: x.avg_response_time or float('inf'))
    
    def _adaptive_selection(self, instances: List[PooledRetriever]) -> PooledRetriever:
        """Adaptive selection based on multiple factors."""
        if not instances:
            return None
        
        # Score each instance based on multiple factors
        best_instance = None
        best_score = float('-inf')
        
        for instance in instances:
            # Calculate composite score
            connection_score = 1.0 / (instance.active_requests + 1)
            response_time_score = 1.0 / (instance.avg_response_time + 0.1)
            weight_score = instance.weight
            
            composite_score = (connection_score * 0.4 + 
                             response_time_score * 0.3 + 
                             weight_score * 0.3)
            
            if composite_score > best_score:
                best_score = composite_score
                best_instance = instance
        
        return best_instance
    
    async def return_retriever(self, retriever: PooledRetriever, success: bool = True, response_time: float = 0.0):
        """Return a retriever to the pool after use."""
        try:
            # Update metrics
            if success:
                retriever.update_success(response_time)
            else:
                retriever.update_failure()
            
            # Return to available state
            retriever.status = PooledRetrieverStatus.AVAILABLE
            
            # Process queued requests
            await self._process_queued_request(retriever)
            
        except Exception as e:
            logger.error(f"Error returning retriever to pool {self.pool_id}: {e}")
    
    async def _process_queued_request(self, retriever: PooledRetriever):
        """Process a queued request with the returned retriever."""
        try:
            if not self._request_queue.empty():
                request_future, start_time = await asyncio.wait_for(
                    self._request_queue.get(),
                    timeout=0.1
                )
                
                if not request_future.done():
                    request_future.set_result(retriever)
                    logger.debug(f"Processed queued request in pool {self.pool_id}")
                
        except asyncio.TimeoutError:
            pass  # No queued requests
        except Exception as e:
            logger.error(f"Error processing queued request: {e}")
    
    async def _can_scale_up(self) -> bool:
        """Check if pool can scale up."""
        async with self._scaling_lock:
            return (
                len(self._instances) < self.max_size and
                (datetime.now() - self._last_scale_event).seconds > settings.POOL_SCALING_COOLDOWN
            )
    
    async def _create_instance(self) -> Optional[PooledRetriever]:
        """Create a new retriever instance."""
        try:
            # Import here to avoid circular imports
            from app.core.retriever_manager import retriever_manager
            
            # Create retriever instance
            retriever_instance = await retriever_manager._create_retriever_instance(
                self.collection_name, self.top_k
            )
            
            if retriever_instance:
                instance_id = f"{self.pool_id}_{len(self._instances)}"
                
                pooled_retriever = PooledRetriever(
                    instance_id=instance_id,
                    retriever_instance=retriever_instance,
                    status=PooledRetrieverStatus.AVAILABLE
                )
                
                async with self._pool_lock:
                    self._instances[instance_id] = pooled_retriever
                
                self._current_size = len(self._instances)
                self._last_scale_event = datetime.now()
                
                logger.info(f"Created new instance {instance_id} for pool {self.pool_id}")
                
                # Log async metrics
                async_metadata_processor.record_performance_async(
                    "pool_instance_creation",
                    0.0,
                    True,
                    {
                        "pool_id": self.pool_id,
                        "collection": self.collection_name,
                        "pool_size": self._current_size
                    }
                )
                
                return pooled_retriever
            
        except Exception as e:
            logger.error(f"Error creating instance for pool {self.pool_id}: {e}")
        
        return None
    
    async def _ensure_minimum_instances(self) -> None:
        """Ensure pool has minimum number of instances."""
        while len(self._instances) < self.min_size:
            instance = await self._create_instance()
            if not instance:
                logger.error(f"Failed to create minimum instances for pool {self.pool_id}")
                break
    
    async def _auto_scaling_loop(self) -> None:
        """Background loop for auto-scaling."""
        logger.info(f"Starting auto-scaling loop for pool {self.pool_id}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(settings.POOL_SCALING_CHECK_INTERVAL)
                
                # Analyze load and decide on scaling
                await self._analyze_and_scale()
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop for pool {self.pool_id}: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_and_scale(self) -> None:
        """Analyze current load and scale if necessary."""
        async with self._pool_lock:
            total_instances = len(self._instances)
            busy_instances = sum(1 for instance in self._instances.values() 
                               if instance.status == PooledRetrieverStatus.BUSY)
            queue_size = self._request_queue.qsize()
            
            # Calculate utilization
            utilization = busy_instances / total_instances if total_instances > 0 else 0
            
            # Scale up if high utilization or queue backlog
            if (utilization > settings.POOL_SCALE_UP_THRESHOLD or queue_size > settings.POOL_QUEUE_THRESHOLD):
                if await self._can_scale_up():
                    await self._create_instance()
                    self._pool_metrics["scaling_events"] += 1
                    logger.info(f"Scaled up pool {self.pool_id} to {len(self._instances)} instances")
            
            # Scale down if low utilization
            elif (utilization < settings.POOL_SCALE_DOWN_THRESHOLD and 
                  total_instances > self.min_size and
                  queue_size == 0):
                await self._scale_down()
    
    async def _scale_down(self) -> None:
        """Scale down the pool by removing an instance."""
        async with self._scaling_lock:
            if len(self._instances) <= self.min_size:
                return
            
            # Find least used instance to remove
            candidate = None
            min_requests = float('inf')
            
            for instance in self._instances.values():
                if (instance.status == PooledRetrieverStatus.AVAILABLE and
                    instance.active_requests == 0 and
                    instance.total_requests < min_requests):
                    candidate = instance
                    min_requests = instance.total_requests
            
            if candidate:
                # Mark for draining
                candidate.status = PooledRetrieverStatus.DRAINING
                
                # Remove after short delay
                await asyncio.sleep(5)
                
                async with self._pool_lock:
                    if candidate.instance_id in self._instances:
                        del self._instances[candidate.instance_id]
                
                self._current_size = len(self._instances)
                self._last_scale_event = datetime.now()
                self._pool_metrics["scaling_events"] += 1
                
                logger.info(f"Scaled down pool {self.pool_id} to {len(self._instances)} instances")
    
    async def _health_monitoring_loop(self) -> None:
        """Background loop for health monitoring."""
        logger.info(f"Starting health monitoring loop for pool {self.pool_id}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL)
                
                # Check health of all instances
                await self._check_instance_health()
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop for pool {self.pool_id}: {e}")
                await asyncio.sleep(60)
    
    async def _check_instance_health(self) -> None:
        """Check health of all pool instances."""
        unhealthy_instances = []
        
        async with self._pool_lock:
            for instance in self._instances.values():
                if not instance.is_healthy():
                    unhealthy_instances.append(instance.instance_id)
                    instance.status = PooledRetrieverStatus.FAILED
                    self._pool_metrics["circuit_breaker_trips"] += 1
        
        # Replace unhealthy instances
        for instance_id in unhealthy_instances:
            await self._replace_instance(instance_id)
    
    async def _replace_instance(self, instance_id: str) -> None:
        """Replace an unhealthy instance."""
        try:
            async with self._pool_lock:
                if instance_id in self._instances:
                    del self._instances[instance_id]
            
            # Create replacement
            replacement = await self._create_instance()
            if replacement:
                logger.info(f"Replaced unhealthy instance {instance_id} in pool {self.pool_id}")
            else:
                logger.error(f"Failed to replace instance {instance_id} in pool {self.pool_id}")
                
        except Exception as e:
            logger.error(f"Error replacing instance {instance_id}: {e}")
    
    async def _request_processor_loop(self) -> None:
        """Background loop for processing queued requests."""
        logger.info(f"Starting request processor loop for pool {self.pool_id}")
        
        while not self._shutdown_event.is_set():
            try:
                # Process requests continuously
                await self._process_request_queue()
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in request processor loop for pool {self.pool_id}: {e}")
                await asyncio.sleep(1)
    
    async def _process_request_queue(self) -> None:
        """Process queued requests."""
        if self._request_queue.empty():
            return
        
        # Find available retriever
        available_retriever = await self._select_retriever_by_strategy()
        
        if available_retriever and not self._request_queue.empty():
            try:
                request_future, start_time = await asyncio.wait_for(
                    self._request_queue.get(),
                    timeout=0.1
                )
                
                if not request_future.done():
                    request_future.set_result(available_retriever)
                    
            except asyncio.TimeoutError:
                pass  # No requests to process
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        instance_stats = {}
        total_active_requests = 0
        
        for instance_id, instance in self._instances.items():
            total_active_requests += instance.active_requests
            instance_stats[instance_id] = {
                "status": instance.status.value,
                "active_requests": instance.active_requests,
                "total_requests": instance.total_requests,
                "success_rate": (instance.successful_requests / instance.total_requests 
                               if instance.total_requests > 0 else 0),
                "avg_response_time": instance.avg_response_time,
                "weight": instance.weight,
                "error_rate": instance.get_error_rate(),
                "uptime": (datetime.now() - instance.created_at).total_seconds()
            }
        
        return {
            "pool_id": self.pool_id,
            "collection_name": self.collection_name,
            "top_k": self.top_k,
            "strategy": self.strategy.value,
            "size_config": {
                "min_size": self.min_size,
                "max_size": self.max_size,
                "current_size": len(self._instances)
            },
            "metrics": self._pool_metrics,
            "queue_size": self._request_queue.qsize(),
            "total_active_requests": total_active_requests,
            "instance_stats": instance_stats
        }
    
    async def cleanup(self) -> None:
        """Clean up pool resources."""
        logger.info(f"Cleaning up RetrieverPool: {self.pool_id}")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Clear instances
        async with self._pool_lock:
            self._instances.clear()
        
        logger.info(f"RetrieverPool cleanup completed: {self.pool_id}")


class RetrieverPoolManager:
    """
    Manager for multiple retriever pools with global optimization.
    
    Features:
    - Pool lifecycle management
    - Global load balancing across pools
    - Resource optimization
    - Cross-pool metrics aggregation
    """
    
    def __init__(self):
        """Initialize the pool manager."""
        self._pools: Dict[str, RetrieverPool] = {}
        self._manager_lock = asyncio.Lock()
        self._global_metrics = {
            "total_pools": 0,
            "total_instances": 0,
            "total_requests": 0
        }
        
        logger.info("RetrieverPoolManager initialized")
    
    async def get_or_create_pool(
        self,
        collection_name: str,
        top_k: int,
        min_size: int = 1,
        max_size: int = 5,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    ) -> RetrieverPool:
        """Get existing pool or create new one."""
        pool_id = f"{collection_name}_{top_k}"
        
        async with self._manager_lock:
            if pool_id not in self._pools:
                pool = RetrieverPool(
                    pool_id=pool_id,
                    collection_name=collection_name,
                    top_k=top_k,
                    min_size=min_size,
                    max_size=max_size,
                    strategy=strategy
                )
                
                await pool.initialize()
                self._pools[pool_id] = pool
                self._global_metrics["total_pools"] += 1
                
                logger.info(f"Created new pool: {pool_id}")
            
            return self._pools[pool_id]
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics across all pools."""
        pool_stats = {}
        total_instances = 0
        total_requests = 0
        
        async with self._manager_lock:
            for pool_id, pool in self._pools.items():
                stats = pool.get_stats()
                pool_stats[pool_id] = stats
                total_instances += stats["size_config"]["current_size"]
                total_requests += stats["metrics"]["total_requests"]
        
        self._global_metrics["total_instances"] = total_instances
        self._global_metrics["total_requests"] = total_requests
        
        return {
            "global_metrics": self._global_metrics,
            "pool_stats": pool_stats
        }
    
    async def cleanup(self) -> None:
        """Clean up all pools."""
        logger.info("Cleaning up RetrieverPoolManager...")
        
        cleanup_tasks = []
        async with self._manager_lock:
            for pool in self._pools.values():
                cleanup_tasks.append(pool.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        async with self._manager_lock:
            self._pools.clear()
        
        logger.info("RetrieverPoolManager cleanup completed")


# Global instance
retriever_pool_manager = RetrieverPoolManager()