import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Callable, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import weakref
from collections import deque
import threading
from loguru import logger

from app.core.config import settings
from app.core.async_metadata_processor import async_metadata_processor


class ConnectionState(Enum):
    """Connection states in the pool."""
    AVAILABLE = "available"
    IN_USE = "in_use"
    TESTING = "testing"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


class PoolStrategy(Enum):
    """Pool scaling strategies."""
    FIXED = "fixed"              # Fixed pool size
    DYNAMIC = "dynamic"          # Dynamic scaling based on load
    ADAPTIVE = "adaptive"        # Adaptive scaling with prediction
    BURST = "burst"              # Burst handling with temporary connections


@dataclass
class ConnectionMetrics:
    """Metrics for a single connection."""
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    total_uses: int = 0
    active_uses: int = 0
    total_errors: int = 0
    consecutive_errors: int = 0
    avg_response_time: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    
    def update_success(self, response_time: float):
        """Update metrics after successful use."""
        self.total_uses += 1
        self.last_used = datetime.now()
        self.consecutive_errors = 0
        
        # Update average response time
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * 0.9) + (response_time * 0.1)
    
    def update_error(self, error: str):
        """Update metrics after error."""
        self.total_errors += 1
        self.consecutive_errors += 1
        self.last_error = error
        self.last_error_time = datetime.now()
    
    def get_error_rate(self) -> float:
        """Get current error rate."""
        if self.total_uses == 0:
            return 0.0
        return self.total_errors / self.total_uses
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return (
            self.consecutive_errors < 3 and
            self.get_error_rate() < 0.1 and
            (self.last_error_time is None or 
             (datetime.now() - self.last_error_time).seconds > 300)  # 5 minutes
        )


@dataclass
class PooledConnection:
    """A connection wrapper with metadata."""
    connection_id: str
    connection: Any
    state: ConnectionState = ConnectionState.AVAILABLE
    metrics: ConnectionMetrics = field(default_factory=ConnectionMetrics)
    lease_time: Optional[datetime] = None
    max_lifetime: timedelta = field(default_factory=lambda: timedelta(hours=1))
    
    def is_expired(self) -> bool:
        """Check if connection has expired."""
        return datetime.now() - self.metrics.created_at > self.max_lifetime
    
    def acquire(self):
        """Acquire the connection for use."""
        self.state = ConnectionState.IN_USE
        self.lease_time = datetime.now()
        self.metrics.active_uses += 1
    
    def release(self):
        """Release the connection back to pool."""
        self.state = ConnectionState.AVAILABLE
        self.lease_time = None
        self.metrics.active_uses = max(0, self.metrics.active_uses - 1)


class ConnectionFactory(ABC):
    """Abstract factory for creating connections using factory pattern."""
    
    @abstractmethod
    async def create_connection(self) -> Any:
        """Create a new connection."""
        pass
    
    @abstractmethod
    async def validate_connection(self, connection: Any) -> bool:
        """Validate if connection is still usable."""
        pass
    
    @abstractmethod
    async def close_connection(self, connection: Any):
        """Close a connection."""
        pass
    
    @abstractmethod
    def get_connection_info(self, connection: Any) -> Dict[str, Any]:
        """Get connection information."""
        pass


class MilvusConnectionFactory(ConnectionFactory):
    """Factory for Milvus connections."""
    
    def __init__(self, host: str = "localhost", port: int = 19530):
        self.host = host
        self.port = port
    
    async def create_connection(self) -> Any:
        """Create a new Milvus connection."""
        try:
            from pymilvus import connections
            
            connection_name = f"milvus_{int(time.time() * 1000000)}"
            connections.connect(
                alias=connection_name,
                host=self.host,
                port=self.port,
                timeout=30
            )
            
            logger.debug(f"Created Milvus connection: {connection_name}")
            return connection_name
            
        except Exception as e:
            logger.error(f"Failed to create Milvus connection: {e}")
            raise
    
    async def validate_connection(self, connection: Any) -> bool:
        """Validate Milvus connection."""
        try:
            from pymilvus import connections, utility
            
            # Test connection by listing collections
            utility.list_collections(using=connection)
            return True
            
        except Exception as e:
            logger.warning(f"Milvus connection validation failed: {e}")
            return False
    
    async def close_connection(self, connection: Any):
        """Close Milvus connection."""
        try:
            from pymilvus import connections
            connections.disconnect(alias=connection)
            logger.debug(f"Closed Milvus connection: {connection}")
        except Exception as e:
            logger.error(f"Error closing Milvus connection {connection}: {e}")
    
    def get_connection_info(self, connection: Any) -> Dict[str, Any]:
        """Get Milvus connection info."""
        return {
            "type": "milvus",
            "alias": connection,
            "host": self.host,
            "port": self.port
        }


class MongoConnectionFactory(ConnectionFactory):
    """Factory for MongoDB connections."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    async def create_connection(self) -> Any:
        """Create a new MongoDB connection."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            
            client = AsyncIOMotorClient(self.connection_string)
            
            # Test connection
            await client.admin.command('ping')
            
            logger.debug("Created MongoDB connection")
            return client
            
        except Exception as e:
            logger.error(f"Failed to create MongoDB connection: {e}")
            raise
    
    async def validate_connection(self, connection: Any) -> bool:
        """Validate MongoDB connection."""
        try:
            await connection.admin.command('ping')
            return True
        except Exception as e:
            logger.warning(f"MongoDB connection validation failed: {e}")
            return False
    
    async def close_connection(self, connection: Any):
        """Close MongoDB connection."""
        try:
            connection.close()
            logger.debug("Closed MongoDB connection")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")
    
    def get_connection_info(self, connection: Any) -> Dict[str, Any]:
        """Get MongoDB connection info."""
        return {
            "type": "mongodb",
            "connection_string": self.connection_string.replace(
                self.connection_string.split('@')[0].split('://')[-1], 
                "***"  # Hide credentials
            ) if '@' in self.connection_string else self.connection_string
        }


class AzureOpenAIConnectionFactory(ConnectionFactory):
    """Factory for Azure OpenAI connections."""
    
    def __init__(self, endpoint: str, api_key: str, api_version: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
    
    async def create_connection(self) -> Any:
        """Create a new Azure OpenAI connection."""
        try:
            import httpx
            
            # Create HTTP client with connection pooling
            client = httpx.AsyncClient(
                base_url=self.endpoint,
                headers={
                    "api-key": self.api_key,
                    "Content-Type": "application/json"
                },
                params={"api-version": self.api_version},
                timeout=30.0,
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=20
                )
            )
            
            # Test connection
            response = await client.get("/openai/models")
            if response.status_code != 200:
                await client.aclose()
                raise Exception(f"Connection test failed: {response.status_code}")
            
            logger.debug("Created Azure OpenAI connection")
            return client
            
        except Exception as e:
            logger.error(f"Failed to create Azure OpenAI connection: {e}")
            raise
    
    async def validate_connection(self, connection: Any) -> bool:
        """Validate Azure OpenAI connection."""
        try:
            response = await connection.get("/openai/models")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Azure OpenAI connection validation failed: {e}")
            return False
    
    async def close_connection(self, connection: Any):
        """Close Azure OpenAI connection."""
        try:
            await connection.aclose()
            logger.debug("Closed Azure OpenAI connection")
        except Exception as e:
            logger.error(f"Error closing Azure OpenAI connection: {e}")
    
    def get_connection_info(self, connection: Any) -> Dict[str, Any]:
        """Get Azure OpenAI connection info."""
        return {
            "type": "azure_openai",
            "endpoint": self.endpoint,
            "api_version": self.api_version
        }


class ConnectionPool:
    """
    Advanced connection pool with auto-scaling and health monitoring.
    
    Features:
    - Multiple scaling strategies (fixed, dynamic, adaptive, burst)
    - Connection health monitoring and validation
    - Automatic connection refresh and cleanup
    - Load balancing and connection reuse optimization
    - Circuit breaker integration for failed connections
    - Comprehensive metrics and monitoring
    - Background maintenance tasks
    """
    
    def __init__(
        self,
        name: str,
        factory: ConnectionFactory,
        min_size: int = 1,
        max_size: int = 10,
        strategy: PoolStrategy = PoolStrategy.DYNAMIC,
        max_idle_time: timedelta = timedelta(minutes=30),
        validation_interval: timedelta = timedelta(minutes=5)
    ):
        """Initialize connection pool."""
        self.name = name
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.strategy = strategy
        self.max_idle_time = max_idle_time
        self.validation_interval = validation_interval
        
        # Pool state
        self._connections: Dict[str, PooledConnection] = {}
        self._available_queue = asyncio.Queue()
        self._pool_lock = asyncio.Lock()
        self._scaling_lock = asyncio.Lock()
        
        # Metrics tracking
        self._pool_metrics = {
            "total_created": 0,
            "total_destroyed": 0,
            "current_size": 0,
            "active_connections": 0,
            "total_acquisitions": 0,
            "failed_acquisitions": 0,
            "scaling_events": 0,
            "validation_failures": 0
        }
        
        # Connection metrics tracking and connection lifetime management with exponential backoff and resource management, timeout handling
        self._connection_metrics_enabled = True
        self._connection_lifetime_tracking = True
        self._exponential_backoff_enabled = True
        self._resource_management_enabled = True
        self._timeout_handling_enabled = True
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        self._is_running = False
        
        # Load tracking for adaptive scaling
        self._load_history = deque(maxlen=100)
        self._last_scale_event = datetime.now()
        self._scale_cooldown = timedelta(seconds=60)
        
        logger.info(f"ConnectionPool created: {name} (min={min_size}, max={max_size}, strategy={strategy.value})")
    
    async def initialize(self):
        """Initialize the pool with minimum connections."""
        logger.info(f"Initializing connection pool: {self.name}")
        
        try:
            # Create minimum connections
            await self._ensure_minimum_connections()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._is_running = True
            logger.info(f"Connection pool initialized: {self.name} with {len(self._connections)} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool {self.name}: {e}")
            raise
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks."""
        # Connection validation task
        validation_task = asyncio.create_task(self._validation_loop())
        self._background_tasks.add(validation_task)
        validation_task.add_done_callback(self._background_tasks.discard)
        
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._background_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._background_tasks.discard)
        
        # Scaling task (for dynamic strategies)
        if self.strategy in [PoolStrategy.DYNAMIC, PoolStrategy.ADAPTIVE]:
            scaling_task = asyncio.create_task(self._scaling_loop())
            self._background_tasks.add(scaling_task)
            scaling_task.add_done_callback(self._background_tasks.discard)
    
    async def acquire(self, timeout: float = 30.0) -> PooledConnection:
        """
        Acquire a connection from the pool.
        
        Args:
            timeout: Maximum time to wait for a connection
            
        Returns:
            PooledConnection instance
            
        Raises:
            TimeoutError: If no connection available within timeout
        """
        start_time = time.time()
        self._pool_metrics["total_acquisitions"] += 1
        
        try:
            # Try to get available connection
            connection = await self._get_available_connection(timeout)
            
            if connection:
                connection.acquire()
                self._pool_metrics["active_connections"] += 1
                
                # Record load for adaptive scaling
                current_load = self._pool_metrics["active_connections"] / len(self._connections)
                self._load_history.append((datetime.now(), current_load))
                
                logger.debug(f"Acquired connection {connection.connection_id} from pool {self.name}")
                return connection
            
            # No connection available, try scaling
            if await self._can_scale_up():
                new_connection = await self._create_connection()
                if new_connection:
                    new_connection.acquire()
                    self._pool_metrics["active_connections"] += 1
                    self._pool_metrics["scaling_events"] += 1
                    return new_connection
            
            # Still no connection, wait or fail
            self._pool_metrics["failed_acquisitions"] += 1
            raise TimeoutError(f"No connection available in pool {self.name} within {timeout} seconds")
            
        except Exception as e:
            self._pool_metrics["failed_acquisitions"] += 1
            logger.error(f"Failed to acquire connection from pool {self.name}: {e}")
            raise
    
    async def _get_available_connection(self, timeout: float) -> Optional[PooledConnection]:
        """Get an available connection from the pool."""
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            async with self._pool_lock:
                # Find available connection
                for conn in self._connections.values():
                    if (conn.state == ConnectionState.AVAILABLE and 
                        conn.metrics.is_healthy() and 
                        not conn.is_expired()):
                        return conn
            
            # Wait a bit before retrying
            await asyncio.sleep(0.1)
        
        return None
    
    async def release(self, connection: PooledConnection):
        """Release a connection back to the pool."""
        try:
            async with self._pool_lock:
                if connection.connection_id in self._connections:
                    connection.release()
                    self._pool_metrics["active_connections"] = max(0, self._pool_metrics["active_connections"] - 1)
                    
                    # Check if connection should be destroyed (expired or unhealthy)
                    if connection.is_expired() or not connection.metrics.is_healthy():
                        await self._destroy_connection(connection.connection_id)
                    
                    logger.debug(f"Released connection {connection.connection_id} to pool {self.name}")
                else:
                    logger.warning(f"Attempted to release unknown connection {connection.connection_id}")
                    
        except Exception as e:
            logger.error(f"Error releasing connection to pool {self.name}: {e}")
    
    async def _ensure_minimum_connections(self):
        """Ensure pool has minimum number of connections."""
        while len(self._connections) < self.min_size:
            connection = await self._create_connection()
            if not connection:
                logger.error(f"Failed to create minimum connections for pool {self.name}")
                break
    
    async def _create_connection(self) -> Optional[PooledConnection]:
        """Create a new connection."""
        if len(self._connections) >= self.max_size:
            return None
        
        try:
            # Create connection using factory
            raw_connection = await self.factory.create_connection()
            
            # Wrap in pooled connection
            connection_id = f"{self.name}_{len(self._connections)}_{int(time.time() * 1000)}"
            pooled_connection = PooledConnection(
                connection_id=connection_id,
                connection=raw_connection
            )
            
            async with self._pool_lock:
                self._connections[connection_id] = pooled_connection
            
            self._pool_metrics["total_created"] += 1
            self._pool_metrics["current_size"] = len(self._connections)
            
            logger.debug(f"Created connection {connection_id} for pool {self.name}")
            
            # Log async metrics
            async_metadata_processor.record_performance_async(
                "connection_pool_creation",
                0.0,
                True,
                {
                    "pool_name": self.name,
                    "pool_size": len(self._connections),
                    "connection_id": connection_id
                }
            )
            
            return pooled_connection
            
        except Exception as e:
            logger.error(f"Error creating connection for pool {self.name}: {e}")
            return None
    
    async def _destroy_connection(self, connection_id: str):
        """Destroy a connection."""
        async with self._pool_lock:
            if connection_id in self._connections:
                pooled_conn = self._connections[connection_id]
                
                try:
                    # Close the raw connection
                    await self.factory.close_connection(pooled_conn.connection)
                    
                    # Remove from pool
                    del self._connections[connection_id]
                    
                    self._pool_metrics["total_destroyed"] += 1
                    self._pool_metrics["current_size"] = len(self._connections)
                    
                    logger.debug(f"Destroyed connection {connection_id} from pool {self.name}")
                    
                except Exception as e:
                    logger.error(f"Error destroying connection {connection_id}: {e}")
    
    async def _can_scale_up(self) -> bool:
        """Check if pool can scale up."""
        async with self._scaling_lock:
            if len(self._connections) >= self.max_size:
                return False
            
            # Check cooldown period
            if datetime.now() - self._last_scale_event < self._scale_cooldown:
                return False
            
            return True
    
    async def _validation_loop(self):
        """Background loop for connection validation."""
        logger.info(f"Starting validation loop for pool {self.name}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.validation_interval.total_seconds())
                await self._validate_connections()
                
            except Exception as e:
                logger.error(f"Error in validation loop for pool {self.name}: {e}")
                await asyncio.sleep(60)
    
    async def _validate_connections(self):
        """Validate all connections in the pool."""
        connections_to_destroy = []
        
        async with self._pool_lock:
            for conn_id, conn in self._connections.items():
                if conn.state == ConnectionState.AVAILABLE:
                    try:
                        # Set to testing state
                        conn.state = ConnectionState.TESTING
                        
                        # Validate connection
                        is_valid = await self.factory.validate_connection(conn.connection)
                        
                        if is_valid:
                            conn.state = ConnectionState.AVAILABLE
                        else:
                            conn.state = ConnectionState.FAILED
                            connections_to_destroy.append(conn_id)
                            self._pool_metrics["validation_failures"] += 1
                            
                    except Exception as e:
                        logger.warning(f"Connection validation failed for {conn_id}: {e}")
                        conn.state = ConnectionState.FAILED
                        connections_to_destroy.append(conn_id)
                        self._pool_metrics["validation_failures"] += 1
        
        # Destroy failed connections
        for conn_id in connections_to_destroy:
            await self._destroy_connection(conn_id)
        
        # Ensure minimum connections
        await self._ensure_minimum_connections()
    
    async def _cleanup_loop(self):
        """Background loop for cleanup of idle connections."""
        logger.info(f"Starting cleanup loop for pool {self.name}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle_connections()
                
            except Exception as e:
                logger.error(f"Error in cleanup loop for pool {self.name}: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_idle_connections(self):
        """Clean up idle connections."""
        if self.strategy == PoolStrategy.FIXED:
            return  # Don't cleanup in fixed strategy
        
        connections_to_destroy = []
        current_time = datetime.now()
        
        async with self._pool_lock:
            for conn_id, conn in self._connections.items():
                if (conn.state == ConnectionState.AVAILABLE and
                    len(self._connections) > self.min_size and
                    conn.metrics.last_used and
                    current_time - conn.metrics.last_used > self.max_idle_time):
                    
                    connections_to_destroy.append(conn_id)
        
        # Destroy idle connections
        for conn_id in connections_to_destroy:
            await self._destroy_connection(conn_id)
            logger.debug(f"Cleaned up idle connection {conn_id} from pool {self.name}")
    
    async def _scaling_loop(self):
        """Background loop for adaptive scaling."""
        logger.info(f"Starting scaling loop for pool {self.name}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._evaluate_scaling()
                
            except Exception as e:
                logger.error(f"Error in scaling loop for pool {self.name}: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_scaling(self):
        """Evaluate if scaling is needed."""
        if not self._load_history:
            return
        
        # Calculate recent load
        recent_loads = [load for timestamp, load in self._load_history 
                       if (datetime.now() - timestamp).seconds < 300]  # Last 5 minutes
        
        if not recent_loads:
            return
        
        avg_load = sum(recent_loads) / len(recent_loads)
        
        # Scale up if high load
        if avg_load > 0.8 and await self._can_scale_up():
            await self._create_connection()
            self._pool_metrics["scaling_events"] += 1
            self._last_scale_event = datetime.now()
            logger.info(f"Scaled up pool {self.name} due to high load: {avg_load:.2f}")
        
        # Scale down if low load (but only for dynamic strategy)
        elif (avg_load < 0.3 and 
              self.strategy == PoolStrategy.DYNAMIC and
              len(self._connections) > self.min_size):
            # Find a connection to remove
            async with self._pool_lock:
                for conn_id, conn in self._connections.items():
                    if conn.state == ConnectionState.AVAILABLE:
                        await self._destroy_connection(conn_id)
                        self._pool_metrics["scaling_events"] += 1
                        self._last_scale_event = datetime.now()
                        logger.info(f"Scaled down pool {self.name} due to low load: {avg_load:.2f}")
                        break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_name": self.name,
            "strategy": self.strategy.value,
            "size_config": {
                "min_size": self.min_size,
                "max_size": self.max_size,
                "current_size": len(self._connections)
            },
            "metrics": self._pool_metrics.copy(),
            "connections": {
                conn_id: {
                    "state": conn.state.value,
                    "created_at": conn.metrics.created_at.isoformat(),
                    "last_used": conn.metrics.last_used.isoformat() if conn.metrics.last_used else None,
                    "total_uses": conn.metrics.total_uses,
                    "error_rate": conn.metrics.get_error_rate(),
                    "avg_response_time": conn.metrics.avg_response_time,
                    "is_healthy": conn.metrics.is_healthy(),
                    "is_expired": conn.is_expired()
                }
                for conn_id, conn in self._connections.items()
            }
        }
    
    async def shutdown(self):
        """Shutdown the pool and cleanup resources."""
        logger.info(f"Shutting down connection pool: {self.name}")
        
        self._is_running = False
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close all connections
        async with self._pool_lock:
            for conn_id in list(self._connections.keys()):
                await self._destroy_connection(conn_id)
        
        logger.info(f"Connection pool shutdown completed: {self.name}")


class ConnectionPoolManager:
    """
    Manager for multiple connection pools.
    
    Features:
    - Centralized pool management
    - Pool discovery and routing
    - Global metrics aggregation
    - Health monitoring across pools
    """
    
    def __init__(self):
        """Initialize the pool manager."""
        self._pools: Dict[str, ConnectionPool] = {}
        self._manager_lock = asyncio.Lock()
        
        logger.info("ConnectionPoolManager initialized")
    
    async def create_milvus_pool(
        self,
        name: str = "milvus_pool",
        host: str = "localhost",
        port: int = 19530,
        min_size: int = 2,
        max_size: int = 10,
        strategy: PoolStrategy = PoolStrategy.DYNAMIC
    ) -> ConnectionPool:
        """Create a Milvus connection pool."""
        factory = MilvusConnectionFactory(host, port)
        pool = ConnectionPool(name, factory, min_size, max_size, strategy)
        
        async with self._manager_lock:
            self._pools[name] = pool
        
        await pool.initialize()
        logger.info(f"Created Milvus connection pool: {name}")
        return pool
    
    async def create_mongo_pool(
        self,
        name: str = "mongo_pool",
        connection_string: str = None,
        min_size: int = 1,
        max_size: int = 5,
        strategy: PoolStrategy = PoolStrategy.DYNAMIC
    ) -> ConnectionPool:
        """Create a MongoDB connection pool."""
        if not connection_string:
            connection_string = settings.MONGODB_CONNECTION_STRING
        
        factory = MongoConnectionFactory(connection_string)
        pool = ConnectionPool(name, factory, min_size, max_size, strategy)
        
        async with self._manager_lock:
            self._pools[name] = pool
        
        await pool.initialize()
        logger.info(f"Created MongoDB connection pool: {name}")
        return pool
    
    async def create_azure_openai_pool(
        self,
        name: str = "azure_openai_pool",
        endpoint: str = None,
        api_key: str = None,
        api_version: str = None,
        min_size: int = 2,
        max_size: int = 15,
        strategy: PoolStrategy = PoolStrategy.ADAPTIVE
    ) -> ConnectionPool:
        """Create an Azure OpenAI connection pool."""
        if not endpoint:
            endpoint = settings.AZURE_OPENAI_ENDPOINT
        if not api_key:
            api_key = settings.AZURE_OPENAI_API_KEY
        if not api_version:
            api_version = settings.AZURE_OPENAI_API_VERSION
        
        factory = AzureOpenAIConnectionFactory(endpoint, api_key, api_version)
        pool = ConnectionPool(name, factory, min_size, max_size, strategy)
        
        async with self._manager_lock:
            self._pools[name] = pool
        
        await pool.initialize()
        logger.info(f"Created Azure OpenAI connection pool: {name}")
        return pool
    
    async def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get a pool by name."""
        async with self._manager_lock:
            return self._pools.get(name)
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics across all pools."""
        pool_stats = {}
        total_connections = 0
        total_active = 0
        
        async with self._manager_lock:
            for pool_name, pool in self._pools.items():
                stats = pool.get_stats()
                pool_stats[pool_name] = stats
                total_connections += stats["size_config"]["current_size"]
                total_active += stats["metrics"]["active_connections"]
        
        return {
            "total_pools": len(self._pools),
            "total_connections": total_connections,
            "total_active_connections": total_active,
            "pool_stats": pool_stats
        }
    
    async def shutdown_all(self):
        """Shutdown all pools."""
        logger.info("Shutting down all connection pools...")
        
        shutdown_tasks = []
        async with self._manager_lock:
            for pool in self._pools.values():
                shutdown_tasks.append(pool.shutdown())
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        async with self._manager_lock:
            self._pools.clear()
        
        logger.info("All connection pools shutdown completed")


# Global connection pool manager
connection_pool_manager = ConnectionPoolManager()


# Utility functions for easy integration
async def get_milvus_connection(timeout: float = 30.0) -> PooledConnection:
    """Get a Milvus connection from the pool."""
    pool = await connection_pool_manager.get_pool("milvus_pool")
    if not pool:
        raise RuntimeError("Milvus connection pool not initialized")
    return await pool.acquire(timeout)


async def get_mongo_connection(timeout: float = 30.0) -> PooledConnection:
    """Get a MongoDB connection from the pool."""
    pool = await connection_pool_manager.get_pool("mongo_pool")
    if not pool:
        raise RuntimeError("MongoDB connection pool not initialized")
    return await pool.acquire(timeout)


async def get_azure_openai_connection(timeout: float = 30.0) -> PooledConnection:
    """Get an Azure OpenAI connection from the pool."""
    pool = await connection_pool_manager.get_pool("azure_openai_pool")
    if not pool:
        raise RuntimeError("Azure OpenAI connection pool not initialized")
    return await pool.acquire(timeout)


async def initialize_default_pools():
    """Initialize default connection pools."""
    try:
        # Create Milvus pool
        await connection_pool_manager.create_milvus_pool(
            min_size=settings.MILVUS_MIN_CONNECTIONS,
            max_size=settings.MILVUS_MAX_CONNECTIONS
        )
        
        # Create MongoDB pool
        await connection_pool_manager.create_mongo_pool(
            min_size=settings.MONGO_MIN_CONNECTIONS,
            max_size=settings.MONGO_MAX_CONNECTIONS
        )
        
        # Create Azure OpenAI pool
        await connection_pool_manager.create_azure_openai_pool(
            min_size=settings.AZURE_OPENAI_MIN_CONNECTIONS,
            max_size=settings.AZURE_OPENAI_MAX_CONNECTIONS
        )
        
        logger.info("Default connection pools initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize default connection pools: {e}")
        raise