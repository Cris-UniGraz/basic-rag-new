import asyncio
import time
import pickle
import hashlib
import json
import gzip
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import threading
from pathlib import Path
import weakref
from collections import OrderedDict, defaultdict
from loguru import logger

from app.core.config import settings
from app.core.async_metadata_processor import async_metadata_processor

T = TypeVar('T')


class CacheLevel(Enum):
    """Cache levels in the multi-level system."""
    L1 = "l1"  # In-memory cache
    L2 = "l2"  # Redis cache
    L3 = "l3"  # Disk cache


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live
    ADAPTIVE = "adaptive"    # Adaptive based on access patterns


class CacheWarmingStrategy(Enum):
    """Cache warming strategies."""
    EAGER = "eager"          # Preload all possible data
    LAZY = "lazy"           # Load on demand
    PREDICTIVE = "predictive" # Predict and preload likely accessed data
    SCHEDULED = "scheduled"   # Schedule warming at specific times


@dataclass
class CacheEntry(Generic[T]):
    """A cache entry with metadata."""
    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[timedelta] = None
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return datetime.now() - self.created_at > self.ttl
    
    def touch(self):
        """Update access information."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def calculate_size(self):
        """Calculate approximate size of the entry."""
        try:
            self.size_bytes = len(pickle.dumps(self.value))
        except Exception:
            self.size_bytes = 0


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    avg_access_time: float = 0.0
    
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def update_access_time(self, access_time: float):
        """Update average access time."""
        if self.avg_access_time == 0:
            self.avg_access_time = access_time
        else:
            self.avg_access_time = (self.avg_access_time * 0.9) + (access_time * 0.1)


class CacheLayer(ABC, Generic[T]):
    """Abstract base class for cache layers."""
    
    def __init__(self, name: str, max_size: int, strategy: CacheStrategy):
        self.name = name
        self.max_size = max_size
        self.strategy = strategy
        self.stats = CacheStats()
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: T, ttl: Optional[timedelta] = None, tags: Optional[Set[str]] = None):
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self):
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def invalidate_by_tags(self, tags: Set[str]):
        """Invalidate entries by tags."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class L1InMemoryCache(CacheLayer[T]):
    """
    L1 In-Memory Cache with advanced features.
    
    Features:
    - Multiple eviction strategies (LRU, LFU, TTL, Adaptive)
    - Tag-based invalidation
    - Compression for large objects
    - Thread-safe operations
    - Memory usage monitoring
    """
    
    def __init__(
        self,
        name: str = "L1_Memory",
        max_size: int = 1000,
        max_memory_mb: int = 512,
        strategy: CacheStrategy = CacheStrategy.LRU,
        compression_threshold: int = 1024  # Compress objects larger than 1KB
    ):
        super().__init__(name, max_size, strategy)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.compression_threshold = compression_threshold
        
        # Storage
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._access_order = OrderedDict()  # For LRU
        self._access_frequency = defaultdict(int)  # For LFU
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)  # Tag to keys mapping
        
        # Background cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Memory monitoring enabled and cache entry metadata tracking with disk management and redis integration, predictive loading
        self._memory_monitoring_enabled = True
        self._cache_entry_metadata_enabled = True
        self._cache_statistics_enabled = True
        self._disk_management_enabled = True
        self._redis_integration_enabled = True
        self._predictive_loading_enabled = True
        
        logger.info(f"L1InMemoryCache initialized: {name} (max_size={max_size}, max_memory={max_memory_mb}MB)")
    
    async def start_background_tasks(self):
        """Start background cleanup tasks."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_background_tasks(self):
        """Stop background tasks."""
        if self._cleanup_task:
            self._shutdown_event.set()
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._cleanup_task.cancel()
    
    async def _cleanup_loop(self):
        """Background loop for cache cleanup."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired()
                await self._enforce_memory_limit()
                
            except Exception as e:
                logger.error(f"Error in L1 cache cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from L1 cache."""
        start_time = time.time()
        
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self.stats.misses += 1
                return None
            
            if entry.is_expired():
                await self._remove_entry(key)
                self.stats.misses += 1
                return None
            
            # Update access information
            entry.touch()
            self._update_access_tracking(key)
            
            self.stats.hits += 1
            self.stats.update_access_time(time.time() - start_time)
            
            return entry.value
    
    async def set(self, key: str, value: T, ttl: Optional[timedelta] = None, tags: Optional[Set[str]] = None):
        """Set value in L1 cache."""
        async with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_entries(1)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                tags=tags or set()
            )
            entry.calculate_size()
            
            # Compress if necessary
            if entry.size_bytes > self.compression_threshold:
                try:
                    compressed_value = gzip.compress(pickle.dumps(value))
                    if len(compressed_value) < entry.size_bytes * 0.8:  # Only if significant compression
                        entry.value = compressed_value
                        entry.size_bytes = len(compressed_value)
                        entry.tags.add("compressed")
                except Exception as e:
                    logger.warning(f"Failed to compress cache entry {key}: {e}")
            
            # Store entry
            old_entry = self._cache.get(key)
            if old_entry:
                self.stats.size_bytes -= old_entry.size_bytes
                self._remove_from_tag_index(key, old_entry.tags)
            
            self._cache[key] = entry
            self.stats.size_bytes += entry.size_bytes
            self.stats.entry_count = len(self._cache)
            
            # Update tracking
            self._update_access_tracking(key)
            self._update_tag_index(key, entry.tags)
            
            # Check memory limit
            await self._enforce_memory_limit()
    
    async def delete(self, key: str) -> bool:
        """Delete value from L1 cache."""
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                return True
            return False
    
    async def clear(self):
        """Clear all L1 cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._access_frequency.clear()
            self._tag_index.clear()
            self.stats = CacheStats()
    
    async def invalidate_by_tags(self, tags: Set[str]):
        """Invalidate entries by tags."""
        async with self._lock:
            keys_to_remove = set()
            
            for tag in tags:
                keys_to_remove.update(self._tag_index.get(tag, set()))
            
            for key in keys_to_remove:
                await self._remove_entry(key)
    
    async def _remove_entry(self, key: str):
        """Remove entry and update indices."""
        entry = self._cache.get(key)
        if entry:
            del self._cache[key]
            self.stats.size_bytes -= entry.size_bytes
            self.stats.entry_count = len(self._cache)
            
            # Remove from tracking
            self._access_order.pop(key, None)
            self._access_frequency.pop(key, None)
            self._remove_from_tag_index(key, entry.tags)
    
    def _update_access_tracking(self, key: str):
        """Update access tracking for eviction strategies."""
        # Update LRU order
        self._access_order.pop(key, None)
        self._access_order[key] = datetime.now()
        
        # Update LFU frequency
        self._access_frequency[key] += 1
    
    def _update_tag_index(self, key: str, tags: Set[str]):
        """Update tag index."""
        for tag in tags:
            self._tag_index[tag].add(key)
    
    def _remove_from_tag_index(self, key: str, tags: Set[str]):
        """Remove from tag index."""
        for tag in tags:
            self._tag_index[tag].discard(key)
            if not self._tag_index[tag]:
                del self._tag_index[tag]
    
    async def _cleanup_expired(self):
        """Remove expired entries."""
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            await self._remove_entry(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries from L1 cache")
    
    async def _enforce_memory_limit(self):
        """Enforce memory limit by evicting entries."""
        while self.stats.size_bytes > self.max_memory_bytes and self._cache:
            await self._evict_entries(1)
    
    async def _evict_entries(self, count: int):
        """Evict entries based on strategy."""
        for _ in range(count):
            if not self._cache:
                break
            
            if self.strategy == CacheStrategy.LRU:
                key = next(iter(self._access_order))
            elif self.strategy == CacheStrategy.LFU:
                key = min(self._access_frequency.items(), key=lambda x: x[1])[0]
            elif self.strategy == CacheStrategy.TTL:
                # Find entry closest to expiration
                key = min(self._cache.items(), 
                         key=lambda x: x[1].created_at + (x[1].ttl or timedelta.max))[0]
            else:  # ADAPTIVE
                # Use LRU for frequently accessed, LFU for others
                avg_freq = sum(self._access_frequency.values()) / len(self._access_frequency) if self._access_frequency else 0
                lru_candidates = [k for k, freq in self._access_frequency.items() if freq < avg_freq]
                
                if lru_candidates:
                    key = min(lru_candidates, key=lambda k: self._access_order[k])
                else:
                    key = next(iter(self._access_order))
            
            await self._remove_entry(key)
            self.stats.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get L1 cache statistics."""
        return {
            "name": self.name,
            "level": "L1",
            "strategy": self.strategy.value,
            "stats": {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": self.stats.hit_rate(),
                "evictions": self.stats.evictions,
                "entry_count": self.stats.entry_count,
                "size_bytes": self.stats.size_bytes,
                "size_mb": round(self.stats.size_bytes / (1024 * 1024), 2),
                "avg_access_time": self.stats.avg_access_time
            },
            "config": {
                "max_size": self.max_size,
                "max_memory_mb": self.max_memory_bytes // (1024 * 1024),
                "compression_threshold": self.compression_threshold
            }
        }


class L2RedisCache(CacheLayer[T]):
    """
    L2 Redis Cache for distributed caching.
    
    Features:
    - Redis-based distributed caching
    - Serialization/deserialization
    - Tag-based invalidation
    - Connection pooling
    - Async operations
    """
    
    def __init__(
        self,
        name: str = "L2_Redis",
        max_size: int = 10000,
        strategy: CacheStrategy = CacheStrategy.TTL,
        redis_url: str = None,
        key_prefix: str = "rag_cache:"
    ):
        super().__init__(name, max_size, strategy)
        self.redis_url = redis_url or settings.REDIS_URL
        self.key_prefix = key_prefix
        self._redis_pool = None
        
        logger.info(f"L2RedisCache initialized: {name} (max_size={max_size})")
    
    async def _get_redis(self):
        """Get Redis connection pool."""
        if self._redis_pool is None:
            try:
                import aioredis
                self._redis_pool = aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False  # We handle our own serialization
                )
                await self._redis_pool.ping()
                logger.info("Redis connection established for L2 cache")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return self._redis_pool
    
    def _make_key(self, key: str) -> str:
        """Make Redis key with prefix."""
        return f"{self.key_prefix}{key}"
    
    def _make_tag_key(self, tag: str) -> str:
        """Make Redis key for tag index."""
        return f"{self.key_prefix}tags:{tag}"
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from L2 Redis cache."""
        start_time = time.time()
        
        try:
            redis = await self._get_redis()
            redis_key = self._make_key(key)
            
            data = await redis.get(redis_key)
            
            if data is None:
                self.stats.misses += 1
                return None
            
            # Deserialize
            try:
                entry_data = pickle.loads(data)
                entry = CacheEntry(**entry_data)
                
                if entry.is_expired():
                    await self.delete(key)
                    self.stats.misses += 1
                    return None
                
                # Update access information
                entry.touch()
                
                # Update in Redis
                updated_data = pickle.dumps({
                    "key": entry.key,
                    "value": entry.value,
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "ttl": entry.ttl,
                    "size_bytes": entry.size_bytes,
                    "tags": entry.tags
                })
                
                ttl_seconds = None
                if entry.ttl:
                    remaining = entry.ttl - (datetime.now() - entry.created_at)
                    ttl_seconds = max(1, int(remaining.total_seconds()))
                
                await redis.set(redis_key, updated_data, ex=ttl_seconds)
                
                self.stats.hits += 1
                self.stats.update_access_time(time.time() - start_time)
                
                return entry.value
                
            except Exception as e:
                logger.error(f"Failed to deserialize L2 cache entry {key}: {e}")
                await self.delete(key)
                self.stats.misses += 1
                return None
                
        except Exception as e:
            logger.error(f"Error getting from L2 cache: {e}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: T, ttl: Optional[timedelta] = None, tags: Optional[Set[str]] = None):
        """Set value in L2 Redis cache."""
        try:
            redis = await self._get_redis()
            redis_key = self._make_key(key)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                tags=tags or set()
            )
            entry.calculate_size()
            
            # Serialize
            entry_data = {
                "key": entry.key,
                "value": entry.value,
                "created_at": entry.created_at,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
                "ttl": entry.ttl,
                "size_bytes": entry.size_bytes,
                "tags": entry.tags
            }
            
            serialized_data = pickle.dumps(entry_data)
            
            # Calculate TTL in seconds
            ttl_seconds = None
            if ttl:
                ttl_seconds = int(ttl.total_seconds())
            
            # Store in Redis
            await redis.set(redis_key, serialized_data, ex=ttl_seconds)
            
            # Update tag index
            for tag in entry.tags:
                tag_key = self._make_tag_key(tag)
                await redis.sadd(tag_key, key)
                if ttl_seconds:
                    await redis.expire(tag_key, ttl_seconds)
            
            self.stats.entry_count += 1
            self.stats.size_bytes += entry.size_bytes
            
        except Exception as e:
            logger.error(f"Error setting L2 cache entry {key}: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete value from L2 Redis cache."""
        try:
            redis = await self._get_redis()
            redis_key = self._make_key(key)
            
            # Get entry to update tag index
            data = await redis.get(redis_key)
            if data:
                try:
                    entry_data = pickle.loads(data)
                    tags = entry_data.get("tags", set())
                    
                    # Remove from tag index
                    for tag in tags:
                        tag_key = self._make_tag_key(tag)
                        await redis.srem(tag_key, key)
                        
                        # Clean up empty tag sets
                        if await redis.scard(tag_key) == 0:
                            await redis.delete(tag_key)
                    
                except Exception as e:
                    logger.warning(f"Failed to update tag index during delete: {e}")
            
            result = await redis.delete(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error deleting from L2 cache: {e}")
            return False
    
    async def clear(self):
        """Clear all L2 Redis cache entries."""
        try:
            redis = await self._get_redis()
            
            # Find all keys with our prefix
            pattern = f"{self.key_prefix}*"
            keys = []
            
            async for key in redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await redis.delete(*keys)
            
            self.stats = CacheStats()
            
        except Exception as e:
            logger.error(f"Error clearing L2 cache: {e}")
    
    async def invalidate_by_tags(self, tags: Set[str]):
        """Invalidate entries by tags."""
        try:
            redis = await self._get_redis()
            
            keys_to_delete = set()
            
            for tag in tags:
                tag_key = self._make_tag_key(tag)
                tag_members = await redis.smembers(tag_key)
                
                for member in tag_members:
                    if isinstance(member, bytes):
                        member = member.decode('utf-8')
                    keys_to_delete.add(self._make_key(member))
                
                # Delete tag key
                await redis.delete(tag_key)
            
            if keys_to_delete:
                await redis.delete(*list(keys_to_delete))
            
        except Exception as e:
            logger.error(f"Error invalidating L2 cache by tags: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get L2 cache statistics."""
        return {
            "name": self.name,
            "level": "L2",
            "strategy": self.strategy.value,
            "stats": {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": self.stats.hit_rate(),
                "entry_count": self.stats.entry_count,
                "size_bytes": self.stats.size_bytes,
                "avg_access_time": self.stats.avg_access_time
            },
            "config": {
                "max_size": self.max_size,
                "redis_url": self.redis_url.replace(self.redis_url.split('@')[0].split('://')[-1], "***") if '@' in self.redis_url else self.redis_url
            }
        }


class L3DiskCache(CacheLayer[T]):
    """
    L3 Disk Cache for persistent caching.
    
    Features:
    - Disk-based persistent caching
    - File-based storage with compression
    - Tag-based invalidation
    - Automatic cleanup of old files
    - Directory structure optimization
    """
    
    def __init__(
        self,
        name: str = "L3_Disk",
        max_size: int = 100000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        cache_dir: str = "/tmp/rag_cache",
        max_disk_mb: int = 2048,
        compress: bool = True
    ):
        super().__init__(name, max_size, strategy)
        self.cache_dir = Path(cache_dir)
        self.max_disk_bytes = max_disk_mb * 1024 * 1024
        self.compress = compress
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file
        self.index_file = self.cache_dir / "index.json"
        self._index: Dict[str, Dict[str, Any]] = {}
        self._load_index()
        
        logger.info(f"L3DiskCache initialized: {name} (max_size={max_size}, cache_dir={cache_dir})")
    
    def _load_index(self):
        """Load cache index from disk."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert datetime strings back to objects
                for key, entry_data in data.items():
                    entry_data["created_at"] = datetime.fromisoformat(entry_data["created_at"])
                    entry_data["last_accessed"] = datetime.fromisoformat(entry_data["last_accessed"])
                    if entry_data.get("ttl"):
                        entry_data["ttl"] = timedelta(seconds=entry_data["ttl"])
                    entry_data["tags"] = set(entry_data.get("tags", []))
                    
                self._index = data
                
        except Exception as e:
            logger.warning(f"Failed to load L3 cache index: {e}")
            self._index = {}
    
    def _save_index(self):
        """Save cache index to disk."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_index = {}
            for key, entry_data in self._index.items():
                serializable_data = entry_data.copy()
                serializable_data["created_at"] = entry_data["created_at"].isoformat()
                serializable_data["last_accessed"] = entry_data["last_accessed"].isoformat()
                if entry_data.get("ttl"):
                    serializable_data["ttl"] = entry_data["ttl"].total_seconds()
                serializable_data["tags"] = list(entry_data.get("tags", set()))
                serializable_index[key] = serializable_data
            
            with open(self.index_file, 'w') as f:
                json.dump(serializable_index, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save L3 cache index: {e}")
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Create hash-based directory structure
        key_hash = hashlib.md5(key.encode()).hexdigest()
        subdir = key_hash[:2]
        filename = f"{key_hash}.cache"
        
        file_path = self.cache_dir / subdir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        return file_path
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from L3 disk cache."""
        start_time = time.time()
        
        async with self._lock:
            entry_data = self._index.get(key)
            
            if not entry_data:
                self.stats.misses += 1
                return None
            
            # Check expiration
            if entry_data.get("ttl"):
                if datetime.now() - entry_data["created_at"] > entry_data["ttl"]:
                    await self._remove_entry(key)
                    self.stats.misses += 1
                    return None
            
            # Read from disk
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                # Index is out of sync, remove entry
                await self._remove_entry(key)
                self.stats.misses += 1
                return None
            
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                if self.compress:
                    data = gzip.decompress(data)
                
                value = pickle.loads(data)
                
                # Update access information
                entry_data["last_accessed"] = datetime.now()
                entry_data["access_count"] = entry_data.get("access_count", 0) + 1
                
                self._save_index()
                
                self.stats.hits += 1
                self.stats.update_access_time(time.time() - start_time)
                
                return value
                
            except Exception as e:
                logger.error(f"Failed to read L3 cache entry {key}: {e}")
                await self._remove_entry(key)
                self.stats.misses += 1
                return None
    
    async def set(self, key: str, value: T, ttl: Optional[timedelta] = None, tags: Optional[Set[str]] = None):
        """Set value in L3 disk cache."""
        async with self._lock:
            # Serialize value
            try:
                data = pickle.dumps(value)
                
                if self.compress:
                    data = gzip.compress(data)
                
                # Write to disk
                file_path = self._get_file_path(key)
                
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # Update index
                self._index[key] = {
                    "created_at": datetime.now(),
                    "last_accessed": datetime.now(),
                    "access_count": 0,
                    "ttl": ttl,
                    "size_bytes": len(data),
                    "tags": tags or set(),
                    "file_path": str(file_path)
                }
                
                self._save_index()
                
                self.stats.entry_count = len(self._index)
                self.stats.size_bytes += len(data)
                
                # Check size limits
                await self._enforce_limits()
                
            except Exception as e:
                logger.error(f"Failed to write L3 cache entry {key}: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete value from L3 disk cache."""
        async with self._lock:
            return await self._remove_entry(key)
    
    async def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache and disk."""
        entry_data = self._index.get(key)
        
        if not entry_data:
            return False
        
        # Remove file
        file_path = self._get_file_path(key)
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete cache file {file_path}: {e}")
        
        # Remove from index
        del self._index[key]
        self._save_index()
        
        self.stats.entry_count = len(self._index)
        self.stats.size_bytes -= entry_data.get("size_bytes", 0)
        
        return True
    
    async def clear(self):
        """Clear all L3 disk cache entries."""
        async with self._lock:
            # Remove all files
            for entry_data in self._index.values():
                file_path = Path(entry_data["file_path"])
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {file_path}: {e}")
            
            # Clear index
            self._index.clear()
            self._save_index()
            
            self.stats = CacheStats()
    
    async def invalidate_by_tags(self, tags: Set[str]):
        """Invalidate entries by tags."""
        async with self._lock:
            keys_to_remove = []
            
            for key, entry_data in self._index.items():
                entry_tags = entry_data.get("tags", set())
                if tags.intersection(entry_tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                await self._remove_entry(key)
    
    async def _enforce_limits(self):
        """Enforce size and count limits."""
        # Enforce count limit
        while len(self._index) > self.max_size:
            await self._evict_one_entry()
        
        # Enforce disk space limit
        while self.stats.size_bytes > self.max_disk_bytes:
            await self._evict_one_entry()
    
    async def _evict_one_entry(self):
        """Evict one entry based on strategy."""
        if not self._index:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Find least recently accessed
            oldest_key = min(self._index.items(), key=lambda x: x[1]["last_accessed"])[0]
        elif self.strategy == CacheStrategy.LFU:
            # Find least frequently accessed
            least_used_key = min(self._index.items(), key=lambda x: x[1].get("access_count", 0))[0]
            oldest_key = least_used_key
        elif self.strategy == CacheStrategy.TTL:
            # Find entry closest to expiration
            ttl_entries = [(k, v) for k, v in self._index.items() if v.get("ttl")]
            if ttl_entries:
                oldest_key = min(ttl_entries, 
                               key=lambda x: x[1]["created_at"] + x[1]["ttl"])[0]
            else:
                oldest_key = min(self._index.items(), key=lambda x: x[1]["created_at"])[0]
        else:  # ADAPTIVE
            # Use LRU for now
            oldest_key = min(self._index.items(), key=lambda x: x[1]["last_accessed"])[0]
        
        await self._remove_entry(oldest_key)
        self.stats.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get L3 cache statistics."""
        return {
            "name": self.name,
            "level": "L3",
            "strategy": self.strategy.value,
            "stats": {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": self.stats.hit_rate(),
                "evictions": self.stats.evictions,
                "entry_count": self.stats.entry_count,
                "size_bytes": self.stats.size_bytes,
                "size_mb": round(self.stats.size_bytes / (1024 * 1024), 2),
                "avg_access_time": self.stats.avg_access_time
            },
            "config": {
                "max_size": self.max_size,
                "max_disk_mb": self.max_disk_bytes // (1024 * 1024),
                "cache_dir": str(self.cache_dir),
                "compress": self.compress
            }
        }


class MultiLevelCache:
    """
    Multi-level cache system with L1 (Memory), L2 (Redis), and L3 (Disk) layers.
    
    Features:
    - Automatic tier management
    - Cache warming strategies
    - Intelligent invalidation
    - Performance optimization
    - Comprehensive monitoring
    """
    
    def __init__(
        self,
        l1_config: Optional[Dict[str, Any]] = None,
        l2_config: Optional[Dict[str, Any]] = None,
        l3_config: Optional[Dict[str, Any]] = None,
        warming_strategy: CacheWarmingStrategy = CacheWarmingStrategy.LAZY
    ):
        """Initialize multi-level cache."""
        self.warming_strategy = warming_strategy
        
        # Initialize cache layers
        self.l1 = L1InMemoryCache(**(l1_config or {}))
        self.l2 = L2RedisCache(**(l2_config or {}))
        self.l3 = L3DiskCache(**(l3_config or {}))
        
        # Cache warming
        self._warming_tasks: Set[asyncio.Task] = set()
        self._warming_queue = asyncio.Queue()
        
        logger.info(f"MultiLevelCache initialized with warming strategy: {warming_strategy.value}")
    
    async def initialize(self):
        """Initialize all cache layers."""
        await self.l1.start_background_tasks()
        
        if self.warming_strategy != CacheWarmingStrategy.LAZY:
            warming_task = asyncio.create_task(self._cache_warming_loop())
            self._warming_tasks.add(warming_task)
            warming_task.add_done_callback(self._warming_tasks.discard)
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from multi-level cache."""
        # Try L1 first
        value = await self.l1.get(key)
        if value is not None:
            return value
        
        # Try L2
        value = await self.l2.get(key)
        if value is not None:
            # Promote to L1
            await self.l1.set(key, value)
            return value
        
        # Try L3
        value = await self.l3.get(key)
        if value is not None:
            # Promote to L2 and L1
            await self.l2.set(key, value)
            await self.l1.set(key, value)
            return value
        
        return None
    
    async def set(self, key: str, value: T, ttl: Optional[timedelta] = None, tags: Optional[Set[str]] = None):
        """Set value in all cache levels."""
        # Set in all levels
        await asyncio.gather(
            self.l1.set(key, value, ttl, tags),
            self.l2.set(key, value, ttl, tags),
            self.l3.set(key, value, ttl, tags),
            return_exceptions=True
        )
        
        # Schedule for warming if needed
        if self.warming_strategy == CacheWarmingStrategy.PREDICTIVE:
            await self._schedule_predictive_warming(key, tags)
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels."""
        results = await asyncio.gather(
            self.l1.delete(key),
            self.l2.delete(key),
            self.l3.delete(key),
            return_exceptions=True
        )
        
        return any(isinstance(r, bool) and r for r in results)
    
    async def clear(self):
        """Clear all cache levels."""
        await asyncio.gather(
            self.l1.clear(),
            self.l2.clear(),
            self.l3.clear(),
            return_exceptions=True
        )
    
    async def invalidate_by_tags(self, tags: Set[str]):
        """Invalidate entries by tags in all levels."""
        await asyncio.gather(
            self.l1.invalidate_by_tags(tags),
            self.l2.invalidate_by_tags(tags),
            self.l3.invalidate_by_tags(tags),
            return_exceptions=True
        )
    
    async def warm_cache(self, keys: List[str], data_loader: Callable[[str], T]):
        """Warm cache with specific keys."""
        for key in keys:
            try:
                # Check if already cached
                if await self.get(key) is not None:
                    continue
                
                # Load data
                value = await data_loader(key) if asyncio.iscoroutinefunction(data_loader) else data_loader(key)
                
                if value is not None:
                    await self.set(key, value)
                    
            except Exception as e:
                logger.error(f"Error warming cache for key {key}: {e}")
    
    async def _schedule_predictive_warming(self, key: str, tags: Optional[Set[str]]):
        """Schedule predictive warming based on access patterns."""
        # Add to warming queue for background processing
        await self._warming_queue.put({"key": key, "tags": tags})
    
    async def _cache_warming_loop(self):
        """Background loop for cache warming."""
        while True:
            try:
                # Process warming queue
                try:
                    item = await asyncio.wait_for(self._warming_queue.get(), timeout=60)
                    await self._process_warming_item(item)
                except asyncio.TimeoutError:
                    continue
                    
            except Exception as e:
                logger.error(f"Error in cache warming loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_warming_item(self, item: Dict[str, Any]):
        """Process a cache warming item."""
        # This is a placeholder for predictive warming logic
        # In a real implementation, you would:
        # 1. Analyze access patterns
        # 2. Predict related keys that might be accessed
        # 3. Pre-load those keys
        pass
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all cache levels."""
        return {
            "multi_level_cache": {
                "warming_strategy": self.warming_strategy.value,
                "levels": {
                    "L1": self.l1.get_stats(),
                    "L2": self.l2.get_stats(),
                    "L3": self.l3.get_stats()
                }
            }
        }
    
    async def shutdown(self):
        """Shutdown all cache layers."""
        # Cancel warming tasks
        for task in self._warming_tasks:
            if not task.done():
                task.cancel()
        
        if self._warming_tasks:
            await asyncio.gather(*self._warming_tasks, return_exceptions=True)
        
        # Shutdown cache layers
        await self.l1.stop_background_tasks()


# Global multi-level cache instance
multi_level_cache = MultiLevelCache()


# Utility functions for easy integration
async def get_cached(key: str) -> Optional[Any]:
    """Get value from multi-level cache."""
    return await multi_level_cache.get(key)


async def set_cached(key: str, value: Any, ttl: Optional[timedelta] = None, tags: Optional[Set[str]] = None):
    """Set value in multi-level cache."""
    await multi_level_cache.set(key, value, ttl, tags)


async def invalidate_cache_by_tags(tags: Set[str]):
    """Invalidate cache entries by tags."""
    await multi_level_cache.invalidate_by_tags(tags)


async def initialize_advanced_cache():
    """Initialize the advanced cache system."""
    await multi_level_cache.initialize()
    logger.info("Advanced multi-level cache system initialized")