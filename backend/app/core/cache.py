import redis
import json
import hashlib
import pickle
from typing import Any, Optional, Dict, Callable, TypeVar, Tuple, List, cast
from functools import wraps
import asyncio
import time
from loguru import logger
from cachetools import TTLCache

from .config import settings
from .metrics import record_cache_result

# Type variables for generic function signatures
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# In-memory cache for small objects with TTL
memory_cache = TTLCache(maxsize=1000, ttl=settings.CACHE_TTL)

# Progress tracking cache
upload_progress_cache = {}

# Initialize Redis connection
try:
    redis_client = redis.Redis.from_url(
        settings.REDIS_URL, 
        decode_responses=False,  # We'll handle decoding ourselves
        socket_timeout=5,
        socket_connect_timeout=5
    )
    logger.info(f"Connected to Redis at {settings.REDIS_URL}")
except Exception as e:
    logger.warning(f"Failed to connect to Redis: {e}. Falling back to memory cache only.")
    redis_client = None


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Generate a deterministic cache key from function arguments.
    
    Args:
        prefix: A string prefix for the cache key, typically the function name
        *args: Positional arguments to include in the key
        **kwargs: Keyword arguments to include in the key
        
    Returns:
        A string hash representation of the arguments
    """
    # Convert args and kwargs to a string representation
    key_parts = [prefix]
    
    # Add positional args
    for arg in args:
        # Handle different types appropriately
        key_parts.append(str(arg))
    
    # Add keyword args (sorted to ensure deterministic keys)
    for k in sorted(kwargs.keys()):
        key_parts.append(f"{k}={kwargs[k]}")
    
    # Join and hash to create a fixed-length key
    key_str = ":".join(key_parts)
    return f"{prefix}:{hashlib.md5(key_str.encode()).hexdigest()}"


def cache_result(
    ttl: int = None, 
    prefix: str = None,
    use_memory: bool = True,
    use_redis: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Cache function results in memory and/or Redis.
    
    Args:
        ttl: Time-to-live in seconds (defaults to settings.CACHE_TTL)
        prefix: Prefix for cache keys (defaults to function name)
        use_memory: Whether to use in-memory cache
        use_redis: Whether to use Redis cache
        
    Returns:
        Decorated function with caching behavior
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if not settings.ENABLE_CACHE:
            return func
            
        func_name = func.__name__
        cache_prefix = prefix or func_name
        cache_ttl = ttl or settings.CACHE_TTL
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            if not settings.ENABLE_CACHE:
                return func(*args, **kwargs)
                
            # Generate cache key
            cache_key = generate_cache_key(cache_prefix, *args, **kwargs)
            result = None
            found_in_cache = False
            
            # Try memory cache first
            if use_memory:
                try:
                    if cache_key in memory_cache:
                        result = memory_cache[cache_key]
                        found_in_cache = True
                        record_cache_result("memory", True)
                        logger.debug(f"Cache hit (memory): {cache_key}")
                except Exception as e:
                    logger.warning(f"Error accessing memory cache: {e}")
            
            # Try Redis if not found in memory
            if not found_in_cache and use_redis and redis_client:
                try:
                    cached_data = redis_client.get(cache_key)
                    if cached_data:
                        result = pickle.loads(cached_data)
                        found_in_cache = True
                        record_cache_result("redis", True)
                        logger.debug(f"Cache hit (Redis): {cache_key}")
                        
                        # Also store in memory for future fast access
                        if use_memory:
                            memory_cache[cache_key] = result
                except Exception as e:
                    logger.warning(f"Error accessing Redis cache: {e}")
            
            # If not found in any cache, execute function
            if not found_in_cache:
                record_cache_result("all", False)
                logger.debug(f"Cache miss: {cache_key}")
                result = func(*args, **kwargs)
                
                # Cache the result
                try:
                    if use_memory:
                        memory_cache[cache_key] = result
                    
                    if use_redis and redis_client:
                        redis_client.setex(
                            cache_key,
                            cache_ttl,
                            pickle.dumps(result)
                        )
                except Exception as e:
                    logger.warning(f"Error storing in cache: {e}")
            
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            if not settings.ENABLE_CACHE:
                return await func(*args, **kwargs)
                
            # Generate cache key
            cache_key = generate_cache_key(cache_prefix, *args, **kwargs)
            result = None
            found_in_cache = False
            
            # Try memory cache first
            if use_memory:
                try:
                    if cache_key in memory_cache:
                        result = memory_cache[cache_key]
                        found_in_cache = True
                        record_cache_result("memory", True)
                        logger.debug(f"Cache hit (memory): {cache_key}")
                except Exception as e:
                    logger.warning(f"Error accessing memory cache: {e}")
            
            # Try Redis if not found in memory
            if not found_in_cache and use_redis and redis_client:
                try:
                    # Run Redis get operation in the default thread pool
                    loop = asyncio.get_event_loop()
                    cached_data = await loop.run_in_executor(
                        None, lambda: redis_client.get(cache_key)
                    )
                    
                    if cached_data:
                        result = pickle.loads(cached_data)
                        found_in_cache = True
                        record_cache_result("redis", True)
                        logger.debug(f"Cache hit (Redis): {cache_key}")
                        
                        # Also store in memory for future fast access
                        if use_memory:
                            memory_cache[cache_key] = result
                except Exception as e:
                    logger.warning(f"Error accessing Redis cache: {e}")
            
            # If not found in any cache, execute function
            if not found_in_cache:
                record_cache_result("all", False)
                logger.debug(f"Cache miss: {cache_key}")
                result = await func(*args, **kwargs)
                
                # Cache the result
                try:
                    if use_memory:
                        memory_cache[cache_key] = result
                    
                    if use_redis and redis_client:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, 
                            lambda: redis_client.setex(
                                cache_key,
                                cache_ttl,
                                pickle.dumps(result)
                            )
                        )
                except Exception as e:
                    logger.warning(f"Error storing in cache: {e}")
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def invalidate_cache(prefix: str, pattern: str = "*") -> None:
    """
    Invalidate cache entries matching a pattern.
    
    Args:
        prefix: Prefix used for the cache keys
        pattern: Pattern to match for cache invalidation (default: "*")
    """
    # Clear memory cache entries matching pattern
    keys_to_delete = []
    search_pattern = f"{prefix}:{pattern}"
    
    for key in memory_cache:
        if key.startswith(prefix):
            if pattern == "*" or pattern in key:
                keys_to_delete.append(key)
    
    # Delete from memory cache
    for key in keys_to_delete:
        memory_cache.pop(key, None)
    
    # Delete from Redis if available
    if redis_client:
        try:
            redis_pattern = f"{prefix}:{pattern}"
            # Get all keys matching pattern
            keys = redis_client.keys(redis_pattern)
            if keys:
                redis_client.delete(*keys)
                logger.debug(f"Invalidated {len(keys)} Redis cache entries matching {redis_pattern}")
        except Exception as e:
            logger.warning(f"Error invalidating Redis cache: {e}")
    
    logger.debug(f"Invalidated {len(keys_to_delete)} memory cache entries matching {search_pattern}")


def clear_all_caches() -> None:
    """Clear all caches completely."""
    # Clear memory cache
    memory_cache.clear()
    
    # Clear Redis cache if available
    if redis_client:
        try:
            redis_client.flushdb()
            logger.info("Cleared all Redis caches")
        except Exception as e:
            logger.warning(f"Error clearing Redis cache: {e}")
    
    logger.info("Cleared all in-memory caches")


# Upload progress tracking functions
def track_upload_progress(task_id: str, progress: int, status: str = None, message: str = None) -> None:
    """
    Update the progress of a document upload task.
    
    Args:
        task_id: Unique identifier for the upload task
        progress: Progress percentage (0-100)
        status: Status of the task (processing, completed, error)
        message: Optional message to include
    """
    progress_data = {
        "progress": progress,
        "status": status or "processing",
        "message": message or f"Processing documents - {progress}% complete",
        "timestamp": time.time()
    }
    
    # Update in-memory cache
    upload_progress_cache[task_id] = progress_data
    
    # Update in Redis if available
    if redis_client:
        try:
            redis_key = f"upload_progress:{task_id}"
            redis_client.setex(
                redis_key,
                3600,  # 1 hour TTL
                pickle.dumps(progress_data)
            )
        except Exception as e:
            logger.warning(f"Error storing upload progress in Redis: {e}")
    
    logger.debug(f"Updated progress for task {task_id}: {progress}%")


def get_upload_progress(task_id: str) -> Dict[str, Any]:
    """
    Get the current progress of a document upload task.
    
    Args:
        task_id: Unique identifier for the upload task
        
    Returns:
        Progress data dict with progress, status, and message
    """
    # Check in-memory cache first
    if task_id in upload_progress_cache:
        return upload_progress_cache[task_id]
    
    # Check Redis if available
    if redis_client:
        try:
            redis_key = f"upload_progress:{task_id}"
            cached_data = redis_client.get(redis_key)
            if cached_data:
                progress_data = pickle.loads(cached_data)
                # Also store in memory for future fast access
                upload_progress_cache[task_id] = progress_data
                return progress_data
        except Exception as e:
            logger.warning(f"Error retrieving upload progress from Redis: {e}")
    
    # Return default if not found
    return {
        "progress": 0,
        "status": "waiting",
        "message": "Waiting to start processing",
        "timestamp": time.time()
    }