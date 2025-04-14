from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from functools import wraps
from typing import Callable, Any, Dict, Optional, List, Union
import asyncio


# Define metrics
REQUESTS_TOTAL = Counter(
    "requests_total", 
    "Total number of requests by endpoint and status",
    ["endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "request_duration_seconds", 
    "Request duration in seconds by endpoint",
    ["endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

DOCUMENT_PROCESSING_DURATION = Histogram(
    "document_processing_duration_seconds", 
    "Document processing duration in seconds by file type",
    ["file_type"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)
)

EMBEDDING_CREATION_DURATION = Histogram(
    "embedding_creation_duration_seconds", 
    "Embedding creation duration in seconds by model",
    ["model"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
)

EMBEDDING_RETRIEVAL_DURATION = Histogram(
    "embedding_retrieval_duration_seconds", 
    "Embedding retrieval duration in seconds by collection",
    ["collection"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
)

ACTIVE_TASKS_GAUGE = Gauge(
    "active_tasks",
    "Number of currently active tasks",
    ["task_type"]
)

CACHE_HITS = Counter(
    "cache_hits_total",
    "Number of cache hits by cache type",
    ["cache_type"]
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Number of cache misses by cache type",
    ["cache_type"]
)

LLM_TOKENS_USED = Counter(
    "llm_tokens_total",
    "Number of tokens used by LLM calls",
    ["model", "operation"]
)

RERANKING_QUALITY = Summary(
    "reranking_quality",
    "Reranking quality scores",
    ["reranker"]
)

ERROR_COUNTER = Counter(
    "errors_total",
    "Total number of errors by type",
    ["error_type", "component"]
)


def measure_time(metric: Histogram, labels: Dict[str, str]) -> Callable:
    """Decorator to measure execution time of a function using a Prometheus Histogram."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.labels(**labels).observe(duration)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.labels(**labels).observe(duration)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def track_active_tasks(task_type: str) -> Callable:
    """Decorator to track active tasks using a Prometheus Gauge."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            ACTIVE_TASKS_GAUGE.labels(task_type=task_type).inc()
            try:
                return func(*args, **kwargs)
            finally:
                ACTIVE_TASKS_GAUGE.labels(task_type=task_type).dec()
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            ACTIVE_TASKS_GAUGE.labels(task_type=task_type).inc()
            try:
                return await func(*args, **kwargs)
            finally:
                ACTIVE_TASKS_GAUGE.labels(task_type=task_type).dec()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def count_error(error_type: str, component: str) -> None:
    """Count an error occurrence."""
    ERROR_COUNTER.labels(error_type=error_type, component=component).inc()


def record_llm_tokens(model: str, operation: str, token_count: int) -> None:
    """Record the number of tokens used in LLM operations."""
    LLM_TOKENS_USED.labels(model=model, operation=operation).inc(token_count)


def record_reranking_score(reranker: str, score: float) -> None:
    """Record a reranking quality score."""
    RERANKING_QUALITY.labels(reranker=reranker).observe(score)


def record_cache_result(cache_type: str, hit: bool) -> None:
    """Record a cache hit or miss."""
    if hit:
        CACHE_HITS.labels(cache_type=cache_type).inc()
    else:
        CACHE_MISSES.labels(cache_type=cache_type).inc()