import asyncio
import traceback
from typing import Any, Callable, Coroutine, TypeVar, Optional, List, Dict, Union
from functools import wraps
import time
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from loguru import logger

from .config import settings
from .metrics import track_active_tasks, measure_time, ERROR_COUNTER

T = TypeVar('T')


class CoroutineManager:
    """
    Enhanced asynchronous coroutine management with advanced features.
    
    This class provides tools for:
    - Tracking and managing active coroutines
    - Enforcing timeouts
    - Graceful cleanup
    - Automatic retries with exponential backoff
    - Thread pool execution for blocking operations
    - Performance metrics
    """
    
    def __init__(self, max_workers: int = None):
        self._active_coroutines = set()
        self._loop = None
        self._thread_pool = ThreadPoolExecutor(
            max_workers=max_workers or settings.MAX_CONCURRENT_TASKS
        )
        self._task_metadata = {}
    
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Get the current event loop or create a new one if needed."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.get_event_loop()
        return self._loop
    
    @property
    def active_task_count(self) -> int:
        """Get the number of currently active tasks."""
        return len(self._active_coroutines)
    
    async def execute_coroutine(
        self, 
        coroutine: Coroutine, 
        task_name: str = None,
        task_type: str = "default"
    ) -> Any:
        """
        Execute a coroutine safely and track it.
        
        Args:
            coroutine: The coroutine to execute
            task_name: Optional name for the task
            task_type: Category of task for metrics
            
        Returns:
            The result of the coroutine
        """
        start_time = time.time()
        task_name = task_name or f"task-{id(coroutine)}"
        
        # Register the task
        task = asyncio.create_task(coroutine)
        self._active_coroutines.add(task)
        self._task_metadata[task] = {
            "name": task_name,
            "type": task_type,
            "start_time": start_time,
            "status": "running"
        }
        
        logger.debug(f"Started task '{task_name}' (type: {task_type})")
        
        try:
            # Execute and time the coroutine
            result = await task
            self._task_metadata[task]["status"] = "completed"
            duration = time.time() - start_time
            logger.debug(f"Completed task '{task_name}' in {duration:.2f}s")
            return result
            
        except asyncio.CancelledError:
            # Handle task cancellation
            self._task_metadata[task]["status"] = "cancelled"
            logger.warning(f"Task '{task_name}' was cancelled")
            raise
            
        except Exception as e:
            # Handle errors
            self._task_metadata[task]["status"] = "failed"
            self._task_metadata[task]["error"] = str(e)
            error_type = e.__class__.__name__
            
            # Increment error counter for metrics
            ERROR_COUNTER.labels(
                error_type=error_type,
                component=task_type
            ).inc()
            
            logger.error(f"Error in task '{task_name}': {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            raise
            
        finally:
            # Clean up
            if task in self._active_coroutines:
                self._active_coroutines.remove(task)
                
            if task in self._task_metadata:
                metadata = self._task_metadata.pop(task)
                if metadata.get("status") == "running":
                    metadata["status"] = "unknown"
    
    async def execute_with_timeout(
        self, 
        coroutine: Coroutine, 
        timeout: float,
        task_name: str = None,
        task_type: str = "default"
    ) -> Any:
        """
        Execute a coroutine with a timeout.
        
        Args:
            coroutine: The coroutine to execute
            timeout: Timeout in seconds
            task_name: Optional name for the task
            task_type: Category of task for metrics
            
        Returns:
            The result of the coroutine
            
        Raises:
            asyncio.TimeoutError: If the coroutine exceeds the timeout
        """
        try:
            return await asyncio.wait_for(
                self.execute_coroutine(
                    coroutine, 
                    task_name=task_name,
                    task_type=task_type
                ), 
                timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Task '{task_name}' timed out after {timeout} seconds")
            raise
    
    def ensure_future(
        self, 
        coroutine: Coroutine,
        task_name: str = None,
        task_type: str = "default"
    ) -> asyncio.Task:
        """
        Schedule a coroutine for execution and track it.
        
        Args:
            coroutine: The coroutine to schedule
            task_name: Optional name for the task
            task_type: Category of task for metrics
            
        Returns:
            The created task
        """
        return self.loop.create_task(
            self.execute_coroutine(
                coroutine,
                task_name=task_name,
                task_type=task_type
            )
        )
    
    async def gather_coroutines(
        self, 
        *coroutines: Coroutine,
        return_exceptions: bool = True,
        task_prefix: str = "batch",
        task_type: str = "batch"
    ) -> List[Any]:
        """
        Execute multiple coroutines in parallel and wait for all results.
        
        Args:
            *coroutines: The coroutines to execute
            return_exceptions: Whether to include exceptions in the results
            task_prefix: Prefix for task names
            task_type: Category of tasks for metrics
            
        Returns:
            List of results. If return_exceptions is True, exceptions are
            included in the results. Otherwise, only successful results are included.
        """
        tasks = []
        for i, coro in enumerate(coroutines):
            task_name = f"{task_prefix}-{i}"
            task = self.ensure_future(coro, task_name=task_name, task_type=task_type)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        
        if not return_exceptions:
            return results
        
        # Filter out exceptions if requested
        return [r for r in results if not isinstance(r, Exception)]
    
    async def run_in_thread(
        self, 
        func: Callable[..., T], 
        *args, 
        **kwargs
    ) -> T:
        """
        Run a blocking function in a separate thread.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function
        """
        func_name = getattr(func, "__name__", str(func))
        logger.debug(f"Running function '{func_name}' in thread pool")
        
        try:
            return await self.loop.run_in_executor(
                self._thread_pool,
                lambda: func(*args, **kwargs)
            )
        except Exception as e:
            logger.error(f"Error in thread pool execution of '{func_name}': {e}")
            raise
    
    def coroutine_handler(
        self, 
        timeout: Optional[float] = None,
        retry_count: int = 0,
        task_type: str = "default",
        measure: bool = True
    ):
        """
        Decorator for handling asynchronous functions with timeouts and retries.
        
        Args:
            timeout: Optional timeout in seconds
            retry_count: Number of retry attempts (0 means no retries)
            task_type: Category of task for metrics
            measure: Whether to measure execution time for metrics
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable[..., Coroutine]):
            # Apply retry if requested
            if retry_count > 0:
                func = retry(
                    wait=wait_exponential(
                        multiplier=settings.RETRY_BACKOFF,
                        min=1,
                        max=60
                    ),
                    stop=stop_after_attempt(retry_count + 1),  # +1 because first attempt is not a retry
                    retry=retry_if_exception_type(
                        (RuntimeError, ConnectionError, TimeoutError)
                    )
                )(func)
            
            @wraps(func)
            @track_active_tasks(task_type)
            async def wrapper(*args, **kwargs):
                func_name = func.__name__
                task_name = f"{func_name}-{id(func)}"
                
                logger.debug(f"Executing '{func_name}' with args: {args}, kwargs: {kwargs}")
                
                coroutine = func(*args, **kwargs)
                
                if timeout is not None:
                    return await self.execute_with_timeout(
                        coroutine, 
                        timeout, 
                        task_name=task_name,
                        task_type=task_type
                    )
                
                return await self.execute_coroutine(
                    coroutine,
                    task_name=task_name,
                    task_type=task_type
                )
            
            # Add the original function as an attribute for introspection
            wrapper.__original_func__ = func
            
            return wrapper
        
        return decorator
    
    async def cleanup(self, timeout: float = 5.0) -> None:
        """
        Clean up and cancel all active coroutines.
        
        Args:
            timeout: Timeout for clean shutdown in seconds
        """
        if not self._active_coroutines:
            logger.debug("No active coroutines to clean up")
            return
        
        logger.info(f"Cleaning up {len(self._active_coroutines)} active coroutines")
        
        # Cancel all active tasks
        for task in self._active_coroutines:
            if isinstance(task, asyncio.Task) and not task.done():
                metadata = self._task_metadata.get(task, {})
                task_name = metadata.get("name", str(id(task)))
                logger.debug(f"Cancelling task '{task_name}'")
                task.cancel()
        
        # Wait for all tasks to complete or be cancelled
        if self._active_coroutines:
            try:
                await asyncio.wait(
                    list(self._active_coroutines), 
                    timeout=timeout
                )
            except Exception as e:
                logger.warning(f"Error during task cleanup: {e}")
        
        # Clear all tracking data
        self._active_coroutines.clear()
        self._task_metadata.clear()
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=False)
        
        logger.info("Coroutine cleanup completed")


# Create a global instance of the coroutine manager
coroutine_manager = CoroutineManager()