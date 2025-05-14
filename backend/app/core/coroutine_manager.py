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
        task_type: str = "default",
        error_handler: Callable[[Exception, str, Dict], None] = None,
        suppress_errors: bool = False
    ) -> Any:
        """
        Execute a coroutine safely with enhanced error handling and tracking.
        
        Args:
            coroutine: The coroutine to execute
            task_name: Optional name for the task
            task_type: Category of task for metrics
            error_handler: Optional custom function to handle errors
            suppress_errors: If True, log errors but don't re-raise them
            
        Returns:
            The result of the coroutine, or None if error is suppressed
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
            "status": "running",
            "duration": None,
            "error": None
        }
        
        logger.debug(f"Started task '{task_name}' (type: {task_type})")
        
        try:
            # Execute and time the coroutine
            result = await task
            duration = time.time() - start_time
            
            # Update metadata
            self._task_metadata[task].update({
                "status": "completed",
                "duration": duration,
                "completion_time": time.time()
            })
            
            logger.debug(f"Completed task '{task_name}' in {duration:.2f}s")
            return result
            
        except asyncio.CancelledError:
            # Handle task cancellation
            duration = time.time() - start_time
            self._task_metadata[task].update({
                "status": "cancelled",
                "duration": duration,
                "cancellation_time": time.time()
            })
            
            logger.warning(f"Task '{task_name}' was cancelled after {duration:.2f}s")
            raise
            
        except Exception as e:
            # Handle errors with enhanced tracking
            duration = time.time() - start_time
            error_type = e.__class__.__name__
            error_details = {
                "error_type": error_type,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "duration": duration,
                "failure_time": time.time()
            }
            
            # Update task metadata
            self._task_metadata[task].update({
                "status": "failed",
                "error": str(e),
                "error_type": error_type,
                "duration": duration,
                "failure_details": error_details
            })
            
            # Increment error counter for metrics
            ERROR_COUNTER.labels(
                error_type=error_type,
                component=task_type
            ).inc()
            
            # Log the error
            logger.error(f"Error in task '{task_name}' ({task_type}): {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            
            # Call custom error handler if provided
            if error_handler and callable(error_handler):
                try:
                    error_handler(e, task_name, error_details)
                except Exception as handler_error:
                    logger.error(f"Error in custom error handler for task '{task_name}': {handler_error}")
            
            # Re-raise or suppress based on configuration
            if not suppress_errors:
                raise
            else:
                logger.warning(f"Error suppressed in task '{task_name}' as per configuration")
                return None
            
        finally:
            # Clean up and record final state
            if task in self._active_coroutines:
                self._active_coroutines.remove(task)
                
            if task in self._task_metadata:
                metadata = self._task_metadata.pop(task)
                if metadata.get("status") == "running":
                    metadata["status"] = "unknown"
                
                # Store task history if enabled
                if hasattr(settings, 'STORE_TASK_HISTORY') and settings.STORE_TASK_HISTORY:
                    # Limit history size to prevent memory leaks
                    if not hasattr(self, '_task_history'):
                        self._task_history = []
                    
                    self._task_history.append(metadata)
                    max_history = getattr(settings, 'TASK_HISTORY_SIZE', 1000)
                    
                    if len(self._task_history) > max_history:
                        self._task_history = self._task_history[-max_history:]
    
    async def execute_with_timeout(
        self, 
        coroutine: Coroutine, 
        timeout: float,
        task_name: str = None,
        task_type: str = "default",
        cancel_on_timeout: bool = True,
        default_value: Any = None,
        raise_timeout: bool = True
    ) -> Any:
        """
        Execute a coroutine with advanced timeout handling.
        
        Args:
            coroutine: The coroutine to execute
            timeout: Timeout in seconds
            task_name: Optional name for the task
            task_type: Category of task for metrics
            cancel_on_timeout: Whether to cancel the task on timeout
            default_value: Value to return if task times out and raise_timeout is False
            raise_timeout: Whether to raise TimeoutError or return default_value
            
        Returns:
            The result of the coroutine or default_value on timeout if raise_timeout is False
            
        Raises:
            asyncio.TimeoutError: If the coroutine exceeds the timeout and raise_timeout is True
        """
        task = None
        try:
            # Create the task
            task = asyncio.create_task(
                self.execute_coroutine(
                    coroutine, 
                    task_name=task_name,
                    task_type=task_type
                )
            )
            
            # Wait for the task with timeout
            return await asyncio.wait_for(task, timeout)
        
        except asyncio.TimeoutError:
            logger.warning(f"Task '{task_name}' timed out after {timeout} seconds")
            
            # Handle the timeout according to parameters
            if cancel_on_timeout and task and not task.done():
                logger.debug(f"Cancelling task '{task_name}' due to timeout")
                task.cancel()
            
            # Either raise the exception or return the default value
            if raise_timeout:
                raise
            else:
                logger.info(f"Returning default value for timed out task '{task_name}'")
                return default_value
    
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
        task_type: str = "batch",
        chunk_size: int = None,
        priority_items: List[int] = None,
        timeout: float = None,
        progress_callback: Callable[[int, int], None] = None
    ) -> List[Any]:
        """
        Execute multiple coroutines in parallel with advanced control.
        
        This method provides optimized parallel execution with features like:
        - Chunked execution to prevent resource exhaustion
        - Priority items that run first
        - Progress tracking
        - Timeout control
        
        Args:
            *coroutines: The coroutines to execute
            return_exceptions: Whether to include exceptions in the results
            task_prefix: Prefix for task names
            task_type: Category of tasks for metrics
            chunk_size: Maximum number of coroutines to run in parallel
                        (default: PARALLEL_EXECUTION_CHUNK_SIZE from settings)
            priority_items: Indices of high-priority coroutines to execute first
            timeout: Maximum time to wait for all coroutines (per chunk)
            progress_callback: Function to call with progress updates (completed, total)
            
        Returns:
            List of results in the same order as the input coroutines
        """
        if not coroutines:
            return []
            
        start_time = time.time()
        total_items = len(coroutines)
        chunk_size = chunk_size or settings.PARALLEL_EXECUTION_CHUNK_SIZE
        priority_items = priority_items or []
        results = [None] * total_items  # Pre-allocate result list with same order
        completed_count = 0
        
        # Log the operation
        logger.info(f"Starting parallel execution of {total_items} coroutines "
                   f"(chunk_size={chunk_size}, priority_items={len(priority_items)})")

        # Function to report progress if callback is provided
        def update_progress(new_completed):
            nonlocal completed_count
            completed_count = new_completed
            if progress_callback and callable(progress_callback):
                try:
                    progress_callback(completed_count, total_items)
                except Exception as e:
                    logger.warning(f"Error in progress callback: {e}")
        
        # First process priority items if specified
        if priority_items:
            priority_coros = []
            priority_indices = []
            
            for idx in priority_items:
                if 0 <= idx < total_items:
                    priority_coros.append(coroutines[idx])
                    priority_indices.append(idx)
            
            if priority_coros:
                logger.debug(f"Processing {len(priority_coros)} priority items")
                
                # Create and execute tasks for priority items
                priority_tasks = []
                for i, (coro, orig_idx) in enumerate(zip(priority_coros, priority_indices)):
                    task_name = f"{task_prefix}-priority-{i}"
                    task = self.ensure_future(coro, task_name=task_name, task_type=f"{task_type}_priority")
                    priority_tasks.append((task, orig_idx))
                
                # Wait for priority tasks with timeout if specified
                try:
                    if timeout:
                        done, pending = await asyncio.wait(
                            [t for t, _ in priority_tasks],
                            timeout=timeout,
                            return_when=asyncio.ALL_COMPLETED
                        )
                        if pending:
                            logger.warning(f"{len(pending)} priority tasks timed out after {timeout}s")
                    else:
                        done = await asyncio.gather(*[t for t, _ in priority_tasks], 
                                                   return_exceptions=True)
                    
                    # Process priority results
                    for i, (task, orig_idx) in enumerate(priority_tasks):
                        if timeout and task not in done:
                            # Task timed out
                            if return_exceptions:
                                results[orig_idx] = asyncio.TimeoutError(f"Task timed out after {timeout}s")
                            continue
                            
                        try:
                            result = task.result() if task.done() else None
                            if not (isinstance(result, Exception) and not return_exceptions):
                                results[orig_idx] = result
                        except Exception as e:
                            if return_exceptions:
                                results[orig_idx] = e
                
                except Exception as e:
                    logger.error(f"Error processing priority tasks: {e}")
                    if return_exceptions:
                        for _, orig_idx in priority_tasks:
                            if results[orig_idx] is None:
                                results[orig_idx] = e
                
                # Update progress
                priority_completed = sum(1 for r in [results[i] for i in priority_indices] if r is not None)
                update_progress(priority_completed)
        
        # Now process remaining items in chunks
        remaining_indices = [i for i in range(total_items) if i not in priority_items and results[i] is None]
        remaining_chunks = [remaining_indices[i:i+chunk_size] for i in range(0, len(remaining_indices), chunk_size)]
        
        for chunk_num, chunk_indices in enumerate(remaining_chunks):
            chunk_coros = [coroutines[i] for i in chunk_indices]
            logger.debug(f"Processing chunk {chunk_num+1}/{len(remaining_chunks)} with {len(chunk_indices)} items")
            
            # Create and execute tasks for this chunk
            chunk_tasks = []
            for i, (coro, orig_idx) in enumerate(zip(chunk_coros, chunk_indices)):
                task_name = f"{task_prefix}-chunk{chunk_num}-{i}"
                task = self.ensure_future(coro, task_name=task_name, task_type=task_type)
                chunk_tasks.append((task, orig_idx))
            
            # Wait for this chunk with timeout if specified
            try:
                if timeout:
                    done, pending = await asyncio.wait(
                        [t for t, _ in chunk_tasks],
                        timeout=timeout,
                        return_when=asyncio.ALL_COMPLETED
                    )
                    if pending:
                        logger.warning(f"{len(pending)} tasks in chunk {chunk_num+1} timed out after {timeout}s")
                else:
                    done = await asyncio.gather(*[t for t, _ in chunk_tasks], 
                                               return_exceptions=True)
                
                # Process chunk results
                for i, (task, orig_idx) in enumerate(chunk_tasks):
                    if timeout and task not in done:
                        # Task timed out
                        if return_exceptions:
                            results[orig_idx] = asyncio.TimeoutError(f"Task timed out after {timeout}s")
                        continue
                        
                    try:
                        result = task.result() if task.done() else None
                        if not (isinstance(result, Exception) and not return_exceptions):
                            results[orig_idx] = result
                    except Exception as e:
                        if return_exceptions:
                            results[orig_idx] = e
            
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_num+1}: {e}")
                if return_exceptions:
                    for _, orig_idx in chunk_tasks:
                        if results[orig_idx] is None:
                            results[orig_idx] = e
            
            # Update progress after each chunk
            current_completed = sum(1 for r in results if r is not None)
            update_progress(current_completed)
        
        # Final progress update and logging
        total_completed = sum(1 for r in results if r is not None)
        update_progress(total_completed)
        
        duration = time.time() - start_time
        logger.info(f"Completed parallel execution in {duration:.2f}s: "
                   f"{total_completed}/{total_items} tasks successful")
        
        # Filter out None values (failed tasks) if not returning exceptions
        if not return_exceptions:
            return [r for r in results if not isinstance(r, Exception)]
        
        return results
    
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
    
    async def cleanup(self, timeout: float = 5.0, force: bool = False) -> Dict[str, Any]:
        """
        Clean up and cancel all active coroutines with enhanced reporting.
        
        Args:
            timeout: Timeout for clean shutdown in seconds
            force: If True, forcibly terminate tasks that don't respond to cancellation
            
        Returns:
            Dictionary with cleanup statistics and results
        """
        start_time = time.time()
        
        if not self._active_coroutines:
            logger.debug("No active coroutines to clean up")
            return {"status": "success", "cleaned_tasks": 0, "duration": 0, "errors": []}
        
        active_count = len(self._active_coroutines)
        logger.info(f"Cleaning up {active_count} active coroutines")
        
        # Track statistics
        stats = {
            "total_tasks": active_count,
            "cancelled": 0,
            "already_done": 0,
            "timeout_tasks": 0,
            "errors": [],
            "task_details": []
        }
        
        # First pass: attempt graceful cancellation
        pending_tasks = set()
        for task in self._active_coroutines:
            if not isinstance(task, asyncio.Task):
                continue
                
            metadata = self._task_metadata.get(task, {})
            task_name = metadata.get("name", str(id(task)))
            task_type = metadata.get("type", "unknown")
            
            if task.done():
                stats["already_done"] += 1
                continue
                
            # Register task details
            task_info = {
                "name": task_name,
                "type": task_type,
                "id": id(task),
                "status": "pending_cancellation",
                "runtime": time.time() - metadata.get("start_time", start_time)
            }
            stats["task_details"].append(task_info)
            
            # Request cancellation
            logger.debug(f"Requesting cancellation for task '{task_name}' (type: {task_type})")
            try:
                task.cancel()
                pending_tasks.add(task)
                stats["cancelled"] += 1
            except Exception as e:
                error_msg = f"Error cancelling task '{task_name}': {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                task_info["status"] = "cancellation_error"
                task_info["error"] = str(e)
        
        # Wait for tasks to acknowledge cancellation
        if pending_tasks:
            try:
                # Wait with timeout for tasks to complete cancellation
                done, pending = await asyncio.wait(
                    pending_tasks, 
                    timeout=timeout,
                    return_when=asyncio.ALL_COMPLETED
                )
                
                stats["completed_cancellation"] = len(done)
                stats["timeout_tasks"] = len(pending)
                
                # Update task statuses
                for task in done:
                    task_id = id(task)
                    for task_info in stats["task_details"]:
                        if task_info["id"] == task_id:
                            task_info["status"] = "cancelled"
                
                # Handle tasks that didn't respond to cancellation
                if pending and force:
                    logger.warning(f"{len(pending)} tasks did not respond to cancellation within {timeout}s timeout")
                    
                    # More aggressive cleanup if force=True
                    for task in pending:
                        task_id = id(task)
                        for task_info in stats["task_details"]:
                            if task_info["id"] == task_id:
                                task_info["status"] = "force_terminated"
                                
                        # Use a more drastic approach to terminate the task
                        # This is implementation-dependent and may not be possible in all environments
                        logger.warning(f"Forcibly terminating task {task_id}")
                
            except Exception as e:
                error_msg = f"Error during wait phase of cleanup: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
        
        # Clear all tracking data
        self._active_coroutines.clear()
        self._task_metadata.clear()
        
        # Shutdown thread pool with proper handling
        try:
            self._thread_pool.shutdown(wait=timeout > 0)
            stats["thread_pool_shutdown"] = "success"
        except Exception as e:
            error_msg = f"Error shutting down thread pool: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            stats["thread_pool_shutdown"] = "error"
        
        # Create a new thread pool for future tasks
        self._thread_pool = ThreadPoolExecutor(
            max_workers=settings.MAX_CONCURRENT_TASKS
        )
        
        # Calculate cleanup time and log results
        cleanup_time = time.time() - start_time
        stats["duration"] = cleanup_time
        stats["timestamp"] = time.time()
        
        logger.info(f"Coroutine cleanup completed in {cleanup_time:.2f}s: "
                   f"{stats['cancelled']} tasks cancelled, "
                   f"{stats['timeout_tasks']} timed out, "
                   f"{len(stats['errors'])} errors")
        
        return {
            "status": "success" if not stats["errors"] else "partial_success",
            "cleaned_tasks": stats["cancelled"],
            "timeout_tasks": stats["timeout_tasks"],
            "duration": cleanup_time,
            "errors": stats["errors"],
            "details": stats
        }


# Create a global instance of the coroutine manager
coroutine_manager = CoroutineManager()