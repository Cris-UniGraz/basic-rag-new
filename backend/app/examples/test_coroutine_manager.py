"""
Test script for the enhanced CoroutineManager.

This script demonstrates the advanced features of the CoroutineManager including:
- Chunked parallel execution
- Priority task handling
- Timeout management
- Progress tracking
- Task history and metrics
"""
import asyncio
import time
import random
import sys
import os
from pathlib import Path

# Add the project root to the Python path to make imports work
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.coroutine_manager import coroutine_manager
from app.core.config import settings
from loguru import logger


# Configure logger to display on console
logger.remove()
logger.add(sys.stdout, level="INFO")


# Define test coroutines with various behaviors
async def task_normal(task_id: int, delay: float = 0.5):
    """A normal task that completes successfully after a delay."""
    logger.info(f"Task {task_id}: Starting normal task with {delay}s delay")
    await asyncio.sleep(delay)
    logger.info(f"Task {task_id}: Completed successfully")
    return {"task_id": task_id, "status": "success", "delay": delay}


async def task_with_error(task_id: int, fail_probability: float = 0.5):
    """A task that may fail with a given probability."""
    logger.info(f"Task {task_id}: Starting task with {fail_probability} chance of failure")
    await asyncio.sleep(0.2)
    
    if random.random() < fail_probability:
        logger.warning(f"Task {task_id}: Simulating failure")
        raise RuntimeError(f"Simulated error in task {task_id}")
    
    logger.info(f"Task {task_id}: Completed successfully (no error triggered)")
    return {"task_id": task_id, "status": "success"}


async def task_long_running(task_id: int, duration: float = 3.0):
    """A long-running task that may exceed timeout thresholds."""
    logger.info(f"Task {task_id}: Starting long-running task ({duration}s)")
    start_time = time.time()
    
    try:
        await asyncio.sleep(duration)
        elapsed = time.time() - start_time
        logger.info(f"Task {task_id}: Completed long-running task in {elapsed:.2f}s")
        return {"task_id": task_id, "status": "success", "duration": elapsed}
    except asyncio.CancelledError:
        elapsed = time.time() - start_time
        logger.warning(f"Task {task_id}: Cancelled after {elapsed:.2f}s")
        raise


def progress_callback(completed: int, total: int):
    """Callback function to report progress."""
    percentage = (completed / total) * 100 if total > 0 else 0
    logger.info(f"Progress: {completed}/{total} tasks completed ({percentage:.1f}%)")


async def test_chunked_execution():
    """Test the chunked parallel execution feature."""
    logger.info("=== Testing Chunked Execution ===")
    
    # Create a large number of tasks
    num_tasks = 50
    chunk_size = 10
    logger.info(f"Creating {num_tasks} tasks to be processed in chunks of {chunk_size}")
    
    # Create tasks with random delays
    tasks = [
        task_normal(i, delay=random.uniform(0.1, 0.5)) 
        for i in range(num_tasks)
    ]
    
    # Execute tasks with chunking
    start_time = time.time()
    results = await coroutine_manager.gather_coroutines(
        *tasks,
        chunk_size=chunk_size,
        task_prefix="chunk-test",
        task_type="demo",
        progress_callback=progress_callback
    )
    
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    
    logger.info(f"Chunked execution completed in {elapsed:.2f}s")
    logger.info(f"Results: {successful}/{num_tasks} tasks completed successfully")


async def test_priority_execution():
    """Test the priority task execution feature."""
    logger.info("=== Testing Priority Execution ===")
    
    # Create regular and priority tasks
    num_regular = 20
    num_priority = 5
    total_tasks = num_regular + num_priority
    
    logger.info(f"Creating {num_regular} regular tasks and {num_priority} priority tasks")
    
    # Create all tasks with longer delays for regular tasks
    all_tasks = [
        task_normal(i, delay=random.uniform(0.5, 1.0)) 
        for i in range(total_tasks)
    ]
    
    # Set priority tasks (the last few tasks)
    priority_indices = list(range(num_regular, total_tasks))
    logger.info(f"Priority tasks: {priority_indices}")
    
    # Execute tasks with priority handling
    start_time = time.time()
    results = await coroutine_manager.gather_coroutines(
        *all_tasks,
        chunk_size=10,
        task_prefix="priority-test",
        task_type="demo",
        priority_items=priority_indices,
        progress_callback=progress_callback
    )
    
    elapsed = time.time() - start_time
    
    logger.info(f"Priority execution completed in {elapsed:.2f}s")
    priority_results = [results[i] for i in priority_indices]
    logger.info(f"Priority results: {len(priority_results)}/{num_priority} completed")


async def test_timeout_handling():
    """Test timeout handling for tasks."""
    logger.info("=== Testing Timeout Handling ===")
    
    # Create a mix of fast and slow tasks
    num_fast = 10
    num_slow = 5
    timeout = 1.0
    
    logger.info(f"Creating {num_fast} fast tasks and {num_slow} slow tasks with {timeout}s timeout")
    
    # Create tasks - fast ones complete within timeout, slow ones exceed it
    fast_tasks = [task_normal(i, delay=random.uniform(0.1, 0.3)) for i in range(num_fast)]
    slow_tasks = [task_long_running(i + num_fast, duration=random.uniform(1.5, 3.0)) for i in range(num_slow)]
    
    all_tasks = fast_tasks + slow_tasks
    
    # Execute with timeout
    start_time = time.time()
    results = await coroutine_manager.gather_coroutines(
        *all_tasks,
        chunk_size=15,  # Process all in one chunk
        task_prefix="timeout-test",
        task_type="demo",
        timeout=timeout,
        progress_callback=progress_callback
    )
    
    elapsed = time.time() - start_time
    
    # Analyze results
    timeout_count = sum(1 for r in results if isinstance(r, asyncio.TimeoutError))
    success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    
    logger.info(f"Timeout test completed in {elapsed:.2f}s")
    logger.info(f"Results: {success_count} successful tasks, {timeout_count} timeouts")


async def test_error_handling():
    """Test error handling capabilities."""
    logger.info("=== Testing Error Handling ===")
    
    # Create tasks with different error probabilities
    num_tasks = 20
    
    logger.info(f"Creating {num_tasks} tasks with varying error probabilities")
    
    # Create tasks with increasing probability of error
    tasks = [
        task_with_error(i, fail_probability=i/num_tasks) 
        for i in range(num_tasks)
    ]
    
    # First test with return_exceptions=True
    logger.info("Running with return_exceptions=True")
    start_time = time.time()
    results_with_exceptions = await coroutine_manager.gather_coroutines(
        *tasks,
        return_exceptions=True,
        task_prefix="error-test-1",
        task_type="demo"
    )
    
    elapsed = time.time() - start_time
    error_count = sum(1 for r in results_with_exceptions if isinstance(r, Exception))
    success_count = sum(1 for r in results_with_exceptions if isinstance(r, dict))
    
    logger.info(f"Error handling test 1 completed in {elapsed:.2f}s")
    logger.info(f"Results: {success_count} successful tasks, {error_count} errors")
    
    # Then test with return_exceptions=False
    logger.info("Running with return_exceptions=False")
    start_time = time.time()
    results_without_exceptions = await coroutine_manager.gather_coroutines(
        *tasks,
        return_exceptions=False,
        task_prefix="error-test-2",
        task_type="demo"
    )
    
    elapsed = time.time() - start_time
    
    logger.info(f"Error handling test 2 completed in {elapsed:.2f}s")
    logger.info(f"Results: {len(results_without_exceptions)} successful tasks returned")


async def test_resource_cleanup():
    """Test automatic resource cleanup."""
    logger.info("=== Testing Resource Cleanup ===")
    
    # Create and start some tasks
    num_tasks = 10
    
    logger.info(f"Creating {num_tasks} tasks before cleanup")
    
    # Start tasks but don't wait for them
    tasks = []
    for i in range(num_tasks):
        coro = task_long_running(i, duration=random.uniform(2.0, 5.0))
        task = coroutine_manager.ensure_future(coro, task_name=f"cleanup-test-{i}", task_type="demo")
        tasks.append(task)
    
    logger.info(f"Started {len(tasks)} tasks, waiting briefly...")
    await asyncio.sleep(1.0)
    
    # Now perform cleanup
    logger.info(f"Active tasks before cleanup: {coroutine_manager.active_task_count}")
    cleanup_result = await coroutine_manager.cleanup(timeout=2.0)
    
    logger.info(f"Cleanup completed: {cleanup_result['status']}")
    logger.info(f"Cleaned up {cleanup_result['cleaned_tasks']} tasks")
    logger.info(f"Active tasks after cleanup: {coroutine_manager.active_task_count}")
    
    if cleanup_result['errors']:
        logger.warning(f"Cleanup errors: {cleanup_result['errors']}")


async def main():
    """Run all tests in sequence."""
    logger.info("Starting CoroutineManager tests")
    
    # Run tests
    await test_chunked_execution()
    await asyncio.sleep(1)  # Pause between tests
    
    await test_priority_execution()
    await asyncio.sleep(1)
    
    await test_timeout_handling()
    await asyncio.sleep(1)
    
    await test_error_handling()
    await asyncio.sleep(1)
    
    await test_resource_cleanup()
    
    logger.info("All tests completed")


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())