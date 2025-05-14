"""
Benchmark script for the enhanced CoroutineManager.

This script compares the performance of different execution strategies:
1. Standard asyncio.gather without any management
2. Basic coroutine manager without chunking
3. Enhanced coroutine manager with chunking
4. Enhanced coroutine manager with chunking and priorities

The benchmark simulates a real-world RAG scenario with mixed workloads:
- Short tasks (like embedding lookups)
- Medium tasks (like document fetching)
- Long tasks (like LLM inference)
- Some tasks that fail or timeout

Results are presented with timing and resource usage statistics.
"""
import asyncio
import time
import random
import sys
import os
import psutil
import statistics
from pathlib import Path
from typing import List, Dict, Any, Callable, Coroutine
import gc

# Add the project root to the Python path to make imports work
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.coroutine_manager import coroutine_manager, CoroutineManager
from app.core.config import settings
from loguru import logger


# Configure logger to display on console
logger.remove()
logger.add(sys.stdout, level="INFO")


# Define different types of workloads to simulate real-world scenarios
async def short_task(task_id: int, duration: float = 0.05):
    """Simulate a lightweight task like a quick embedding lookup."""
    await asyncio.sleep(duration)
    return {"task_id": task_id, "type": "short", "duration": duration}


async def medium_task(task_id: int, duration: float = 0.2):
    """Simulate a medium-weight task like fetching a document."""
    # Simulate a bit of CPU work
    start = time.time()
    while time.time() - start < duration * 0.3:
        _ = [i * i for i in range(10000)]
    
    await asyncio.sleep(duration * 0.7)
    return {"task_id": task_id, "type": "medium", "duration": duration}


async def long_task(task_id: int, duration: float = 0.5):
    """Simulate a heavy task like LLM inference."""
    # Simulate memory usage
    mem_data = [0] * 1000000  # Allocate some memory
    
    # Simulate mixed CPU and IO work
    start = time.time()
    while time.time() - start < duration * 0.4:
        _ = [i ** 2 for i in range(50000)]
    
    await asyncio.sleep(duration * 0.6)
    
    # Don't need explicit del as Python will handle this
    return {"task_id": task_id, "type": "long", "duration": duration}


async def failing_task(task_id: int):
    """Simulate a task that fails."""
    await asyncio.sleep(0.05)
    raise RuntimeError(f"Simulated failure in task {task_id}")


async def timeout_task(task_id: int, duration: float = 2.0):
    """Simulate a task that will likely time out."""
    await asyncio.sleep(duration)
    return {"task_id": task_id, "type": "timeout", "duration": duration}


def create_mixed_workload(count: int = 100, fail_rate: float = 0.05, timeout_rate: float = 0.05) -> List[Coroutine]:
    """Create a mixed workload of various task types."""
    tasks = []
    
    for i in range(count):
        # Distribute task types according to realistic workload
        r = random.random()
        
        if r < fail_rate:
            tasks.append(failing_task(i))
        elif r < fail_rate + timeout_rate:
            tasks.append(timeout_task(i))
        elif r < 0.7:
            # 60% short tasks
            duration = random.uniform(0.01, 0.1)
            tasks.append(short_task(i, duration))
        elif r < 0.9:
            # 20% medium tasks
            duration = random.uniform(0.1, 0.3)
            tasks.append(medium_task(i, duration))
        else:
            # 10% long tasks
            duration = random.uniform(0.3, 0.8)
            tasks.append(long_task(i, duration))
    
    random.shuffle(tasks)  # Shuffle to ensure random distribution
    return tasks


def get_system_metrics() -> Dict[str, float]:
    """Get current system resource usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "cpu_percent": process.cpu_percent(),
        "memory_rss_mb": memory_info.rss / (1024 * 1024),
        "memory_vms_mb": memory_info.vms / (1024 * 1024),
        "thread_count": process.num_threads(),
        "open_files": len(process.open_files()),
        "system_cpu": psutil.cpu_percent(),
        "system_memory_percent": psutil.virtual_memory().percent
    }


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.duration = 0.0
        self.task_count = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.timeout_tasks = 0
        self.start_metrics = {}
        self.end_metrics = {}
        self.task_durations = []
    
    def calculate_stats(self) -> Dict[str, Any]:
        """Calculate performance statistics."""
        # Skip if no tasks were run
        if not self.task_durations:
            return {}
        
        return {
            "min_duration": min(self.task_durations),
            "max_duration": max(self.task_durations),
            "avg_duration": statistics.mean(self.task_durations),
            "median_duration": statistics.median(self.task_durations),
            "stddev_duration": statistics.stdev(self.task_durations) if len(self.task_durations) > 1 else 0,
            "memory_increase_mb": self.end_metrics.get("memory_rss_mb", 0) - self.start_metrics.get("memory_rss_mb", 0),
            "cpu_average": (self.start_metrics.get("cpu_percent", 0) + self.end_metrics.get("cpu_percent", 0)) / 2,
            "throughput": self.successful_tasks / self.duration if self.duration > 0 else 0
        }
    
    def print_report(self):
        """Print a formatted report of benchmark results."""
        stats = self.calculate_stats()
        
        logger.info(f"=== {self.name} Benchmark Results ===")
        logger.info(f"Total duration: {self.duration:.2f}s")
        logger.info(f"Tasks: {self.successful_tasks} successful, {self.failed_tasks} failed, {self.timeout_tasks} timeouts")
        logger.info(f"Throughput: {stats.get('throughput', 0):.2f} tasks/second")
        logger.info(f"Task timing: min={stats.get('min_duration', 0):.4f}s, max={stats.get('max_duration', 0):.4f}s, avg={stats.get('avg_duration', 0):.4f}s")
        logger.info(f"Memory usage increase: {stats.get('memory_increase_mb', 0):.2f} MB")
        logger.info(f"CPU usage: {stats.get('cpu_average', 0):.1f}%")
        logger.info("-" * 50)


async def benchmark_standard_gather(workload: List[Coroutine], timeout: float = 1.0) -> BenchmarkResult:
    """Benchmark standard asyncio.gather without any management."""
    result = BenchmarkResult("Standard asyncio.gather")
    result.task_count = len(workload)
    
    # Capture starting metrics
    result.start_metrics = get_system_metrics()
    
    start_time = time.time()
    try:
        # Create tasks
        tasks = [asyncio.create_task(coro) for coro in workload]
        
        # Wait for tasks with timeout
        done, pending = await asyncio.wait(
            tasks, 
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED
        )
        
        # Process results
        for task in done:
            try:
                task_result = task.result()
                result.successful_tasks += 1
                if isinstance(task_result, dict) and "duration" in task_result:
                    result.task_durations.append(task_result["duration"])
            except Exception:
                result.failed_tasks += 1
        
        result.timeout_tasks = len(pending)
        
        # Cancel any pending tasks
        for task in pending:
            task.cancel()
        
    except Exception as e:
        logger.error(f"Error in standard gather benchmark: {e}")
    
    # Capture end metrics
    result.duration = time.time() - start_time
    result.end_metrics = get_system_metrics()
    
    # Ensure proper cleanup
    gc.collect()
    await asyncio.sleep(0.1)
    
    return result


async def benchmark_basic_manager(workload: List[Coroutine], timeout: float = 1.0) -> BenchmarkResult:
    """Benchmark basic coroutine manager without chunking."""
    result = BenchmarkResult("Basic CoroutineManager")
    result.task_count = len(workload)
    
    # Create a fresh instance for clean test
    manager = CoroutineManager()
    
    # Capture starting metrics
    result.start_metrics = get_system_metrics()
    
    start_time = time.time()
    try:
        # Execute all tasks at once (no chunking)
        all_results = await asyncio.wait_for(
            asyncio.gather(*[
                manager.execute_coroutine(coro, task_name=f"basic-{i}", task_type="benchmark") 
                for i, coro in enumerate(workload)
            ], return_exceptions=True),
            timeout=timeout
        )
        
        # Process results
        for res in all_results:
            if isinstance(res, Exception):
                result.failed_tasks += 1
            else:
                result.successful_tasks += 1
                if isinstance(res, dict) and "duration" in res:
                    result.task_durations.append(res["duration"])
        
    except asyncio.TimeoutError:
        result.timeout_tasks = result.task_count - result.successful_tasks - result.failed_tasks
    except Exception as e:
        logger.error(f"Error in basic manager benchmark: {e}")
    
    # Capture end metrics
    result.duration = time.time() - start_time
    result.end_metrics = get_system_metrics()
    
    # Ensure proper cleanup
    await manager.cleanup()
    gc.collect()
    await asyncio.sleep(0.1)
    
    return result


async def benchmark_chunked_manager(workload: List[Coroutine], chunk_size: int = 10, timeout: float = 1.0) -> BenchmarkResult:
    """Benchmark enhanced coroutine manager with chunking."""
    result = BenchmarkResult(f"Enhanced Manager with Chunking (size={chunk_size})")
    result.task_count = len(workload)
    
    # Capture starting metrics
    result.start_metrics = get_system_metrics()
    
    start_time = time.time()
    try:
        # Execute tasks with chunking
        all_results = await coroutine_manager.gather_coroutines(
            *workload,
            chunk_size=chunk_size,
            task_prefix="chunked",
            task_type="benchmark",
            timeout=timeout,
            return_exceptions=True
        )
        
        # Process results
        for res in all_results:
            if isinstance(res, Exception):
                if isinstance(res, asyncio.TimeoutError):
                    result.timeout_tasks += 1
                else:
                    result.failed_tasks += 1
            else:
                result.successful_tasks += 1
                if isinstance(res, dict) and "duration" in res:
                    result.task_durations.append(res["duration"])
        
    except Exception as e:
        logger.error(f"Error in chunked manager benchmark: {e}")
    
    # Capture end metrics
    result.duration = time.time() - start_time
    result.end_metrics = get_system_metrics()
    
    # Ensure proper cleanup
    await coroutine_manager.cleanup()
    gc.collect()
    await asyncio.sleep(0.1)
    
    return result


async def benchmark_prioritized_manager(
    workload: List[Coroutine], 
    chunk_size: int = 10, 
    timeout: float = 1.0,
    priority_count: int = 10
) -> BenchmarkResult:
    """Benchmark enhanced coroutine manager with chunking and priorities."""
    result = BenchmarkResult(f"Enhanced Manager with Priorities (size={chunk_size}, priority={priority_count})")
    result.task_count = len(workload)
    
    # Determine which tasks are high priority (pick some long tasks to prioritize)
    priority_indices = []
    for i, coro in enumerate(workload):
        if len(priority_indices) < priority_count:
            # Check if it's a long task (this is a heuristic and might not be perfect)
            # In real scenarios, you'd know which tasks are more important
            if hasattr(coro, "__qualname__") and "long_task" in coro.__qualname__:
                priority_indices.append(i)
    
    # Fill remaining priority slots with random tasks
    while len(priority_indices) < priority_count:
        idx = random.randint(0, len(workload) - 1)
        if idx not in priority_indices:
            priority_indices.append(idx)
    
    # Capture starting metrics
    result.start_metrics = get_system_metrics()
    
    start_time = time.time()
    try:
        # Execute tasks with chunking and priorities
        all_results = await coroutine_manager.gather_coroutines(
            *workload,
            chunk_size=chunk_size,
            task_prefix="prioritized",
            task_type="benchmark",
            timeout=timeout,
            priority_items=priority_indices,
            return_exceptions=True
        )
        
        # Process results
        for res in all_results:
            if isinstance(res, Exception):
                if isinstance(res, asyncio.TimeoutError):
                    result.timeout_tasks += 1
                else:
                    result.failed_tasks += 1
            else:
                result.successful_tasks += 1
                if isinstance(res, dict) and "duration" in res:
                    result.task_durations.append(res["duration"])
        
    except Exception as e:
        logger.error(f"Error in prioritized manager benchmark: {e}")
    
    # Capture end metrics
    result.duration = time.time() - start_time
    result.end_metrics = get_system_metrics()
    
    # Ensure proper cleanup
    await coroutine_manager.cleanup()
    gc.collect()
    await asyncio.sleep(0.1)
    
    return result


async def run_benchmarks(task_count: int = 100, iterations: int = 3):
    """Run all benchmarks multiple times and report average results."""
    logger.info(f"Starting benchmarks with {task_count} tasks, {iterations} iterations each")
    
    # Benchmark parameters
    timeout = 2.0
    chunk_sizes = [5, 10, 20]
    
    # Results storage
    all_results = []
    
    # Run each benchmark multiple times
    for iteration in range(iterations):
        logger.info(f"Starting iteration {iteration+1}/{iterations}")
        
        # Create a fresh workload for each iteration
        workload = create_mixed_workload(count=task_count)
        
        # Baseline benchmark: standard asyncio.gather
        result = await benchmark_standard_gather(workload, timeout=timeout)
        result.name += f" (Run {iteration+1})"
        result.print_report()
        all_results.append(result)
        
        # Basic manager benchmark
        result = await benchmark_basic_manager(workload, timeout=timeout)
        result.name += f" (Run {iteration+1})"
        result.print_report()
        all_results.append(result)
        
        # Chunked manager benchmarks with different chunk sizes
        for chunk_size in chunk_sizes:
            result = await benchmark_chunked_manager(workload, chunk_size=chunk_size, timeout=timeout)
            result.name += f" (Run {iteration+1})"
            result.print_report()
            all_results.append(result)
        
        # Prioritized manager benchmark
        result = await benchmark_prioritized_manager(
            workload, 
            chunk_size=10, 
            timeout=timeout, 
            priority_count=int(task_count * 0.1)  # 10% of tasks as priority
        )
        result.name += f" (Run {iteration+1})"
        result.print_report()
        all_results.append(result)
        
        # Small delay between iterations to let system stabilize
        await asyncio.sleep(1)
        logger.info("=" * 70)
    
    # Generate summary report
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    
    # Group results by benchmark type
    grouped_results = {}
    for result in all_results:
        # Extract the base name without the run number
        base_name = result.name.split(" (Run")[0]
        if base_name not in grouped_results:
            grouped_results[base_name] = []
        grouped_results[base_name].append(result)
    
    # Generate summary for each benchmark type
    for benchmark_name, results in grouped_results.items():
        durations = [r.duration for r in results]
        throughputs = [r.successful_tasks / r.duration for r in results if r.duration > 0]
        success_rates = [r.successful_tasks / r.task_count for r in results if r.task_count > 0]
        
        logger.info(f"Benchmark: {benchmark_name}")
        logger.info(f"  Avg Duration: {statistics.mean(durations):.2f}s")
        logger.info(f"  Avg Throughput: {statistics.mean(throughputs):.2f} tasks/second")
        logger.info(f"  Avg Success Rate: {statistics.mean(success_rates) * 100:.1f}%")
        logger.info("-" * 50)


if __name__ == "__main__":
    # Parse command line arguments
    task_count = 100
    iterations = 3
    
    if len(sys.argv) > 1:
        try:
            task_count = int(sys.argv[1])
        except ValueError:
            pass
    
    if len(sys.argv) > 2:
        try:
            iterations = int(sys.argv[2])
        except ValueError:
            pass
    
    # Run the benchmarks
    asyncio.run(run_benchmarks(task_count=task_count, iterations=iterations))