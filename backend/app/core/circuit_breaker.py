import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import functools
from loguru import logger

from app.core.config import settings
from app.core.async_metadata_processor import async_metadata_processor

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit is open, calls fail fast
    HALF_OPEN = "half_open" # Testing if service has recovered


class FailureType(Enum):
    """Types of failures that can trip a circuit breaker."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    CUSTOM_ERROR = "custom_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    name: str
    failure_threshold: int = 5          # Failures before opening circuit
    recovery_timeout: float = 60.0     # Seconds to wait before trying half-open
    request_timeout: float = 30.0      # Default timeout for requests
    expected_exception: type = Exception
    success_threshold: int = 3          # Successes in half-open before closing
    monitoring_period: float = 300.0   # Period to track failures (5 minutes)
    
    # Performance-based thresholds
    performance_threshold: float = 10.0  # Max response time before considering degraded
    performance_failure_rate: float = 0.5  # Rate of slow responses to trigger
    
    # Custom conditions
    custom_failure_condition: Optional[Callable[[Any], bool]] = None


@dataclass
class CallResult:
    """Result of a circuit breaker protected call."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    response_time: float = 0.0
    failure_type: Optional[FailureType] = None
    timestamp: datetime = field(default_factory=datetime.now)


class CircuitBreakerMetrics:
    """Metrics tracking for circuit breaker."""
    
    def __init__(self, monitoring_period: float = 300.0):
        self.monitoring_period = monitoring_period
        self.call_history: List[CallResult] = []
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.timeout_failures = 0
        self.exception_failures = 0
        self.performance_failures = 0
        self.circuit_opens = 0
        self.avg_response_time = 0.0
        
    def record_call(self, result: CallResult):
        """Record a call result."""
        self.total_calls += 1
        self.call_history.append(result)
        
        # Clean old entries
        cutoff_time = datetime.now() - timedelta(seconds=self.monitoring_period)
        self.call_history = [call for call in self.call_history if call.timestamp > cutoff_time]
        
        if result.success:
            self.successful_calls += 1
            self._update_response_time(result.response_time)
        else:
            self.failed_calls += 1
            if result.failure_type == FailureType.TIMEOUT:
                self.timeout_failures += 1
            elif result.failure_type == FailureType.EXCEPTION:
                self.exception_failures += 1
            elif result.failure_type == FailureType.PERFORMANCE_DEGRADATION:
                self.performance_failures += 1
    
    def _update_response_time(self, response_time: float):
        """Update average response time."""
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * 0.9) + (response_time * 0.1)
    
    def get_failure_rate(self) -> float:
        """Get current failure rate within monitoring period."""
        if not self.call_history:
            return 0.0
        
        failures = sum(1 for call in self.call_history if not call.success)
        return failures / len(self.call_history)
    
    def get_performance_degradation_rate(self, threshold: float) -> float:
        """Get rate of performance degradation."""
        if not self.call_history:
            return 0.0
        
        slow_calls = sum(1 for call in self.call_history 
                        if call.success and call.response_time > threshold)
        return slow_calls / len(self.call_history)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "failure_rate": self.get_failure_rate(),
            "avg_response_time": self.avg_response_time,
            "timeout_failures": self.timeout_failures,
            "exception_failures": self.exception_failures,
            "performance_failures": self.performance_failures,
            "circuit_opens": self.circuit_opens,
            "recent_calls": len(self.call_history)
        }


class CircuitBreaker:
    """
    Advanced circuit breaker implementation with multiple failure modes.
    
    Features:
    - Traditional exception-based failure detection
    - Timeout-based failure detection
    - Performance degradation detection
    - Custom failure conditions
    - Exponential backoff for recovery
    - Comprehensive metrics and logging
    - Async support with proper exception handling
    - Fallback strategies for graceful degradation
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics(config.monitoring_period)
        
        # State management
        self._state_lock = asyncio.Lock()
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._next_attempt_time: Optional[datetime] = None
        
        # Recovery management
        self._recovery_exponential_base = 1.5
        self._max_recovery_timeout = 300.0  # 5 minutes max
        self._current_recovery_timeout = config.recovery_timeout
        
        logger.info(f"CircuitBreaker created: {config.name}")
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Original exception: When function fails and circuit allows it
        """
        # Check if circuit allows the call
        if not await self._can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.config.name}' is OPEN. "
                f"Next attempt allowed at: {self._next_attempt_time}"
            )
        
        start_time = time.time()
        call_result = None
        
        try:
            # Execute the function with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.request_timeout
                )
            else:
                result = func(*args, **kwargs)
            
            response_time = time.time() - start_time
            
            # Check for performance degradation
            if response_time > self.config.performance_threshold:
                call_result = CallResult(
                    success=False,
                    result=None,
                    response_time=response_time,
                    failure_type=FailureType.PERFORMANCE_DEGRADATION
                )
                await self._on_failure(call_result)
                # Still return the result but mark as degraded performance
                logger.warning(f"Performance degradation detected in {self.config.name}: {response_time:.2f}s")
            else:
                call_result = CallResult(
                    success=True,
                    result=result,
                    response_time=response_time
                )
                await self._on_success(call_result)
            
            # Check custom failure condition
            if (self.config.custom_failure_condition and 
                self.config.custom_failure_condition(result)):
                call_result.success = False
                call_result.failure_type = FailureType.CUSTOM_ERROR
                await self._on_failure(call_result)
            
            return result
            
        except asyncio.TimeoutError as e:
            response_time = time.time() - start_time
            call_result = CallResult(
                success=False,
                error=e,
                response_time=response_time,
                failure_type=FailureType.TIMEOUT
            )
            await self._on_failure(call_result)
            raise
            
        except self.config.expected_exception as e:
            response_time = time.time() - start_time
            call_result = CallResult(
                success=False,
                error=e,
                response_time=response_time,
                failure_type=FailureType.EXCEPTION
            )
            await self._on_failure(call_result)
            raise
            
        except Exception as e:
            # Unexpected exception
            response_time = time.time() - start_time
            call_result = CallResult(
                success=False,
                error=e,
                response_time=response_time,
                failure_type=FailureType.EXCEPTION
            )
            await self._on_failure(call_result)
            raise
        
        finally:
            # Record metrics
            if call_result:
                self.metrics.record_call(call_result)
                
                # Log async
                async_metadata_processor.log_async(
                    "DEBUG" if call_result.success else "WARNING",
                    f"Circuit breaker call completed: {self.config.name}",
                    {
                        "circuit_name": self.config.name,
                        "success": call_result.success,
                        "response_time": call_result.response_time,
                        "failure_type": call_result.failure_type.value if call_result.failure_type else None,
                        "circuit_state": self.state.value
                    }
                )
    
    async def _can_execute(self) -> bool:
        """Check if circuit allows execution."""
        async with self._state_lock:
            now = datetime.now()
            
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if self._next_attempt_time and now >= self._next_attempt_time:
                    # Transition to half-open
                    await self._transition_to_half_open()
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
            
            return False
    
    async def _on_success(self, result: CallResult):
        """Handle successful call."""
        async with self._state_lock:
            if self.state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    await self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    async def _on_failure(self, result: CallResult):
        """Handle failed call."""
        async with self._state_lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            
            if self.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                failure_rate = self.metrics.get_failure_rate()
                
                if (self._failure_count >= self.config.failure_threshold or
                    failure_rate > 0.5):  # 50% failure rate
                    await self._transition_to_open()
                    
            elif self.state == CircuitState.HALF_OPEN:
                # Failure in half-open state, go back to open
                await self._transition_to_open()
    
    async def _transition_to_open(self):
        """Transition circuit to OPEN state."""
        self.state = CircuitState.OPEN
        self.metrics.circuit_opens += 1
        self._failure_count = 0
        self._success_count = 0
        
        # Calculate next attempt time with exponential backoff
        self._next_attempt_time = (
            datetime.now() + timedelta(seconds=self._current_recovery_timeout)
        )
        
        # Increase recovery timeout for next time (exponential backoff)
        self._current_recovery_timeout = min(
            self._current_recovery_timeout * self._recovery_exponential_base,
            self._max_recovery_timeout
        )
        
        logger.warning(f"Circuit breaker OPENED: {self.config.name}. "
                      f"Next attempt at: {self._next_attempt_time}")
        
        # Log async
        async_metadata_processor.log_async(
            "WARNING",
            f"Circuit breaker opened: {self.config.name}",
            {
                "circuit_name": self.config.name,
                "failure_count": self._failure_count,
                "failure_rate": self.metrics.get_failure_rate(),
                "next_attempt": self._next_attempt_time.isoformat()
            },
            priority=2
        )
    
    async def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self._success_count = 0
        
        logger.info(f"Circuit breaker transitioned to HALF_OPEN: {self.config.name}")
        
        # Log async
        async_metadata_processor.log_async(
            "INFO",
            f"Circuit breaker half-open: {self.config.name}",
            {
                "circuit_name": self.config.name,
                "recovery_timeout": self._current_recovery_timeout
            }
        )
    
    async def _transition_to_closed(self):
        """Transition circuit to CLOSED state."""
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._next_attempt_time = None
        
        # Reset recovery timeout on successful recovery
        self._current_recovery_timeout = self.config.recovery_timeout
        
        logger.info(f"Circuit breaker CLOSED (recovered): {self.config.name}")
        
        # Log async
        async_metadata_processor.log_async(
            "INFO",
            f"Circuit breaker recovered: {self.config.name}",
            {
                "circuit_name": self.config.name,
                "state": "closed"
            }
        )
    
    async def force_open(self):
        """Force circuit to open state."""
        async with self._state_lock:
            await self._transition_to_open()
        
        logger.warning(f"Circuit breaker force opened: {self.config.name}")
    
    async def force_close(self):
        """Force circuit to closed state."""
        async with self._state_lock:
            await self._transition_to_closed()
        
        logger.info(f"Circuit breaker force closed: {self.config.name}")
    
    async def force_half_open(self):
        """Force circuit to half-open state."""
        async with self._state_lock:
            await self._transition_to_half_open()
        
        logger.info(f"Circuit breaker force half-open: {self.config.name}")
    
    async def get_fallback_strategies(self) -> List[str]:
        """Get available fallback strategies when circuit is open."""
        return [
            "Graceful degradation to reduced service mode",
            "Cache-only responses when available",
            "Alternative service endpoints",
            "Default response patterns",
            "Service bypass for critical operations"
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "next_attempt_time": self._next_attempt_time.isoformat() if self._next_attempt_time else None,
            "current_recovery_timeout": self._current_recovery_timeout,
            "metrics": self.metrics.get_stats(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "request_timeout": self.config.request_timeout,
                "success_threshold": self.config.success_threshold,
                "performance_threshold": self.config.performance_threshold
            }
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers across the application.
    
    Features:
    - Centralized circuit breaker management
    - Default configurations for common patterns
    - Global monitoring and statistics
    - Automatic registration and cleanup
    """
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._manager_lock = asyncio.Lock()
        
        logger.info("CircuitBreakerManager initialized")
    
    async def get_or_create_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        async with self._manager_lock:
            if name not in self._circuit_breakers:
                if config is None:
                    config = self._get_default_config(name)
                
                circuit_breaker = CircuitBreaker(config)
                self._circuit_breakers[name] = circuit_breaker
                
                logger.info(f"Created circuit breaker: {name}")
            
            return self._circuit_breakers[name]
    
    def _get_default_config(self, name: str) -> CircuitBreakerConfig:
        """Get default configuration based on circuit breaker name."""
        # Define default configurations for different services
        if "vector_store" in name.lower():
            return CircuitBreakerConfig(
                name=name,
                failure_threshold=5,
                recovery_timeout=30.0,
                request_timeout=10.0,
                performance_threshold=5.0
            )
        elif "embedding" in name.lower():
            return CircuitBreakerConfig(
                name=name,
                failure_threshold=3,
                recovery_timeout=60.0,
                request_timeout=30.0,
                performance_threshold=15.0
            )
        elif "llm" in name.lower() or "chat" in name.lower():
            return CircuitBreakerConfig(
                name=name,
                failure_threshold=3,
                recovery_timeout=120.0,
                request_timeout=60.0,
                performance_threshold=30.0
            )
        else:
            # Generic default
            return CircuitBreakerConfig(
                name=name,
                failure_threshold=5,
                recovery_timeout=60.0,
                request_timeout=30.0,
                performance_threshold=10.0
            )
    
    async def get_global_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        async with self._manager_lock:
            circuit_statuses = {}
            summary = {
                "total_circuits": len(self._circuit_breakers),
                "open_circuits": 0,
                "half_open_circuits": 0,
                "closed_circuits": 0
            }
            
            for name, circuit in self._circuit_breakers.items():
                status = circuit.get_status()
                circuit_statuses[name] = status
                
                if status["state"] == "open":
                    summary["open_circuits"] += 1
                elif status["state"] == "half_open":
                    summary["half_open_circuits"] += 1
                else:
                    summary["closed_circuits"] += 1
            
            return {
                "summary": summary,
                "circuits": circuit_statuses,
                "timestamp": datetime.now().isoformat()
            }
    
    async def force_open_all(self):
        """Force all circuit breakers to open state."""
        async with self._manager_lock:
            for circuit in self._circuit_breakers.values():
                await circuit.force_open()
        
        logger.warning("All circuit breakers force opened")
    
    async def force_close_all(self):
        """Force all circuit breakers to closed state."""
        async with self._manager_lock:
            for circuit in self._circuit_breakers.values():
                await circuit.force_close()
        
        logger.info("All circuit breakers force closed")
    
    async def cleanup(self):
        """Clean up all circuit breakers."""
        async with self._manager_lock:
            self._circuit_breakers.clear()
        
        logger.info("CircuitBreakerManager cleanup completed")


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()


# Decorator for automatic circuit breaker protection
def circuit_breaker(
    name: Optional[str] = None,
    config: Optional[CircuitBreakerConfig] = None
):
    """
    Decorator to automatically apply circuit breaker protection to a function.
    
    Args:
        name: Circuit breaker name (defaults to function name)
        config: Circuit breaker configuration
    """
    def decorator(func):
        circuit_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            circuit = await circuit_breaker_manager.get_or_create_circuit_breaker(
                circuit_name, config
            )
            return await circuit.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to get circuit in async context
            async def _execute():
                circuit = await circuit_breaker_manager.get_or_create_circuit_breaker(
                    circuit_name, config
                )
                return await circuit.call(func, *args, **kwargs)
            
            return asyncio.create_task(_execute())
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage of @circuit_breaker decorator
@circuit_breaker(name="example_function")
async def example_protected_function():
    """Example function protected by circuit breaker."""
    pass


# Predefined circuit breakers for common use cases
async def get_vector_store_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for vector store operations."""
    config = CircuitBreakerConfig(
        name="vector_store_operations",
        failure_threshold=5,
        recovery_timeout=30.0,
        request_timeout=10.0,
        performance_threshold=5.0,
        monitoring_period=300.0
    )
    return await circuit_breaker_manager.get_or_create_circuit_breaker(
        "vector_store_operations", config
    )


async def get_embedding_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for embedding operations."""
    config = CircuitBreakerConfig(
        name="embedding_operations",
        failure_threshold=3,
        recovery_timeout=60.0,
        request_timeout=30.0,
        performance_threshold=15.0,
        monitoring_period=300.0
    )
    return await circuit_breaker_manager.get_or_create_circuit_breaker(
        "embedding_operations", config
    )


async def get_llm_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for LLM operations."""
    config = CircuitBreakerConfig(
        name="llm_operations",
        failure_threshold=3,
        recovery_timeout=120.0,
        request_timeout=60.0,
        performance_threshold=30.0,
        monitoring_period=300.0
    )
    return await circuit_breaker_manager.get_or_create_circuit_breaker(
        "llm_operations", config
    )


async def get_retriever_circuit_breaker(retriever_type: str) -> CircuitBreaker:
    """Get circuit breaker for specific retriever type."""
    config = CircuitBreakerConfig(
        name=f"retriever_{retriever_type}",
        failure_threshold=4,
        recovery_timeout=45.0,
        request_timeout=20.0,
        performance_threshold=10.0,
        monitoring_period=300.0
    )
    return await circuit_breaker_manager.get_or_create_circuit_breaker(
        f"retriever_{retriever_type}", config
    )