import asyncio
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import weakref
from loguru import logger

from app.core.config import settings
from app.core.async_metadata_processor import async_metadata_processor


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert levels for health issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component_name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_name": self.component_name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "response_time": self.response_time,
            "error": self.error
        }


@dataclass
class HealthCheckConfig:
    """Configuration for a health check."""
    component_name: str
    check_function: Callable
    interval: float = 30.0  # seconds
    timeout: float = 10.0   # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0
    critical_threshold: int = 3  # failures before critical
    warning_threshold: int = 2   # failures before warning
    enabled: bool = True


class ComponentHealthMonitor:
    """Monitors health of a specific component."""
    
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self.last_result: Optional[HealthCheckResult] = None
        self.failure_count = 0
        self.consecutive_failures = 0
        self.last_check_time: Optional[datetime] = None
        self.check_history: List[HealthCheckResult] = []
        self.max_history = 100
        
        # Statistics
        self.total_checks = 0
        self.successful_checks = 0
        self.failed_checks = 0
        self.avg_response_time = 0.0
        self.circuit_breaker_failures = 0
        
        logger.info(f"ComponentHealthMonitor created for {config.component_name}")
    
    async def perform_check(self) -> HealthCheckResult:
        """Perform a health check with retries."""
        start_time = time.time()
        self.total_checks += 1
        self.last_check_time = datetime.now()
        
        for attempt in range(self.config.retry_attempts):
            try:
                # Execute health check with timeout
                check_task = asyncio.create_task(self.config.check_function())
                result = await asyncio.wait_for(check_task, timeout=self.config.timeout)
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Process successful result
                if isinstance(result, dict):
                    status = HealthStatus(result.get("status", "healthy"))
                    message = result.get("message", "Health check passed")
                    details = result.get("details", {})
                    error = result.get("error")
                else:
                    status = HealthStatus.HEALTHY
                    message = "Health check passed"
                    details = result if isinstance(result, dict) else {}
                    error = None
                
                # Create result
                health_result = HealthCheckResult(
                    component_name=self.config.component_name,
                    status=status,
                    message=message,
                    details=details,
                    response_time=response_time,
                    error=error
                )
                
                # Update statistics for success
                if status == HealthStatus.HEALTHY:
                    self.successful_checks += 1
                    self.consecutive_failures = 0
                    self._update_response_time(response_time)
                else:
                    self._handle_failure()
                
                self.last_result = health_result
                self._add_to_history(health_result)
                
                return health_result
                
            except asyncio.TimeoutError:
                logger.warning(f"Health check timeout for {self.config.component_name} (attempt {attempt + 1})")
                if attempt == self.config.retry_attempts - 1:
                    return self._create_failure_result("Health check timeout", "TimeoutError")
                await asyncio.sleep(self.config.retry_delay)
                
            except Exception as e:
                logger.error(f"Health check error for {self.config.component_name} (attempt {attempt + 1}): {e}")
                if attempt == self.config.retry_attempts - 1:
                    return self._create_failure_result(f"Health check failed: {str(e)}", type(e).__name__)
                await asyncio.sleep(self.config.retry_delay)
        
        # Should not reach here
        return self._create_failure_result("Health check failed after all retries", "MaxRetriesExceeded")
    
    def _create_failure_result(self, message: str, error_type: str) -> HealthCheckResult:
        """Create a failure result."""
        self._handle_failure()
        
        status = self._determine_failure_status()
        
        result = HealthCheckResult(
            component_name=self.config.component_name,
            status=status,
            message=message,
            details={
                "failure_count": self.failure_count,
                "consecutive_failures": self.consecutive_failures,
                "last_success": self._get_last_success_time()
            },
            error=error_type
        )
        
        self.last_result = result
        self._add_to_history(result)
        
        return result
    
    def _handle_failure(self):
        """Handle a check failure."""
        self.failed_checks += 1
        self.failure_count += 1
        self.consecutive_failures += 1
    
    def _determine_failure_status(self) -> HealthStatus:
        """Determine status based on failure count."""
        if self.consecutive_failures >= self.config.critical_threshold:
            return HealthStatus.CRITICAL
        elif self.consecutive_failures >= self.config.warning_threshold:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY
    
    def _update_response_time(self, response_time: float):
        """Update average response time."""
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * 0.9) + (response_time * 0.1)
    
    def _add_to_history(self, result: HealthCheckResult):
        """Add result to history."""
        self.check_history.append(result)
        if len(self.check_history) > self.max_history:
            self.check_history.pop(0)
    
    def _get_last_success_time(self) -> Optional[str]:
        """Get timestamp of last successful check."""
        for result in reversed(self.check_history):
            if result.status == HealthStatus.HEALTHY:
                return result.timestamp.isoformat()
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get component statistics."""
        uptime_percent = (self.successful_checks / self.total_checks * 100) if self.total_checks > 0 else 0
        
        return {
            "component_name": self.config.component_name,
            "total_checks": self.total_checks,
            "successful_checks": self.successful_checks,
            "failed_checks": self.failed_checks,
            "uptime_percent": uptime_percent,
            "consecutive_failures": self.consecutive_failures,
            "avg_response_time": self.avg_response_time,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "current_status": self.last_result.status.value if self.last_result else "unknown"
        }


class HealthChecker:
    """
    Comprehensive health checking system for all application components.
    
    Features:
    - Continuous monitoring of critical components
    - Configurable health checks with retries
    - Alert generation and notification
    - Health history and analytics
    - Dashboard-ready status information
    - Automatic recovery detection
    """
    
    def __init__(self):
        """Initialize the health checker."""
        self._monitors: Dict[str, ComponentHealthMonitor] = {}
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        self._is_running = False
        self._check_lock = asyncio.Lock()
        
        # Global health status
        self._global_status = HealthStatus.UNKNOWN
        self._system_metrics = {}
        
        # Alert system
        self._alert_callbacks: List[Callable] = []
        self._alert_history: List[Dict[str, Any]] = []
        self._max_alert_history = 1000
        
        logger.info("HealthChecker initialized")
    
    def register_component(self, config: HealthCheckConfig):
        """Register a component for health monitoring."""
        if config.component_name in self._monitors:
            logger.warning(f"Component {config.component_name} already registered, replacing...")
        
        monitor = ComponentHealthMonitor(config)
        self._monitors[config.component_name] = monitor
        
        logger.info(f"Registered health check for component: {config.component_name}")
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for health alerts."""
        self._alert_callbacks.append(callback)
        logger.info("Alert callback registered")
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._is_running:
            logger.warning("Health monitoring is already running")
            return
        
        self._is_running = True
        self._shutdown_event.clear()
        
        logger.info("Starting health monitoring...")
        
        # Start monitoring tasks for each component
        for component_name, monitor in self._monitors.items():
            if monitor.config.enabled:
                task = asyncio.create_task(self._monitor_component(monitor))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
        
        # Start system metrics collection
        metrics_task = asyncio.create_task(self._collect_system_metrics())
        self._background_tasks.add(metrics_task)
        metrics_task.add_done_callback(self._background_tasks.discard)
        
        # Start global status updates
        status_task = asyncio.create_task(self._update_global_status())
        self._background_tasks.add(status_task)
        status_task.add_done_callback(self._background_tasks.discard)
        
        logger.info(f"Health monitoring started for {len(self._monitors)} components")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        if not self._is_running:
            return
        
        logger.info("Stopping health monitoring...")
        
        self._is_running = False
        self._shutdown_event.set()
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        
        logger.info("Health monitoring stopped")
    
    async def _monitor_component(self, monitor: ComponentHealthMonitor):
        """Monitor a specific component continuously."""
        logger.info(f"Starting monitoring for component: {monitor.config.component_name}")
        
        while not self._shutdown_event.is_set():
            try:
                # Perform health check
                result = await monitor.perform_check()
                
                # Generate alert if needed
                await self._check_for_alerts(result, monitor)
                
                # Log health check async
                async_metadata_processor.log_async(
                    "DEBUG" if result.status == HealthStatus.HEALTHY else "WARNING",
                    f"Health check completed for {result.component_name}",
                    {
                        "component": result.component_name,
                        "status": result.status.value,
                        "response_time": result.response_time,
                        "consecutive_failures": monitor.consecutive_failures
                    }
                )
                
                # Wait for next check
                await asyncio.sleep(monitor.config.interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring for {monitor.config.component_name}: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _check_for_alerts(self, result: HealthCheckResult, monitor: ComponentHealthMonitor):
        """Check if an alert should be generated."""
        should_alert = False
        alert_level = AlertLevel.INFO
        
        # Determine if alert is needed
        if result.status == HealthStatus.CRITICAL:
            should_alert = True
            alert_level = AlertLevel.CRITICAL
        elif result.status == HealthStatus.UNHEALTHY and monitor.consecutive_failures == 1:
            should_alert = True
            alert_level = AlertLevel.ERROR
        elif result.status == HealthStatus.DEGRADED and monitor.consecutive_failures == monitor.config.warning_threshold:
            should_alert = True
            alert_level = AlertLevel.WARNING
        elif result.status == HealthStatus.HEALTHY and monitor.consecutive_failures == 0 and monitor.failure_count > 0:
            # Recovery alert
            should_alert = True
            alert_level = AlertLevel.INFO
        
        if should_alert:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "component": result.component_name,
                "level": alert_level.value,
                "status": result.status.value,
                "message": result.message,
                "details": result.details,
                "consecutive_failures": monitor.consecutive_failures,
                "response_time": result.response_time
            }
            
            await self._generate_alert(alert)
    
    async def _generate_alert(self, alert: Dict[str, Any]):
        """Generate and process an alert."""
        # Add to history
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_alert_history:
            self._alert_history.pop(0)
        
        # Log alert
        log_level = "CRITICAL" if alert["level"] == "critical" else "WARNING"
        async_metadata_processor.log_async(
            log_level,
            f"Health alert: {alert['component']} - {alert['message']}",
            alert,
            priority=3 if alert["level"] == "critical" else 2
        )
        
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                await callback(alert) if asyncio.iscoroutinefunction(callback) else callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        while not self._shutdown_event.is_set():
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self._system_metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                }
                
                # Record metrics async
                async_metadata_processor.record_metric_async(
                    "system_cpu_percent",
                    cpu_percent,
                    {},
                    "gauge"
                )
                
                async_metadata_processor.record_metric_async(
                    "system_memory_percent",
                    memory.percent,
                    {},
                    "gauge"
                )
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)
    
    async def _update_global_status(self):
        """Update global health status based on all components."""
        while not self._shutdown_event.is_set():
            try:
                async with self._check_lock:
                    statuses = []
                    for monitor in self._monitors.values():
                        if monitor.last_result:
                            statuses.append(monitor.last_result.status)
                    
                    # Determine global status
                    if not statuses:
                        self._global_status = HealthStatus.UNKNOWN
                    elif any(s == HealthStatus.CRITICAL for s in statuses):
                        self._global_status = HealthStatus.CRITICAL
                    elif any(s == HealthStatus.UNHEALTHY for s in statuses):
                        self._global_status = HealthStatus.UNHEALTHY
                    elif any(s == HealthStatus.DEGRADED for s in statuses):
                        self._global_status = HealthStatus.DEGRADED
                    else:
                        self._global_status = HealthStatus.HEALTHY
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error updating global status: {e}")
                await asyncio.sleep(30)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        async with self._check_lock:
            component_statuses = {}
            
            for name, monitor in self._monitors.items():
                if monitor.last_result:
                    component_statuses[name] = monitor.last_result.to_dict()
                else:
                    component_statuses[name] = {
                        "component_name": name,
                        "status": "unknown",
                        "message": "No health check performed yet"
                    }
        
        health_response = {
            "global_status": self._global_status.value,
            "timestamp": datetime.now().isoformat(),
            "components": component_statuses,
            "system_metrics": self._system_metrics,
            "statistics": {
                "total_components": len(self._monitors),
                "monitoring_active": self._is_running,
                "total_alerts": len(self._alert_history)
            }
        }
        
        return health_response
    
    async def get_component_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics for all components."""
        stats = {}
        
        for name, monitor in self._monitors.items():
            stats[name] = monitor.get_statistics()
        
        return {
            "component_statistics": stats,
            "global_statistics": {
                "total_components": len(self._monitors),
                "active_monitoring": self._is_running,
                "global_status": self._global_status.value
            }
        }
    
    async def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        return self._alert_history[-limit:] if self._alert_history else []
    
    async def force_check(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """Force an immediate health check."""
        if component_name:
            if component_name in self._monitors:
                monitor = self._monitors[component_name]
                result = await monitor.perform_check()
                return {component_name: result.to_dict()}
            else:
                raise ValueError(f"Component {component_name} not found")
        else:
            # Check all components
            results = {}
            for name, monitor in self._monitors.items():
                result = await monitor.perform_check()
                results[name] = result.to_dict()
            return results


# Global health checker instance
health_checker = HealthChecker()


# Predefined health check functions for common components
async def check_vector_store_health() -> Dict[str, Any]:
    """Health check for vector store database connections."""
    try:
        from app.models.vector_store import vector_store_manager
        health_status = await vector_store_manager.get_health_status()
        
        return {
            "status": "healthy" if health_status["health_status"]["is_healthy"] else "unhealthy",
            "message": "Vector store database connection healthy" if health_status["health_status"]["is_healthy"] else "Vector store database connection issues",
            "details": health_status
        }
    except Exception as e:
        return {
            "status": "critical",
            "message": f"Vector store database health check failed: {str(e)}",
            "error": type(e).__name__
        }


async def check_embedding_manager_health() -> Dict[str, Any]:
    """Health check for embedding model manager and embedding operations."""
    try:
        from app.core.embedding_manager import embedding_manager
        health_status = await embedding_manager.get_health_status()
        
        return {
            "status": "healthy" if health_status["startup_completed"] else "degraded",
            "message": "Embedding model manager operational" if health_status["startup_completed"] else "Embedding model manager initializing",
            "details": health_status
        }
    except Exception as e:
        return {
            "status": "critical",
            "message": f"Embedding model health check failed: {str(e)}",
            "error": type(e).__name__
        }


async def check_persistent_rag_service_health() -> Dict[str, Any]:
    """Health check for persistent RAG service."""
    try:
        # This would be called from main app with access to app.state
        # For now, return a basic check
        return {
            "status": "healthy",
            "message": "Persistent RAG service health check placeholder",
            "details": {}
        }
    except Exception as e:
        return {
            "status": "critical",
            "message": f"Persistent RAG service health check failed: {str(e)}",
            "error": type(e).__name__
        }


def setup_default_health_checks():
    """Setup default health checks for common components."""
    # Vector Store Health Check
    health_checker.register_component(HealthCheckConfig(
        component_name="vector_store",
        check_function=check_vector_store_health,
        interval=30.0,
        timeout=10.0,
        retry_attempts=3,
        critical_threshold=3,
        warning_threshold=2
    ))
    
    # Embedding Manager Health Check
    health_checker.register_component(HealthCheckConfig(
        component_name="embedding_manager",
        check_function=check_embedding_manager_health,
        interval=45.0,
        timeout=15.0,
        retry_attempts=2,
        critical_threshold=3,
        warning_threshold=2
    ))
    
    # Persistent RAG Service Health Check
    health_checker.register_component(HealthCheckConfig(
        component_name="persistent_rag_service",
        check_function=check_persistent_rag_service_health,
        interval=60.0,
        timeout=20.0,
        retry_attempts=2,
        critical_threshold=2,
        warning_threshold=1
    ))
    
    logger.info("Default health checks configured")