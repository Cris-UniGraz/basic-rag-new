"""
Comprehensive observability system for RAG API.

This module provides:
- Prometheus metrics integration
- Distributed tracing with OpenTelemetry
- Structured logging
- Custom metrics for retrievers and performance
- Alerting integration
- Health monitoring dashboards
"""

import asyncio
import time
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
import threading
from collections import defaultdict, deque
import uuid
from pathlib import Path

from loguru import logger
import structlog
from prometheus_client import (
    Counter, Histogram, Gauge, Info, Enum as PrometheusEnum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    start_http_server, push_to_gateway, delete_from_gateway
)

from app.core.config import settings
from app.core.async_metadata_processor import async_metadata_processor


class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    INFO = "info"
    ENUM = "enum"


class TraceLevel(Enum):
    """Trace severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricConfig:
    """Configuration for a metric."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    states: Optional[List[str]] = None  # For enums


@dataclass
class TraceSpan:
    """Represents a trace span."""
    span_id: str
    trace_id: str
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    parent_span_id: Optional[str] = None


class PrometheusMetrics:
    """
    Prometheus metrics collector for RAG API.
    
    Provides comprehensive metrics for:
    - API performance
    - Retriever operations
    - System resources
    - Cache performance
    - Background tasks
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus metrics."""
        self.registry = registry or CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._metric_configs = self._define_metrics()
        self._initialize_metrics()
        
        logger.info("Prometheus metrics initialized")
    
    def _define_metrics(self) -> List[MetricConfig]:
        """Define all metrics configurations."""
        return [
            # API Metrics
            MetricConfig(
                name="rag_api_requests_total",
                description="Total number of API requests",
                metric_type=MetricType.COUNTER,
                labels=["method", "endpoint", "status_code"]
            ),
            MetricConfig(
                name="rag_api_request_duration_seconds",
                description="API request duration in seconds",
                metric_type=MetricType.HISTOGRAM,
                labels=["method", "endpoint"],
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
            ),
            MetricConfig(
                name="rag_api_active_requests",
                description="Number of active API requests",
                metric_type=MetricType.GAUGE,
                labels=["endpoint"]
            ),
            
            # Retriever Metrics
            MetricConfig(
                name="rag_retriever_operations_total",
                description="Total number of retriever operations",
                metric_type=MetricType.COUNTER,
                labels=["collection", "retriever_type", "status"]
            ),
            MetricConfig(
                name="rag_retriever_duration_seconds",
                description="Retriever operation duration in seconds",
                metric_type=MetricType.HISTOGRAM,
                labels=["collection", "retriever_type"],
                buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            ),
            MetricConfig(
                name="rag_retriever_cache_hits_total",
                description="Total number of retriever cache hits",
                metric_type=MetricType.COUNTER,
                labels=["collection", "cache_level"]
            ),
            MetricConfig(
                name="rag_retriever_active_count",
                description="Number of active retrievers",
                metric_type=MetricType.GAUGE,
                labels=["collection"]
            ),
            
            # System Metrics
            MetricConfig(
                name="rag_system_cpu_usage_percent",
                description="System CPU usage percentage",
                metric_type=MetricType.GAUGE
            ),
            MetricConfig(
                name="rag_system_memory_usage_bytes",
                description="System memory usage in bytes",
                metric_type=MetricType.GAUGE,
                labels=["type"]  # total, available, used, free
            ),
            MetricConfig(
                name="rag_system_disk_usage_bytes",
                description="System disk usage in bytes",
                metric_type=MetricType.GAUGE,
                labels=["type", "path"]  # total, used, free
            ),
            
            # Connection Pool Metrics
            MetricConfig(
                name="rag_connection_pool_active",
                description="Active connections in pool",
                metric_type=MetricType.GAUGE,
                labels=["pool_name", "service_type"]
            ),
            MetricConfig(
                name="rag_connection_pool_total",
                description="Total connections in pool",
                metric_type=MetricType.GAUGE,
                labels=["pool_name", "service_type"]
            ),
            MetricConfig(
                name="rag_connection_pool_operations_total",
                description="Total connection pool operations",
                metric_type=MetricType.COUNTER,
                labels=["pool_name", "operation", "status"]
            ),
            
            # Cache Metrics
            MetricConfig(
                name="rag_cache_operations_total",
                description="Total cache operations",
                metric_type=MetricType.COUNTER,
                labels=["cache_level", "operation", "status"]
            ),
            MetricConfig(
                name="rag_cache_hit_rate",
                description="Cache hit rate",
                metric_type=MetricType.GAUGE,
                labels=["cache_level"]
            ),
            MetricConfig(
                name="rag_cache_size_bytes",
                description="Cache size in bytes",
                metric_type=MetricType.GAUGE,
                labels=["cache_level"]
            ),
            
            # Background Task Metrics
            MetricConfig(
                name="rag_background_tasks_total",
                description="Total background tasks executed",
                metric_type=MetricType.COUNTER,
                labels=["task_type", "status"]
            ),
            MetricConfig(
                name="rag_background_task_duration_seconds",
                description="Background task duration in seconds",
                metric_type=MetricType.HISTOGRAM,
                labels=["task_type"],
                buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0]
            ),
            MetricConfig(
                name="rag_background_tasks_active",
                description="Number of active background tasks",
                metric_type=MetricType.GAUGE
            ),
            
            # Application Info
            MetricConfig(
                name="rag_application_info",
                description="Application information",
                metric_type=MetricType.INFO,
                labels=["version", "environment", "build_date"]
            ),
        ]
    
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics."""
        for config in self._metric_configs:
            if config.metric_type == MetricType.COUNTER:
                metric = Counter(
                    config.name,
                    config.description,
                    config.labels,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.HISTOGRAM:
                metric = Histogram(
                    config.name,
                    config.description,
                    config.labels,
                    buckets=config.buckets,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    config.name,
                    config.description,
                    config.labels,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.INFO:
                metric = Info(
                    config.name,
                    config.description,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.ENUM:
                metric = PrometheusEnum(
                    config.name,
                    config.description,
                    config.labels,
                    states=config.states,
                    registry=self.registry
                )
            
            self._metrics[config.name] = metric
    
    def get_metric(self, name: str):
        """Get a metric by name."""
        return self._metrics.get(name)
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1):
        """Increment a counter metric."""
        metric = self.get_metric(name)
        if metric and labels:
            metric.labels(**labels).inc(value)
        elif metric:
            metric.inc(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value in a histogram metric."""
        metric = self.get_metric(name)
        if metric and labels:
            metric.labels(**labels).observe(value)
        elif metric:
            metric.observe(value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        metric = self.get_metric(name)
        if metric and labels:
            metric.labels(**labels).set(value)
        elif metric:
            metric.set(value)
    
    def set_info(self, name: str, info: Dict[str, str]):
        """Set info metric."""
        metric = self.get_metric(name)
        if metric:
            metric.info(info)
    
    def collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("rag_system_cpu_usage_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.set_gauge("rag_system_memory_usage_bytes", memory.total, {"type": "total"})
            self.set_gauge("rag_system_memory_usage_bytes", memory.available, {"type": "available"})
            self.set_gauge("rag_system_memory_usage_bytes", memory.used, {"type": "used"})
            self.set_gauge("rag_system_memory_usage_bytes", memory.free, {"type": "free"})
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.set_gauge("rag_system_disk_usage_bytes", disk.total, {"type": "total", "path": "/"})
            self.set_gauge("rag_system_disk_usage_bytes", disk.used, {"type": "used", "path": "/"})
            self.set_gauge("rag_system_disk_usage_bytes", disk.free, {"type": "free", "path": "/"})
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class DistributedTracing:
    """
    Distributed tracing system for request tracking.
    
    Provides OpenTelemetry-compatible tracing for:
    - Request flow tracking
    - Performance analysis
    - Error tracking
    - Service dependencies
    """
    
    def __init__(self):
        """Initialize distributed tracing."""
        self._spans: Dict[str, TraceSpan] = {}
        self._active_traces: Dict[str, str] = {}  # thread_id -> trace_id
        self._trace_storage = deque(maxlen=10000)
        self._lock = threading.Lock()
        
        logger.info("Distributed tracing initialized")
    
    def start_trace(self, operation_name: str, trace_id: Optional[str] = None) -> str:
        """Start a new trace."""
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        thread_id = threading.get_ident()
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            operation_name=operation_name,
            start_time=datetime.now()
        )
        
        with self._lock:
            self._spans[span_id] = span
            self._active_traces[thread_id] = trace_id
        
        return span_id
    
    def start_span(self, operation_name: str, parent_span_id: Optional[str] = None) -> str:
        """Start a new span within a trace."""
        thread_id = threading.get_ident()
        
        with self._lock:
            trace_id = self._active_traces.get(thread_id)
            if not trace_id and parent_span_id:
                parent_span = self._spans.get(parent_span_id)
                trace_id = parent_span.trace_id if parent_span else str(uuid.uuid4())
            elif not trace_id:
                trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            parent_span_id=parent_span_id
        )
        
        with self._lock:
            self._spans[span_id] = span
            self._active_traces[thread_id] = trace_id
        
        return span_id
    
    def finish_span(self, span_id: str, status: str = "success", tags: Optional[Dict[str, Any]] = None):
        """Finish a span."""
        with self._lock:
            span = self._spans.get(span_id)
            if span:
                span.end_time = datetime.now()
                span.duration = (span.end_time - span.start_time).total_seconds()
                span.status = status
                if tags:
                    span.tags.update(tags)
                
                # Store completed span
                self._trace_storage.append(span)
    
    def add_span_log(self, span_id: str, level: TraceLevel, message: str, data: Optional[Dict[str, Any]] = None):
        """Add a log entry to a span."""
        with self._lock:
            span = self._spans.get(span_id)
            if span:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": level.value,
                    "message": message,
                    "data": data or {}
                }
                span.logs.append(log_entry)
    
    def add_span_tag(self, span_id: str, key: str, value: Any):
        """Add a tag to a span."""
        with self._lock:
            span = self._spans.get(span_id)
            if span:
                span.tags[key] = value
    
    @contextmanager
    def trace_operation(self, operation_name: str, parent_span_id: Optional[str] = None):
        """Context manager for tracing operations."""
        span_id = self.start_span(operation_name, parent_span_id)
        try:
            yield span_id
            self.finish_span(span_id, "success")
        except Exception as e:
            self.add_span_tag(span_id, "error", True)
            self.add_span_tag(span_id, "error.message", str(e))
            self.add_span_log(span_id, TraceLevel.ERROR, f"Operation failed: {e}")
            self.finish_span(span_id, "error")
            raise
    
    def get_trace_data(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        return [span for span in self._trace_storage if span.trace_id == trace_id]
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get a summary of a trace."""
        spans = self.get_trace_data(trace_id)
        if not spans:
            return {}
        
        total_duration = max(span.duration for span in spans if span.duration)
        span_count = len(spans)
        error_count = len([span for span in spans if span.status == "error"])
        
        return {
            "trace_id": trace_id,
            "total_duration": total_duration,
            "span_count": span_count,
            "error_count": error_count,
            "success_rate": (span_count - error_count) / span_count if span_count > 0 else 0,
            "spans": [
                {
                    "span_id": span.span_id,
                    "operation_name": span.operation_name,
                    "duration": span.duration,
                    "status": span.status,
                    "tags": span.tags
                }
                for span in spans
            ]
        }


class StructuredLogger:
    """
    Structured logging system with JSON output.
    
    Provides:
    - Structured JSON logging
    - Context preservation
    - Performance tracking
    - Error correlation
    """
    
    def __init__(self):
        """Initialize structured logger."""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger()
        logger.info("Structured logging initialized")
    
    def log_request(self, method: str, path: str, status_code: int, duration: float, 
                   request_id: Optional[str] = None, user_id: Optional[str] = None):
        """Log HTTP request."""
        self.logger.info(
            "http_request",
            method=method,
            path=path,
            status_code=status_code,
            duration_seconds=duration,
            request_id=request_id,
            user_id=user_id,
            event_type="http_request"
        )
    
    def log_retriever_operation(self, collection: str, retriever_type: str, 
                              query: str, duration: float, result_count: int,
                              cache_hit: bool = False, span_id: Optional[str] = None):
        """Log retriever operation."""
        self.logger.info(
            "retriever_operation",
            collection=collection,
            retriever_type=retriever_type,
            query_length=len(query),
            duration_seconds=duration,
            result_count=result_count,
            cache_hit=cache_hit,
            span_id=span_id,
            event_type="retriever_operation"
        )
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                 span_id: Optional[str] = None):
        """Log error with context."""
        self.logger.error(
            "application_error",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            span_id=span_id,
            event_type="error"
        )
    
    def log_performance(self, operation: str, duration: float, success: bool,
                       metadata: Optional[Dict[str, Any]] = None,
                       span_id: Optional[str] = None):
        """Log performance metrics."""
        self.logger.info(
            "performance_metric",
            operation=operation,
            duration_seconds=duration,
            success=success,
            metadata=metadata or {},
            span_id=span_id,
            event_type="performance"
        )


class AlertManager:
    """
    Alert management system for monitoring.
    
    Provides:
    - Threshold-based alerting
    - Multiple notification channels
    - Alert aggregation and deduplication
    - Escalation policies
    """
    
    def __init__(self):
        """Initialize alert manager."""
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._active_alerts: Dict[str, Dict[str, Any]] = {}
        self._alert_history = deque(maxlen=1000)
        self._notification_channels: List[Callable] = []
        
        self._setup_default_rules()
        logger.info("Alert manager initialized")
    
    def _setup_default_rules(self):
        """Setup default alerting rules."""
        # High error rate
        self.add_rule(
            "high_error_rate",
            threshold=0.05,  # 5% error rate
            comparison="gt",
            duration=300,  # 5 minutes
            severity="warning",
            message="High error rate detected: {value:.2%}"
        )
        
        # High response time
        self.add_rule(
            "high_response_time",
            threshold=10.0,  # 10 seconds
            comparison="gt",
            duration=300,
            severity="warning",
            message="High response time detected: {value:.2f}s"
        )
        
        # System resource alerts
        self.add_rule(
            "high_cpu_usage",
            threshold=85.0,  # 85% CPU
            comparison="gt",
            duration=300,
            severity="warning",
            message="High CPU usage: {value:.1f}%"
        )
        
        self.add_rule(
            "high_memory_usage",
            threshold=90.0,  # 90% memory
            comparison="gt",
            duration=300,
            severity="critical",
            message="High memory usage: {value:.1f}%"
        )
    
    def add_rule(self, name: str, threshold: float, comparison: str,
                duration: int, severity: str, message: str):
        """Add an alert rule."""
        self._alert_rules[name] = {
            "threshold": threshold,
            "comparison": comparison,
            "duration": duration,
            "severity": severity,
            "message": message,
            "triggered_at": None
        }
    
    def check_metric(self, rule_name: str, value: float):
        """Check a metric against alert rules."""
        rule = self._alert_rules.get(rule_name)
        if not rule:
            return
        
        triggered = False
        if rule["comparison"] == "gt" and value > rule["threshold"]:
            triggered = True
        elif rule["comparison"] == "lt" and value < rule["threshold"]:
            triggered = True
        elif rule["comparison"] == "eq" and value == rule["threshold"]:
            triggered = True
        
        current_time = datetime.now()
        
        if triggered:
            if rule["triggered_at"] is None:
                rule["triggered_at"] = current_time
            elif (current_time - rule["triggered_at"]).total_seconds() >= rule["duration"]:
                self._fire_alert(rule_name, value, rule)
        else:
            if rule["triggered_at"] is not None:
                self._resolve_alert(rule_name)
            rule["triggered_at"] = None
    
    def _fire_alert(self, rule_name: str, value: float, rule: Dict[str, Any]):
        """Fire an alert."""
        alert_id = f"{rule_name}_{int(time.time())}"
        
        alert = {
            "id": alert_id,
            "rule_name": rule_name,
            "severity": rule["severity"],
            "message": rule["message"].format(value=value),
            "value": value,
            "threshold": rule["threshold"],
            "fired_at": datetime.now(),
            "status": "firing"
        }
        
        self._active_alerts[rule_name] = alert
        self._alert_history.append(alert.copy())
        
        # Send notifications
        for channel in self._notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")
        
        logger.warning(f"Alert fired: {alert['message']}")
    
    def _resolve_alert(self, rule_name: str):
        """Resolve an active alert."""
        if rule_name in self._active_alerts:
            alert = self._active_alerts[rule_name]
            alert["status"] = "resolved"
            alert["resolved_at"] = datetime.now()
            
            self._alert_history.append(alert.copy())
            del self._active_alerts[rule_name]
            
            logger.info(f"Alert resolved: {rule_name}")
    
    def add_notification_channel(self, channel: Callable):
        """Add a notification channel."""
        self._notification_channels.append(channel)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return list(self._active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history."""
        return list(self._alert_history)[-limit:]


class ObservabilityManager:
    """
    Main observability manager that coordinates all observability components.
    
    Integrates:
    - Prometheus metrics
    - Distributed tracing
    - Structured logging
    - Alert management
    - Health monitoring
    """
    
    def __init__(self):
        """Initialize observability manager."""
        self.metrics = PrometheusMetrics()
        self.tracing = DistributedTracing()
        self.structured_logger = StructuredLogger()
        self.alert_manager = AlertManager()
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._is_running = False
        self._shutdown_event = asyncio.Event()
        
        # Setup application info
        self._setup_application_info()
        
        logger.info("Observability manager initialized")
    
    def _setup_application_info(self):
        """Setup application information metrics."""
        self.metrics.set_info("rag_application_info", {
            "version": getattr(settings, "VERSION", "unknown"),
            "environment": settings.ENVIRONMENT,
            "build_date": getattr(settings, "BUILD_DATE", "unknown")
        })
    
    async def start(self):
        """Start observability services."""
        if self._is_running:
            return
        
        self._is_running = True
        self._shutdown_event.clear()
        
        # Start metrics collection
        if settings.METRICS_EXPORT_ENABLED:
            metrics_task = asyncio.create_task(self._metrics_collection_loop())
            self._background_tasks.add(metrics_task)
            metrics_task.add_done_callback(self._background_tasks.discard)
        
        # Start Prometheus HTTP server
        if settings.PROMETHEUS_ENABLED:
            try:
                start_http_server(settings.PROMETHEUS_PORT, registry=self.metrics.registry)
                logger.info(f"Prometheus metrics server started on port {settings.PROMETHEUS_PORT}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus server: {e}")
        
        # Start alert checking
        alert_task = asyncio.create_task(self._alert_checking_loop())
        self._background_tasks.add(alert_task)
        alert_task.add_done_callback(self._background_tasks.discard)
        
        logger.info("Observability services started")
    
    async def stop(self):
        """Stop observability services."""
        if not self._is_running:
            return
        
        self._is_running = False
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        logger.info("Observability services stopped")
    
    async def _metrics_collection_loop(self):
        """Background loop for metrics collection."""
        while not self._shutdown_event.is_set():
            try:
                # Collect system metrics
                self.metrics.collect_system_metrics()
                
                # Collect application metrics from other components
                await self._collect_application_metrics()
                
                await asyncio.sleep(settings.METRICS_EXPORT_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            # Connection pool metrics
            try:
                from app.core.connection_pools import connection_pool_manager
                stats = await connection_pool_manager.get_global_stats()
                
                for pool_name, pool_stats in stats.get("pool_stats", {}).items():
                    service_type = pool_name.replace("_pool", "")
                    
                    self.metrics.set_gauge(
                        "rag_connection_pool_active",
                        pool_stats["metrics"]["active_connections"],
                        {"pool_name": pool_name, "service_type": service_type}
                    )
                    
                    self.metrics.set_gauge(
                        "rag_connection_pool_total",
                        pool_stats["size_config"]["current_size"],
                        {"pool_name": pool_name, "service_type": service_type}
                    )
            except Exception as e:
                logger.debug(f"Could not collect connection pool metrics: {e}")
            
            # Cache metrics
            try:
                from app.core.advanced_cache import multi_level_cache
                cache_stats = multi_level_cache.get_global_stats()
                
                for level, stats in cache_stats["multi_level_cache"]["levels"].items():
                    hit_rate = stats["stats"]["hit_rate"]
                    size_bytes = stats["stats"]["size_bytes"]
                    
                    self.metrics.set_gauge("rag_cache_hit_rate", hit_rate, {"cache_level": level})
                    self.metrics.set_gauge("rag_cache_size_bytes", size_bytes, {"cache_level": level})
            except Exception as e:
                logger.debug(f"Could not collect cache metrics: {e}")
            
            # Background task metrics
            try:
                from app.core.background_tasks import background_task_manager
                task_stats = background_task_manager.get_task_stats()
                
                self.metrics.set_gauge(
                    "rag_background_tasks_active",
                    task_stats["manager_status"]["currently_running_tasks"]
                )
            except Exception as e:
                logger.debug(f"Could not collect background task metrics: {e}")
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    async def _alert_checking_loop(self):
        """Background loop for checking alerts."""
        while not self._shutdown_event.is_set():
            try:
                # Check system metrics against alert rules
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                self.alert_manager.check_metric("high_cpu_usage", cpu_percent)
                self.alert_manager.check_metric("high_memory_usage", memory_percent)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in alert checking loop: {e}")
                await asyncio.sleep(60)
    
    # Context managers for operation tracking
    @asynccontextmanager
    async def trace_request(self, method: str, path: str, request_id: Optional[str] = None):
        """Trace an HTTP request."""
        span_id = self.tracing.start_trace(f"{method} {path}")
        start_time = time.time()
        
        # Increment active requests
        self.metrics.increment_counter("rag_api_active_requests", {"endpoint": path})
        
        try:
            yield span_id
            
            duration = time.time() - start_time
            status_code = 200  # Default success
            
            # Record metrics
            self.metrics.increment_counter(
                "rag_api_requests_total",
                {"method": method, "endpoint": path, "status_code": str(status_code)}
            )
            self.metrics.observe_histogram(
                "rag_api_request_duration_seconds",
                duration,
                {"method": method, "endpoint": path}
            )
            
            # Log request
            self.structured_logger.log_request(method, path, status_code, duration, request_id)
            
            # Finish trace
            self.tracing.finish_span(span_id, "success")
            
        except Exception as e:
            duration = time.time() - start_time
            status_code = 500  # Error
            
            # Record error metrics
            self.metrics.increment_counter(
                "rag_api_requests_total",
                {"method": method, "endpoint": path, "status_code": str(status_code)}
            )
            
            # Log error
            self.structured_logger.log_error(e, {"method": method, "path": path}, span_id)
            
            # Finish trace with error
            self.tracing.finish_span(span_id, "error", {"error": str(e)})
            
            raise
        
        finally:
            # Decrement active requests
            self.metrics.set_gauge("rag_api_active_requests", -1, {"endpoint": path})
    
    @contextmanager
    def trace_retriever_operation(self, collection: str, retriever_type: str, query: str):
        """Trace a retriever operation."""
        span_id = self.tracing.start_span(f"retriever_{retriever_type}")
        start_time = time.time()
        
        try:
            yield span_id
            
            duration = time.time() - start_time
            
            # Record metrics
            self.metrics.increment_counter(
                "rag_retriever_operations_total",
                {"collection": collection, "retriever_type": retriever_type, "status": "success"}
            )
            self.metrics.observe_histogram(
                "rag_retriever_duration_seconds",
                duration,
                {"collection": collection, "retriever_type": retriever_type}
            )
            
            # Log operation
            self.structured_logger.log_retriever_operation(
                collection, retriever_type, query, duration, 0, span_id=span_id
            )
            
            # Finish trace
            self.tracing.finish_span(span_id, "success")
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record error metrics
            self.metrics.increment_counter(
                "rag_retriever_operations_total",
                {"collection": collection, "retriever_type": retriever_type, "status": "error"}
            )
            
            # Log error
            self.structured_logger.log_error(e, {
                "collection": collection,
                "retriever_type": retriever_type,
                "query_length": len(query)
            }, span_id)
            
            # Finish trace with error
            self.tracing.finish_span(span_id, "error", {"error": str(e)})
            
            raise
    
    def get_metrics_data(self) -> bytes:
        """Get Prometheus metrics data."""
        return generate_latest(self.metrics.registry)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "observability": {
                "metrics_enabled": settings.METRICS_EXPORT_ENABLED,
                "tracing_enabled": settings.TRACING_ENABLED,
                "structured_logging": settings.STRUCTURED_LOGGING_ENABLED,
                "prometheus_enabled": settings.PROMETHEUS_ENABLED,
                "active_alerts": len(self.alert_manager.get_active_alerts()),
                "is_running": self._is_running
            },
            "active_alerts": self.alert_manager.get_active_alerts(),
            "system_metrics": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        }


# Global observability manager
observability_manager = ObservabilityManager()


# Utility functions for easy access
async def start_observability():
    """Start observability services."""
    if settings.OBSERVABILITY_ENABLED:
        await observability_manager.start()
        logger.info("Observability services started")


async def stop_observability():
    """Stop observability services."""
    await observability_manager.stop()


def get_metrics_data() -> bytes:
    """Get Prometheus metrics data."""
    return observability_manager.get_metrics_data()


def get_observability_health() -> Dict[str, Any]:
    """Get observability health status."""
    return observability_manager.get_health_status()


# Decorators for easy instrumentation
def trace_operation(operation_name: str):
    """Decorator to trace function operations."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with observability_manager.tracing.trace_operation(operation_name) as span_id:
                    observability_manager.tracing.add_span_tag(span_id, "function", func.__name__)
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with observability_manager.tracing.trace_operation(operation_name) as span_id:
                    observability_manager.tracing.add_span_tag(span_id, "function", func.__name__)
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator


def measure_performance(metric_name: str):
    """Decorator to measure function performance."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    observability_manager.structured_logger.log_performance(
                        metric_name, duration, True, {"function": func.__name__}
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    observability_manager.structured_logger.log_performance(
                        metric_name, duration, False, {"function": func.__name__, "error": str(e)}
                    )
                    raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    observability_manager.structured_logger.log_performance(
                        metric_name, duration, True, {"function": func.__name__}
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    observability_manager.structured_logger.log_performance(
                        metric_name, duration, False, {"function": func.__name__, "error": str(e)}
                    )
                    raise
            return sync_wrapper
    return decorator