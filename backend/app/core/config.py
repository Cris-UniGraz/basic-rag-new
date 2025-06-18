from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    PROJECT_NAME: str = Field(default="RAG API")
    VERSION: str = Field(default="0.1.0")
    ENABLE_CORS: bool = Field(default=True)
    ALLOWED_ORIGINS: List[str] = Field(default=["http://localhost:8501", "http://frontend:8501"])

    # Azure LLM Configuration
    AZURE_LLM_MODEL: str

    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_API_LLM_DEPLOYMENT_ID: str
    AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID: str
    AZURE_OPENAI_LLM_MODEL: str
    AZURE_OPENAI_EMBEDDING_MODEL: str
    
    # Azure Meta Configuration
    AZURE_META_API_KEY: str
    AZURE_META_ENDPOINT: str
    AZURE_META_API_VERSION: str
    AZURE_META_API_LLM_DEPLOYMENT_ID: str
    AZURE_META_LLM_MODEL: str
    
    # Azure Cohere API
    AZURE_COHERE_ENDPOINT: str
    AZURE_COHERE_API_KEY: str

    # Embedding Models Configuration
    EMBEDDING_MODEL_NAME: str

    # Chunking Parameters
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=16)
    PARENT_CHUNK_SIZE: int = Field(default=4096)
    PARENT_CHUNK_OVERLAP: int = Field(default=32)
    PAGE_OVERLAP: int = Field(default=16)

    # Reranking Configuration
    RERANKING_TYPE: str = Field(default="cohere")
    COHERE_RERANKING_MODEL: str
    MIN_RERANKING_SCORE: float = Field(default=0.2)
    
    # Retriever Weights Configuration
    # Pesos para los diferentes retrievers en el ensemble
    RETRIEVER_WEIGHTS_BASE: float = Field(default=0.1)  # Base vectorial retriever
    RETRIEVER_WEIGHTS_PARENT: float = Field(default=0.3)  # Parent document retriever  
    RETRIEVER_WEIGHTS_MULTI_QUERY: float = Field(default=0.4)  # Multi-query retriever
    RETRIEVER_WEIGHTS_HYDE: float = Field(default=0.1)  # HyDE retriever
    RETRIEVER_WEIGHTS_BM25: float = Field(default=0.1)  # BM25 retriever

    # Collection and Document Paths
    COLLECTION_NAME: str
    SOURCES_PATH: str = Field(default="/app/data/documents")
    MAX_CHUNKS_CONSIDERED: int = Field(default=10)
    MAX_CHUNKS_LLM: int = Field(default=6)

    # MongoDB Configuration
    MONGODB_CONNECTION_STRING: str
    MONGODB_DATABASE_NAME: str

    # Logging and Debugging
    LOG_LEVEL: str = Field(default="INFO")
    SHOW_INTERNAL_MESSAGES: bool = Field(default=False)

    # Cache Settings
    CACHE_TTL: int = Field(default=3600)  # 1 hour in seconds
    ENABLE_CACHE: bool = Field(default=True)
    REDIS_URL: str = Field(default="redis://redis:6379/0")
    
    # Advanced Cache Settings
    ADVANCED_CACHE_ENABLED: bool = Field(default=True)
    ADVANCED_CACHE_MAX_SIZE: int = Field(default=1000)  # Máximo número de entradas en caché
    ADVANCED_CACHE_TTL_HOURS: int = Field(default=24)  # Tiempo de vida en horas
    ADVANCED_CACHE_SIMILARITY_THRESHOLD: float = Field(default=0.85)  # Umbral para considerar consultas similares
    
    # Query Optimization Settings
    QUERY_OPTIMIZATION_ENABLED: bool = Field(default=True)
    QUERY_SIMILARITY_THRESHOLD: float = Field(default=0.85)  # Umbral de similitud para consultas (reducido de 1.0 a 0.85)
    QUERY_HISTORY_SIZE: int = Field(default=100)  # Tamaño del historial de consultas
    APPLY_QUERY_REWRITING: bool = Field(default=True)  # Reescribir consultas para mayor precisión
    SEMANTIC_CACHING_ENABLED: bool = Field(default=True)  # Caché basado en similitud semántica

    # System Parameters
    USER_AGENT: str = Field(default="rag_assistant")

    # Resource Management
    MAX_CONCURRENT_TASKS: int = Field(default=5)  # Increased from 3 to 5
    TASK_TIMEOUT: int = Field(default=60)  # Seconds
    MAX_RETRIES: int = Field(default=3)
    RETRY_BACKOFF: float = Field(default=1.5)  # Exponential backoff factor
    
    # Enhanced Coroutine Management
    STORE_TASK_HISTORY: bool = Field(default=True)  # Keep history of completed tasks
    TASK_HISTORY_SIZE: int = Field(default=1000)  # Maximum number of historical tasks to store
    DEFAULT_SUPPRESS_ERRORS: bool = Field(default=False)  # Default behavior for error handling
    PARALLEL_EXECUTION_CHUNK_SIZE: int = Field(default=10)  # Chunk size for parallel processing
    
    # Vector Store Management
    DONT_KEEP_COLLECTIONS: bool = Field(default=False)
    
    # Advanced Pipeline Configuration
    ASYNC_PIPELINE_PHASE_LOGGING: bool = Field(default=True)  # Log detailed phase timings
    ASYNC_PIPELINE_PARALLEL_LIMIT: int = Field(default=10)  # Max parallel tasks in pipeline
    
    # Timeout Configuration
    CHAT_REQUEST_TIMEOUT: int = Field(default=180)  # 3 minutes timeout for chat requests
    RETRIEVAL_TASK_TIMEOUT: int = Field(default=90)  # 30 seconds timeout for individual retrieval tasks
    LLM_GENERATION_TIMEOUT: int = Field(default=120)  # 2 minutes timeout for LLM generation
    
    # Persistent RAG Service Configuration
    RETRIEVER_CACHE_TTL: int = Field(default=3600)  # 1 hour TTL for cached retrievers
    MAX_RETRIEVER_ERRORS: int = Field(default=5)  # Max errors before marking retriever unhealthy
    HEALTH_CHECK_INTERVAL: int = Field(default=300)  # 5 minutes between health checks
    STARTUP_TIMEOUT: int = Field(default=300)  # 5 minutes timeout for startup initialization
    
    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_THRESHOLD: int = Field(default=5)  # Failures before opening circuit
    CIRCUIT_BREAKER_TIMEOUT: int = Field(default=300)  # 5 minutes before trying recovery
    
    # Connection Pooling Configuration
    AZURE_OPENAI_MAX_CONNECTIONS: int = Field(default=10)  # Max connections to Azure OpenAI
    AZURE_OPENAI_RATE_LIMIT: int = Field(default=1000)  # Requests per minute
    MILVUS_MAX_CONNECTIONS: int = Field(default=5)  # Max connections to Milvus
    MILVUS_CONNECTION_TIMEOUT: int = Field(default=30)  # Timeout for Milvus connections
    
    # Advanced Retriever Management Configuration
    RETRIEVER_CACHE_MAX_SIZE: int = Field(default=100)  # Max retrievers in cache
    RETRIEVER_MAX_AGE: int = Field(default=7200)  # Max age for retrievers (2 hours)
    RETRIEVER_UNUSED_TIMEOUT: int = Field(default=3600)  # Cleanup unused retrievers (1 hour)
    RETRIEVER_ERROR_THRESHOLD: float = Field(default=0.1)  # Max error rate (10%)
    MAX_RETRIEVER_ERROR_RATE: float = Field(default=0.2)  # Max error rate before marking unhealthy
    MAX_CONCURRENT_REQUESTS_PER_RETRIEVER: int = Field(default=5)  # Max concurrent requests per retriever
    
    # Retriever Pool Configuration
    RETRIEVER_POOL_MAX_SIZE: int = Field(default=3)  # Max instances per pool
    POOL_SCALING_COOLDOWN: int = Field(default=60)  # Cooldown between scaling events (seconds)
    POOL_SCALING_CHECK_INTERVAL: int = Field(default=30)  # Interval for checking scaling needs
    POOL_SCALE_UP_THRESHOLD: float = Field(default=0.8)  # Utilization threshold for scaling up
    POOL_SCALE_DOWN_THRESHOLD: float = Field(default=0.3)  # Utilization threshold for scaling down
    POOL_QUEUE_THRESHOLD: int = Field(default=5)  # Queue size threshold for scaling up
    
    # Health Checker Configuration (Phase 4)
    HEALTH_CHECK_ENABLED: bool = Field(default=True)  # Enable health monitoring
    HEALTH_CHECK_INTERVAL_SECONDS: int = Field(default=30)  # Health check frequency
    HEALTH_CHECK_TIMEOUT_SECONDS: int = Field(default=10)  # Timeout for health checks
    HEALTH_CHECK_RETRY_ATTEMPTS: int = Field(default=3)  # Retry attempts for failed checks
    HEALTH_CHECK_CRITICAL_THRESHOLD: int = Field(default=3)  # Failures before critical
    HEALTH_CHECK_WARNING_THRESHOLD: int = Field(default=2)  # Failures before warning
    HEALTH_ALERT_ENABLED: bool = Field(default=True)  # Enable health alerts
    HEALTH_ALERT_HISTORY_SIZE: int = Field(default=1000)  # Max alerts in history
    
    # Circuit Breaker Configuration (Phase 4) - Enhanced
    CIRCUIT_BREAKER_ENABLED: bool = Field(default=True)  # Enable circuit breakers
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=5)  # Failures before opening
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: float = Field(default=60.0)  # Recovery timeout (seconds)
    CIRCUIT_BREAKER_REQUEST_TIMEOUT: float = Field(default=30.0)  # Request timeout (seconds)
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD: int = Field(default=3)  # Successes before closing
    CIRCUIT_BREAKER_MONITORING_PERIOD: float = Field(default=300.0)  # Monitoring window (seconds)
    CIRCUIT_BREAKER_PERFORMANCE_THRESHOLD: float = Field(default=10.0)  # Performance threshold (seconds)
    
    # Graceful Degradation Configuration (Phase 4)
    DEGRADATION_ENABLED: bool = Field(default=True)  # Enable graceful degradation
    DEGRADATION_AUTO_RECOVERY: bool = Field(default=True)  # Enable automatic recovery
    DEGRADATION_RECOVERY_CHECK_INTERVAL: float = Field(default=30.0)  # Recovery check frequency
    DEGRADATION_ERROR_RATE_THRESHOLD: float = Field(default=0.1)  # Error rate threshold for degradation
    DEGRADATION_RESPONSE_TIME_THRESHOLD: float = Field(default=15.0)  # Response time threshold
    DEGRADATION_AVAILABILITY_THRESHOLD: float = Field(default=95.0)  # Availability threshold
    
    # Performance Thresholds for Different Degradation Levels
    DEGRADATION_ENHANCED_ERROR_RATE: float = Field(default=0.05)  # Enhanced mode max error rate
    DEGRADATION_ENHANCED_RESPONSE_TIME: float = Field(default=10.0)  # Enhanced mode max response time
    DEGRADATION_ENHANCED_AVAILABILITY: float = Field(default=98.0)  # Enhanced mode min availability
    
    DEGRADATION_STANDARD_ERROR_RATE: float = Field(default=0.10)  # Standard mode max error rate
    DEGRADATION_STANDARD_RESPONSE_TIME: float = Field(default=15.0)  # Standard mode max response time
    DEGRADATION_STANDARD_AVAILABILITY: float = Field(default=95.0)  # Standard mode min availability
    
    DEGRADATION_REDUCED_ERROR_RATE: float = Field(default=0.20)  # Reduced mode max error rate
    DEGRADATION_REDUCED_RESPONSE_TIME: float = Field(default=25.0)  # Reduced mode max response time
    DEGRADATION_REDUCED_AVAILABILITY: float = Field(default=90.0)  # Reduced mode min availability
    
    DEGRADATION_MINIMAL_ERROR_RATE: float = Field(default=0.35)  # Minimal mode max error rate
    DEGRADATION_MINIMAL_RESPONSE_TIME: float = Field(default=40.0)  # Minimal mode max response time
    DEGRADATION_MINIMAL_AVAILABILITY: float = Field(default=80.0)  # Minimal mode min availability
    
    # Monitoring and Observability (Phase 4)
    SYSTEM_METRICS_ENABLED: bool = Field(default=True)  # Enable system metrics collection
    SYSTEM_METRICS_INTERVAL: int = Field(default=30)  # System metrics collection interval
    COMPONENT_STATISTICS_ENABLED: bool = Field(default=True)  # Enable component statistics
    PERFORMANCE_TRACKING_ENABLED: bool = Field(default=True)  # Enable performance tracking
    ALERT_CALLBACK_TIMEOUT: float = Field(default=5.0)  # Timeout for alert callbacks
    
    # Connection Pooling Configuration (Phase 5)
    CONNECTION_POOLING_ENABLED: bool = Field(default=True)  # Enable connection pooling
    
    # Milvus Connection Pool
    MILVUS_MIN_CONNECTIONS: int = Field(default=2)  # Minimum Milvus connections
    MILVUS_MAX_CONNECTIONS: int = Field(default=10)  # Maximum Milvus connections
    MILVUS_POOL_STRATEGY: str = Field(default="dynamic")  # Milvus pool strategy
    MILVUS_CONNECTION_LIFETIME: int = Field(default=3600)  # Connection lifetime in seconds
    MILVUS_VALIDATION_INTERVAL: int = Field(default=300)  # Validation interval in seconds
    
    # MongoDB Connection Pool
    MONGO_MIN_CONNECTIONS: int = Field(default=1)  # Minimum MongoDB connections
    MONGO_MAX_CONNECTIONS: int = Field(default=5)  # Maximum MongoDB connections
    MONGO_POOL_STRATEGY: str = Field(default="dynamic")  # MongoDB pool strategy
    MONGO_CONNECTION_LIFETIME: int = Field(default=3600)  # Connection lifetime in seconds
    
    # Azure OpenAI Connection Pool
    AZURE_OPENAI_MIN_CONNECTIONS: int = Field(default=2)  # Minimum Azure OpenAI connections
    AZURE_OPENAI_MAX_CONNECTIONS: int = Field(default=15)  # Maximum Azure OpenAI connections
    AZURE_OPENAI_POOL_STRATEGY: str = Field(default="adaptive")  # Azure OpenAI pool strategy
    AZURE_OPENAI_CONNECTION_LIFETIME: int = Field(default=1800)  # Connection lifetime in seconds
    
    # Advanced Cache Configuration (Phase 5)
    ADVANCED_CACHE_MULTI_LEVEL_ENABLED: bool = Field(default=True)  # Enable multi-level cache
    
    # L1 Cache (In-Memory)
    L1_CACHE_MAX_SIZE: int = Field(default=1000)  # Maximum entries in L1 cache
    L1_CACHE_MAX_MEMORY_MB: int = Field(default=512)  # Maximum memory for L1 cache
    L1_CACHE_STRATEGY: str = Field(default="lru")  # L1 cache strategy
    L1_CACHE_COMPRESSION_THRESHOLD: int = Field(default=1024)  # Compression threshold in bytes
    
    # L2 Cache (Redis)
    L2_CACHE_ENABLED: bool = Field(default=True)  # Enable L2 Redis cache
    L2_CACHE_MAX_SIZE: int = Field(default=10000)  # Maximum entries in L2 cache
    L2_CACHE_KEY_PREFIX: str = Field(default="rag_cache:")  # Redis key prefix
    
    # L3 Cache (Disk)
    L3_CACHE_ENABLED: bool = Field(default=True)  # Enable L3 disk cache
    L3_CACHE_MAX_SIZE: int = Field(default=100000)  # Maximum entries in L3 cache
    L3_CACHE_MAX_DISK_MB: int = Field(default=2048)  # Maximum disk space for L3 cache
    L3_CACHE_DIRECTORY: str = Field(default="/tmp/rag_cache")  # L3 cache directory
    L3_CACHE_COMPRESSION: bool = Field(default=True)  # Enable L3 compression
    L3_CACHE_STRATEGY: str = Field(default="lru")  # L3 cache strategy
    
    # Cache Warming Configuration
    CACHE_WARMING_ENABLED: bool = Field(default=True)  # Enable cache warming
    CACHE_WARMING_STRATEGY: str = Field(default="lazy")  # Cache warming strategy
    CACHE_PRELOAD_POPULAR_QUERIES: bool = Field(default=True)  # Preload popular queries
    CACHE_PREDICTIVE_LOADING: bool = Field(default=False)  # Enable predictive loading
    
    # Background Tasks Configuration (Phase 5)
    BACKGROUND_TASKS_ENABLED: bool = Field(default=True)  # Enable background tasks
    MAX_CONCURRENT_BACKGROUND_TASKS: int = Field(default=5)  # Max concurrent background tasks
    BACKGROUND_TASK_RESOURCE_THRESHOLD: float = Field(default=80.0)  # Resource usage threshold
    
    # Task Intervals (in minutes unless specified)
    TASK_RETRIEVER_REFRESH_INTERVAL: int = Field(default=120)  # Retriever refresh interval
    TASK_RESOURCE_CLEANUP_INTERVAL: int = Field(default=60)  # Resource cleanup interval
    TASK_INDEX_OPTIMIZATION_INTERVAL: int = Field(default=360)  # Index optimization interval (6 hours)
    TASK_CONFIG_BACKUP_INTERVAL: int = Field(default=720)  # Config backup interval (12 hours)
    TASK_METRICS_COLLECTION_INTERVAL: int = Field(default=15)  # Metrics collection interval
    TASK_CACHE_MAINTENANCE_INTERVAL: int = Field(default=30)  # Cache maintenance interval
    TASK_CONNECTION_POOL_MAINTENANCE_INTERVAL: int = Field(default=20)  # Connection pool maintenance
    TASK_LOG_CLEANUP_INTERVAL: int = Field(default=1440)  # Log cleanup interval (24 hours)
    
    # Background Task Priorities and Retries
    BACKGROUND_TASK_DEFAULT_RETRIES: int = Field(default=3)  # Default retry count
    BACKGROUND_TASK_TIMEOUT: int = Field(default=300)  # Default timeout in seconds
    
    # Performance Optimization (Phase 5)
    PERFORMANCE_OPTIMIZATION_ENABLED: bool = Field(default=True)  # Enable performance optimizations
    AUTO_SCALING_ENABLED: bool = Field(default=True)  # Enable auto-scaling features
    RESOURCE_MONITORING_ENABLED: bool = Field(default=True)  # Enable resource monitoring
    ADAPTIVE_CONCURRENCY_ENABLED: bool = Field(default=True)  # Enable adaptive concurrency
    
    # Memory Management
    MEMORY_USAGE_WARNING_THRESHOLD: float = Field(default=80.0)  # Memory warning threshold
    MEMORY_USAGE_CRITICAL_THRESHOLD: float = Field(default=90.0)  # Memory critical threshold
    AUTO_MEMORY_CLEANUP_ENABLED: bool = Field(default=True)  # Enable automatic memory cleanup
    
    # Disk Space Management
    DISK_USAGE_WARNING_THRESHOLD: float = Field(default=80.0)  # Disk warning threshold
    DISK_USAGE_CRITICAL_THRESHOLD: float = Field(default=90.0)  # Disk critical threshold
    AUTO_DISK_CLEANUP_ENABLED: bool = Field(default=True)  # Enable automatic disk cleanup
    
    # Backup and Recovery (Phase 5)
    BACKUP_ENABLED: bool = Field(default=True)  # Enable automated backups
    BACKUP_DIRECTORY: str = Field(default="/app/backups")  # Backup directory
    BACKUP_RETENTION_DAYS: int = Field(default=30)  # Backup retention period
    CONFIG_BACKUP_ENABLED: bool = Field(default=True)  # Enable configuration backups
    METRICS_BACKUP_ENABLED: bool = Field(default=True)  # Enable metrics backups
    
    # Environment Configuration (Phase 6.1)
    ENVIRONMENT: str = Field(default="development")  # Environment: development, staging, production
    
    # Production Configuration Profiles
    PRODUCTION_MODE: bool = Field(default=False)  # Enable production optimizations
    STAGING_MODE: bool = Field(default=False)  # Enable staging optimizations
    DEBUG_MODE: bool = Field(default=True)  # Enable debug mode (disabled in production)
    
    # Connection Pooling Production Settings
    PRODUCTION_CONNECTION_POOL_ENABLED: bool = Field(default=True)  # Force enable in production
    PRODUCTION_MIN_CONNECTIONS_MULTIPLIER: float = Field(default=2.0)  # Multiply min connections in production
    PRODUCTION_MAX_CONNECTIONS_MULTIPLIER: float = Field(default=3.0)  # Multiply max connections in production
    
    # Retriever Persistence Production Settings
    PRODUCTION_RETRIEVER_CACHE_SIZE: int = Field(default=500)  # Larger cache in production
    PRODUCTION_RETRIEVER_MAX_AGE: int = Field(default=14400)  # 4 hours in production
    PRODUCTION_PRELOAD_POPULAR_RETRIEVERS: bool = Field(default=True)  # Preload in production
    PRODUCTION_RETRIEVER_WARMING_ENABLED: bool = Field(default=True)  # Enable warming in production
    
    # Health Checks Production Settings
    PRODUCTION_HEALTH_CHECK_INTERVAL: int = Field(default=15)  # More frequent in production
    PRODUCTION_HEALTH_CHECK_TIMEOUT: int = Field(default=5)  # Shorter timeout in production
    PRODUCTION_HEALTH_CHECK_CRITICAL_THRESHOLD: int = Field(default=2)  # Stricter in production
    PRODUCTION_HEALTH_ALERT_ENABLED: bool = Field(default=True)  # Always enable alerts in production
    
    # Performance Tuning Production Settings
    PRODUCTION_MAX_CONCURRENT_REQUESTS: int = Field(default=100)  # Higher concurrency in production
    PRODUCTION_REQUEST_TIMEOUT: int = Field(default=60)  # Shorter timeout in production
    PRODUCTION_BACKGROUND_TASKS_MULTIPLIER: float = Field(default=2.0)  # More background tasks
    PRODUCTION_CACHE_TTL_MULTIPLIER: float = Field(default=2.0)  # Longer cache TTL
    
    # Security Settings for Production
    PRODUCTION_SECURITY_ENABLED: bool = Field(default=True)  # Enable security features
    PRODUCTION_RATE_LIMITING_ENABLED: bool = Field(default=True)  # Enable rate limiting
    PRODUCTION_API_KEY_REQUIRED: bool = Field(default=False)  # Require API keys
    PRODUCTION_CORS_STRICT: bool = Field(default=True)  # Strict CORS in production
    PRODUCTION_LOG_SENSITIVE_DATA: bool = Field(default=False)  # Don't log sensitive data
    
    # Monitoring and Observability
    OBSERVABILITY_ENABLED: bool = Field(default=True)  # Enable observability features
    METRICS_EXPORT_ENABLED: bool = Field(default=True)  # Enable metrics export
    METRICS_EXPORT_INTERVAL: int = Field(default=30)  # Metrics export interval in seconds
    TRACING_ENABLED: bool = Field(default=False)  # Enable distributed tracing
    STRUCTURED_LOGGING_ENABLED: bool = Field(default=True)  # Enable structured logging
    
    # Prometheus Integration
    PROMETHEUS_ENABLED: bool = Field(default=False)  # Enable Prometheus metrics
    PROMETHEUS_PORT: int = Field(default=8080)  # Prometheus metrics port
    PROMETHEUS_ENDPOINT: str = Field(default="/metrics")  # Prometheus metrics endpoint
    
    # Grafana Integration
    GRAFANA_ENABLED: bool = Field(default=False)  # Enable Grafana integration
    GRAFANA_URL: str = Field(default="")  # Grafana URL
    GRAFANA_API_KEY: str = Field(default="")  # Grafana API key
    
    # Alerting Configuration
    ALERTING_ENABLED: bool = Field(default=False)  # Enable alerting
    ALERTING_WEBHOOK_URL: str = Field(default="")  # Webhook URL for alerts
    ALERTING_EMAIL_ENABLED: bool = Field(default=False)  # Enable email alerts
    ALERTING_SLACK_ENABLED: bool = Field(default=False)  # Enable Slack alerts
    
    # Resource Limits for Production
    PRODUCTION_MAX_MEMORY_MB: int = Field(default=4096)  # Max memory usage in MB
    PRODUCTION_MAX_CPU_PERCENT: float = Field(default=80.0)  # Max CPU usage percentage
    PRODUCTION_MAX_DISK_USAGE_PERCENT: float = Field(default=85.0)  # Max disk usage
    
    # Database Connection Limits
    PRODUCTION_DB_POOL_SIZE: int = Field(default=20)  # Database pool size
    PRODUCTION_DB_MAX_OVERFLOW: int = Field(default=10)  # Database overflow connections
    PRODUCTION_DB_TIMEOUT: int = Field(default=30)  # Database connection timeout
    
    # Cache Configuration for Production
    PRODUCTION_REDIS_MAX_CONNECTIONS: int = Field(default=100)  # Redis max connections
    PRODUCTION_REDIS_TIMEOUT: int = Field(default=5)  # Redis operation timeout
    PRODUCTION_CACHE_COMPRESSION_ENABLED: bool = Field(default=True)  # Enable cache compression
    
    # API Configuration
    API_RATE_LIMIT_PER_MINUTE: int = Field(default=1000)  # API rate limit per minute
    API_REQUEST_SIZE_LIMIT_MB: int = Field(default=10)  # API request size limit
    API_RESPONSE_SIZE_LIMIT_MB: int = Field(default=50)  # API response size limit
    
    # Deployment Configuration
    DEPLOYMENT_STRATEGY: str = Field(default="rolling")  # Deployment strategy: rolling, blue-green, canary
    DEPLOYMENT_HEALTH_CHECK_PATH: str = Field(default="/health")  # Health check endpoint
    DEPLOYMENT_READINESS_TIMEOUT: int = Field(default=300)  # Readiness timeout in seconds
    DEPLOYMENT_GRACEFUL_SHUTDOWN_TIMEOUT: int = Field(default=30)  # Graceful shutdown timeout
    
    # Container Configuration
    CONTAINER_MEMORY_LIMIT: str = Field(default="2g")  # Container memory limit
    CONTAINER_CPU_LIMIT: str = Field(default="1")  # Container CPU limit
    CONTAINER_RESTART_POLICY: str = Field(default="unless-stopped")  # Container restart policy

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def get_sources_path(self):
        """Get the path to source documents."""
        return str(Path(self.SOURCES_PATH))
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production" or self.PRODUCTION_MODE
    
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.ENVIRONMENT.lower() == "staging" or self.STAGING_MODE
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development" and not self.PRODUCTION_MODE and not self.STAGING_MODE
    
    def get_connection_pool_config(self) -> dict:
        """Get connection pool configuration based on environment."""
        base_config = {
            "milvus_min": self.MILVUS_MIN_CONNECTIONS,
            "milvus_max": self.MILVUS_MAX_CONNECTIONS,
            "mongo_min": self.MONGO_MIN_CONNECTIONS,
            "mongo_max": self.MONGO_MAX_CONNECTIONS,
            "azure_openai_min": self.AZURE_OPENAI_MIN_CONNECTIONS,
            "azure_openai_max": self.AZURE_OPENAI_MAX_CONNECTIONS,
        }
        
        if self.is_production():
            # Scale up connections for production
            base_config.update({
                "milvus_min": int(self.MILVUS_MIN_CONNECTIONS * self.PRODUCTION_MIN_CONNECTIONS_MULTIPLIER),
                "milvus_max": int(self.MILVUS_MAX_CONNECTIONS * self.PRODUCTION_MAX_CONNECTIONS_MULTIPLIER),
                "mongo_min": int(self.MONGO_MIN_CONNECTIONS * self.PRODUCTION_MIN_CONNECTIONS_MULTIPLIER),
                "mongo_max": int(self.MONGO_MAX_CONNECTIONS * self.PRODUCTION_MAX_CONNECTIONS_MULTIPLIER),
                "azure_openai_min": int(self.AZURE_OPENAI_MIN_CONNECTIONS * self.PRODUCTION_MIN_CONNECTIONS_MULTIPLIER),
                "azure_openai_max": int(self.AZURE_OPENAI_MAX_CONNECTIONS * self.PRODUCTION_MAX_CONNECTIONS_MULTIPLIER),
            })
        
        return base_config
    
    def get_health_check_config(self) -> dict:
        """Get health check configuration based on environment."""
        if self.is_production():
            return {
                "interval": self.PRODUCTION_HEALTH_CHECK_INTERVAL,
                "timeout": self.PRODUCTION_HEALTH_CHECK_TIMEOUT,
                "critical_threshold": self.PRODUCTION_HEALTH_CHECK_CRITICAL_THRESHOLD,
                "alerts_enabled": self.PRODUCTION_HEALTH_ALERT_ENABLED,
            }
        else:
            return {
                "interval": self.HEALTH_CHECK_INTERVAL_SECONDS,
                "timeout": self.HEALTH_CHECK_TIMEOUT_SECONDS,
                "critical_threshold": self.HEALTH_CHECK_CRITICAL_THRESHOLD,
                "alerts_enabled": self.HEALTH_ALERT_ENABLED,
            }
    
    def get_performance_config(self) -> dict:
        """Get performance configuration based on environment."""
        base_config = {
            "max_concurrent_requests": self.MAX_CONCURRENT_TASKS,
            "request_timeout": self.TASK_TIMEOUT,
            "background_tasks": self.MAX_CONCURRENT_BACKGROUND_TASKS,
            "cache_ttl": self.CACHE_TTL,
        }
        
        if self.is_production():
            base_config.update({
                "max_concurrent_requests": self.PRODUCTION_MAX_CONCURRENT_REQUESTS,
                "request_timeout": self.PRODUCTION_REQUEST_TIMEOUT,
                "background_tasks": int(self.MAX_CONCURRENT_BACKGROUND_TASKS * self.PRODUCTION_BACKGROUND_TASKS_MULTIPLIER),
                "cache_ttl": int(self.CACHE_TTL * self.PRODUCTION_CACHE_TTL_MULTIPLIER),
            })
        
        return base_config
    
    def get_retriever_config(self) -> dict:
        """Get retriever configuration based on environment."""
        if self.is_production():
            return {
                "cache_size": self.PRODUCTION_RETRIEVER_CACHE_SIZE,
                "max_age": self.PRODUCTION_RETRIEVER_MAX_AGE,
                "preload_popular": self.PRODUCTION_PRELOAD_POPULAR_RETRIEVERS,
                "warming_enabled": self.PRODUCTION_RETRIEVER_WARMING_ENABLED,
            }
        else:
            return {
                "cache_size": self.RETRIEVER_CACHE_MAX_SIZE,
                "max_age": self.RETRIEVER_MAX_AGE,
                "preload_popular": False,
                "warming_enabled": False,
            }
    
    def get_security_config(self) -> dict:
        """Get security configuration based on environment."""
        if self.is_production():
            return {
                "security_enabled": self.PRODUCTION_SECURITY_ENABLED,
                "rate_limiting": self.PRODUCTION_RATE_LIMITING_ENABLED,
                "api_key_required": self.PRODUCTION_API_KEY_REQUIRED,
                "cors_strict": self.PRODUCTION_CORS_STRICT,
                "log_sensitive_data": self.PRODUCTION_LOG_SENSITIVE_DATA,
            }
        else:
            return {
                "security_enabled": False,
                "rate_limiting": False,
                "api_key_required": False,
                "cors_strict": False,
                "log_sensitive_data": True,
            }
    
    def get_resource_limits(self) -> dict:
        """Get resource limits based on environment."""
        if self.is_production():
            return {
                "max_memory_mb": self.PRODUCTION_MAX_MEMORY_MB,
                "max_cpu_percent": self.PRODUCTION_MAX_CPU_PERCENT,
                "max_disk_percent": self.PRODUCTION_MAX_DISK_USAGE_PERCENT,
            }
        else:
            return {
                "max_memory_mb": 2048,  # 2GB for development
                "max_cpu_percent": 95.0,
                "max_disk_percent": 95.0,
            }
    
    def get_observability_config(self) -> dict:
        """Get observability configuration."""
        return {
            "enabled": self.OBSERVABILITY_ENABLED,
            "metrics_export": self.METRICS_EXPORT_ENABLED,
            "metrics_interval": self.METRICS_EXPORT_INTERVAL,
            "tracing": self.TRACING_ENABLED,
            "structured_logging": self.STRUCTURED_LOGGING_ENABLED,
            "prometheus": {
                "enabled": self.PROMETHEUS_ENABLED,
                "port": self.PROMETHEUS_PORT,
                "endpoint": self.PROMETHEUS_ENDPOINT,
            },
            "grafana": {
                "enabled": self.GRAFANA_ENABLED,
                "url": self.GRAFANA_URL,
                "api_key": self.GRAFANA_API_KEY,
            },
            "alerting": {
                "enabled": self.ALERTING_ENABLED,
                "webhook_url": self.ALERTING_WEBHOOK_URL,
                "email_enabled": self.ALERTING_EMAIL_ENABLED,
                "slack_enabled": self.ALERTING_SLACK_ENABLED,
            }
        }
    
    def apply_environment_overrides(self):
        """Apply environment-specific overrides."""
        if self.is_production():
            # Force certain settings in production
            self.LOG_LEVEL = "INFO"
            self.SHOW_INTERNAL_MESSAGES = False
            self.DEBUG_MODE = False
            
            # Override security settings
            self.ENABLE_CORS = not self.PRODUCTION_CORS_STRICT
            
            # Override performance settings
            self.MAX_CONCURRENT_TASKS = self.PRODUCTION_MAX_CONCURRENT_REQUESTS
            
        elif self.is_staging():
            # Staging specific overrides
            self.LOG_LEVEL = "INFO"
            self.SHOW_INTERNAL_MESSAGES = False
            
        # No overrides needed for development - keep defaults


settings = Settings()

# Apply environment-specific overrides automatically
settings.apply_environment_overrides()