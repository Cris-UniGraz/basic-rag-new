# Import milvus override FIRST to ensure it sets environment variables
# before any other imports that might use pymilvus
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import milvus_override
sys.path.insert(0, str(Path(__file__).parent.parent))
import milvus_override

# Now import the rest of the modules
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from prometheus_client import make_asgi_app
import time
import asyncio
from datetime import datetime

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.async_metadata_processor import async_metadata_processor
from app.api import api_router
from app.core.metrics import REQUEST_DURATION, REQUESTS_TOTAL


# Initialize logging
logger = setup_logging()

# VSCode Debugging
import debugpy
debugpy.listen(("0.0.0.0", 5678))
# print("Esperando conexión del depurador...")
# debugpy.wait_for_client()
# print("¡Depurador conectado!")


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up application")
    
    # Initialize async metadata processor
    await async_metadata_processor.start()
    logger.info("Async metadata processor started")
    
    # Initialize enhanced services for production
    from app.core.embedding_manager import embedding_manager
    from app.models.vector_store import vector_store_manager
    from app.services.llm_service import llm_service
    from app.services.persistent_rag_service import create_persistent_rag_service
    
    logger.info("Initializing production-ready services with enhanced startup...")
    
    # Initialize startup status tracker
    startup_status = {
        "embedding_manager": False,
        "vector_store_manager": False,
        "persistent_rag_service": False,
        "startup_mode": "full",
        "startup_time": time.time()
    }
    
    # Store startup status for monitoring
    app.state.startup_status = startup_status
    
    # Phase 3.1: Enhanced initialization with timeouts and retry logic
    max_retries = 3
    timeout_per_service = 60  # 60 seconds per service
    
    # Validate settings.CHAT_REQUEST_TIMEOUT is available and use MAX_CHUNKS_CONSIDERED
    logger.info(f"Using chat request timeout: {settings.CHAT_REQUEST_TIMEOUT}s, MAX_CHUNKS_CONSIDERED: {settings.MAX_CHUNKS_CONSIDERED}")
    
    # 1. Initialize embedding manager with retry logic
    for attempt in range(max_retries):
        try:
            logger.info(f"Initializing embedding manager (attempt {attempt + 1}/{max_retries})...")
            
            # Add timeout for embedding manager initialization
            await asyncio.wait_for(
                embedding_manager.startup_initialize(),
                timeout=timeout_per_service
            )
            
            startup_status["embedding_manager"] = True
            logger.info("✓ Embedding manager initialized successfully")
            break
            
        except asyncio.TimeoutError:
            logger.warning(f"Embedding manager initialization timeout (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                logger.error("Embedding manager initialization failed after all retries")
                # Set startup_mode to degraded on failure
                startup_status["startup_mode"] = "degraded"
        except Exception as e:
            logger.error(f"Embedding manager initialization error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                logger.error("Embedding manager initialization failed after all retries")
                # Set startup_mode to degraded on failure
                startup_status["startup_mode"] = "degraded"
            else:
                # Use exponential backoff before retry
                await asyncio.sleep(2 ** attempt)
    
    # 2. Initialize vector store manager with retry logic
    for attempt in range(max_retries):
        try:
            logger.info(f"Initializing vector store manager (attempt {attempt + 1}/{max_retries})...")
            
            await asyncio.wait_for(
                vector_store_manager.initialize_pools(),
                timeout=timeout_per_service
            )
            
            # Test connection and get collections
            collections = vector_store_manager.list_collections()
            logger.info(f"Vector store connected with connection pooling. Collections: {collections}")
            
            startup_status["vector_store_manager"] = True
            logger.info("✓ Vector store manager initialized successfully")
            break
            
        except asyncio.TimeoutError:
            logger.warning(f"Vector store manager initialization timeout (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                logger.error("Vector store manager initialization failed after all retries")
                # Set startup_mode to degraded on failure
                startup_status["startup_mode"] = "degraded"
        except Exception as e:
            logger.error(f"Vector store manager initialization error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                logger.error("Vector store manager initialization failed after all retries")
                # Set startup_mode to degraded on failure
                startup_status["startup_mode"] = "degraded"
            else:
                # Use exponential backoff before retry
                await asyncio.sleep(2 ** attempt)
    
    # 3. Initialize PersistentRAGService en lifespan with retry logic
    persistent_rag_service = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Initializing persistent RAG service (attempt {attempt + 1}/{max_retries})...")
            
            persistent_rag_service = create_persistent_rag_service(llm_service)
            
            # This includes automatic pre-loading of retrievers for existing collections
            await asyncio.wait_for(
                persistent_rag_service.startup_initialization(),
                timeout=timeout_per_service * 2  # More time for RAG service
            )
            
            startup_status["persistent_rag_service"] = True
            logger.info("✓ Persistent RAG service initialized successfully")
            break
            
        except asyncio.TimeoutError:
            logger.warning(f"Persistent RAG service initialization timeout (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                logger.error("Persistent RAG service initialization failed after all retries")
                startup_status["startup_mode"] = "basic"
        except Exception as e:
            logger.error(f"Persistent RAG service initialization error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                logger.error("Persistent RAG service initialization failed after all retries")
                startup_status["startup_mode"] = "basic"
            else:
                # Use exponential backoff before retry
                await asyncio.sleep(2 ** attempt)
    
    # Store persistent RAG service reference
    if persistent_rag_service:
        app.state.persistent_rag_service = persistent_rag_service
    
    # Phase 3.3: Determine final startup mode and fallback strategies
    if startup_status["startup_mode"] == "full":
        logger.info("🎉 All production services initialized successfully - FULL MODE")
    elif startup_status["startup_mode"] == "degraded":
        logger.warning("⚠️ Starting in DEGRADED MODE - Some services failed")
        
        # Attempt fallback to basic initialization for critical services
        try:
            if not startup_status["embedding_manager"]:
                logger.info("Attempting fallback embedding manager initialization...")
                embedding_manager.initialize_model()
                startup_status["embedding_manager"] = True
                
            if not startup_status["vector_store_manager"]:
                logger.info("Attempting fallback vector store manager initialization...")
                vector_store_manager.connect()
                startup_status["vector_store_manager"] = True
                
            logger.info("✓ Fallback initialization completed")
        except Exception as fallback_error:
            logger.error(f"Even fallback initialization failed: {fallback_error}")
            startup_status["startup_mode"] = "basic"
    else:
        logger.error("🚨 Starting in BASIC MODE - Critical failures occurred")
    
    # Record startup metrics
    startup_duration = time.time() - startup_status["startup_time"]
    startup_status["startup_duration"] = startup_duration
    
    # Log startup completion with async metadata processor
    async_metadata_processor.log_async("INFO", 
        "Application startup completed",
        {
            "startup_mode": startup_status["startup_mode"],
            "startup_duration": startup_duration,
            "embedding_manager_ok": startup_status["embedding_manager"],
            "vector_store_manager_ok": startup_status["vector_store_manager"],
            "persistent_rag_service_ok": startup_status["persistent_rag_service"]
        })
    
    logger.info(f"Application startup completed in {startup_duration:.2f}s - Mode: {startup_status['startup_mode']}")
    
    # Create data directory
    os.makedirs(settings.SOURCES_PATH, exist_ok=True)
    os.makedirs(f"{settings.SOURCES_PATH}/de", exist_ok=True)
    os.makedirs(f"{settings.SOURCES_PATH}/en", exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    # Stop async metadata processor
    await async_metadata_processor.stop()
    logger.info("Async metadata processor stopped")
    
    # Clean up production services
    try:
        # 1. Clean up persistent RAG service
        if hasattr(app.state, 'persistent_rag_service'):
            logger.info("Cleaning up persistent RAG service...")
            await app.state.persistent_rag_service.cleanup()
        
        # 2. Clean up vector store manager
        logger.info("Cleaning up vector store manager...")
        await vector_store_manager.cleanup()
        
        # 3. Clean up embedding manager
        logger.info("Cleaning up embedding manager...")
        await embedding_manager.cleanup()
        
        logger.info("All production services cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during service cleanup: {e}")
        
        # Fallback to basic cleanup
        try:
            vector_store_manager.disconnect()
            embedding_manager.clear_models()
            logger.info("Basic cleanup completed as fallback")
        except Exception as fallback_error:
            logger.error(f"Even fallback cleanup failed: {fallback_error}")
    
    # Clean up coroutine manager
    from app.core.coroutine_manager import coroutine_manager
    await coroutine_manager.cleanup()


# Create application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Configure CORS
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Request metrics middleware
@app.middleware("http")
async def add_metrics(request: Request, call_next):
    # Skip metrics endpoint
    if request.url.path == "/metrics":
        return await call_next(request)
    
    # Log incoming request details asíncronamente
    async_metadata_processor.log_async("INFO", f"INCOMING REQUEST: {request.method} {request.url.path}", {
        "method": request.method,
        "path": request.url.path,
        "client": str(request.client)
    })
    if "/documents/collections" in request.url.path:
        async_metadata_processor.log_async("INFO", "Collections request received", {
            "headers": dict(request.headers),
            "path": request.url.path
        })
    
    # Start timer
    start_time = time.time()
    
    # Increment request counter
    REQUESTS_TOTAL.labels(
        endpoint=request.url.path,
        status="pending"
    ).inc()
    
    # Process request
    try:
        response = await call_next(request)
        status = response.status_code
        
        # Log response details for specific endpoints asíncronamente
        if "/documents/collections" in request.url.path:
            async_metadata_processor.log_async("INFO", "Collections response generated", {
                "status": status,
                "path": request.url.path
            })
            # Try to get the response body if it's a collections request
            if hasattr(response, "body"):
                try:
                    body_content = response.body.decode('utf-8')
                    async_metadata_processor.log_async("INFO", "Collections response body", {
                        "body_length": len(body_content),
                        "body_preview": body_content[:200] + "..." if len(body_content) > 200 else body_content
                    })
                except Exception as body_err:
                    async_metadata_processor.log_async("WARNING", f"Could not log response body: {body_err}", {
                        "error": str(body_err)
                    })
        
    except Exception as e:
        async_metadata_processor.log_async("ERROR", f"Error handling request: {e}", {
            "error": str(e),
            "path": request.url.path,
            "method": request.method
        }, priority=3)
        status = 500
        response = JSONResponse(
            status_code=status,
            content={"detail": "Internal server error"}
        )
    
    # Record request duration
    duration = time.time() - start_time
    REQUEST_DURATION.labels(
        endpoint=request.url.path
    ).observe(duration)
    
    # Update request counter with final status
    REQUESTS_TOTAL.labels(
        endpoint=request.url.path,
        status=str(status)
    ).inc()
    
    # Log completion details asíncronamente
    async_metadata_processor.log_async("INFO", "Request completed", {
        "method": request.method,
        "path": request.url.path,
        "status": status,
        "duration": duration
    })
    
    return response

# Include API router
app.include_router(api_router, prefix="/api")

# Health check endpoint
@app.get("/health", summary="Health check")
async def health_check():
    """
    Enhanced health check endpoint with comprehensive diagnostics.
    """
    from app.models.vector_store import vector_store_manager
    from app.core.embedding_manager import embedding_manager
    
    health_response = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {},
        "startup_info": {}
    }
    
    # Include startup information
    if hasattr(app.state, 'startup_status'):
        health_response["startup_info"] = app.state.startup_status
    
    overall_healthy = True
    
    # Check Vector Store Manager
    try:
        logger.info("Health check: Testing vector store manager")
        vector_store_health = await vector_store_manager.get_health_status()
        
        health_response["components"]["vector_store"] = {
            "status": "healthy" if vector_store_health["health_status"]["is_healthy"] else "unhealthy",
            "details": vector_store_health
        }
        
        if not vector_store_health["health_status"]["is_healthy"]:
            overall_healthy = False
            
        logger.info(f"Health check: Vector store status: {vector_store_health['health_status']['is_healthy']}")
        
    except Exception as e:
        health_response["components"]["vector_store"] = {
            "status": "error",
            "error": str(e)
        }
        overall_healthy = False
        logger.error(f"Health check: Vector store error: {e}")
    
    # Check Embedding Manager
    try:
        logger.info("Health check: Testing embedding manager")
        embedding_health = await embedding_manager.get_health_status()
        
        health_response["components"]["embeddings"] = {
            "status": "healthy" if embedding_health["startup_completed"] else "initializing",
            "details": embedding_health
        }
        
        if not embedding_health["startup_completed"]:
            overall_healthy = False
            
        logger.info(f"Health check: Embedding manager startup completed: {embedding_health['startup_completed']}")
        
    except Exception as e:
        health_response["components"]["embeddings"] = {
            "status": "error",
            "error": str(e)
        }
        overall_healthy = False
        logger.error(f"Health check: Embedding manager error: {e}")
    
    # Check Persistent RAG Service
    try:
        if hasattr(app.state, 'persistent_rag_service'):
            logger.info("Health check: Testing persistent RAG service")
            rag_health = await app.state.persistent_rag_service.get_health_status()
            
            health_response["components"]["persistent_rag"] = {
                "status": "healthy" if rag_health["startup_completed"] else "initializing",
                "details": rag_health
            }
            
            if not rag_health["startup_completed"]:
                overall_healthy = False
                
            logger.info(f"Health check: Persistent RAG service status: {rag_health['service_status']}")
        else:
            health_response["components"]["persistent_rag"] = {
                "status": "not_initialized",
                "message": "Persistent RAG service not found"
            }
            overall_healthy = False
            
    except Exception as e:
        health_response["components"]["persistent_rag"] = {
            "status": "error",
            "error": str(e)
        }
        overall_healthy = False
        logger.error(f"Health check: Persistent RAG service error: {e}")
    
    # Set overall status
    if overall_healthy:
        health_response["status"] = "healthy"
    else:
        health_response["status"] = "degraded"
        
    return health_response


# Root endpoint
@app.get("/", summary="Root endpoint")
async def root():
    """
    Root endpoint.
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_url": "/health",
        "metrics_url": "/metrics",
    }