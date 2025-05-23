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

from app.core.config import settings
from app.core.logging import setup_logging
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
    
    # Initialize services
    from app.core.embedding_manager import embedding_manager
    from app.models.vector_store import vector_store_manager
    
    # Initialize embedding models
    embedding_manager.initialize_models()
    
    # Initialize vector store with retries
    for attempt in range(3):
        try:
            logger.info(f"Attempting to connect to Milvus (attempt {attempt+1}/3)")
            vector_store_manager.connect()
            collections = vector_store_manager.list_collections()
            logger.info(f"Successfully connected to Milvus. Available collections: {collections}")
            break
        except Exception as e:
            logger.warning(f"Failed to connect to Milvus (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                logger.info("Waiting 5 seconds before retrying...")
                time.sleep(5)
            else:
                logger.error("Failed to connect to Milvus after 3 attempts")
                # Don't raise here, just continue - we'll retry when needed
    
    # Create data directory
    os.makedirs(settings.SOURCES_PATH, exist_ok=True)
    os.makedirs(f"{settings.SOURCES_PATH}/de", exist_ok=True)
    os.makedirs(f"{settings.SOURCES_PATH}/en", exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    # Clean up
    vector_store_manager.disconnect()
    embedding_manager.clear_models()
    
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
    
    # Log incoming request details
    logger.info(f"INCOMING REQUEST: {request.method} {request.url.path} - Client: {request.client}")
    if "/documents/collections" in request.url.path:
        logger.info(f"COLLECTIONS REQUEST HEADERS: {request.headers}")
    
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
        
        # Log response details for specific endpoints
        if "/documents/collections" in request.url.path:
            logger.info(f"COLLECTIONS RESPONSE: Status {status}")
            # Try to get the response body if it's a collections request
            if hasattr(response, "body"):
                try:
                    logger.info(f"COLLECTIONS RESPONSE BODY: {response.body.decode('utf-8')}")
                except Exception as body_err:
                    logger.warning(f"Could not log response body: {body_err}")
        
    except Exception as e:
        logger.error(f"Error handling request: {e}")
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
    
    # Log completion details
    logger.info(f"COMPLETED: {request.method} {request.url.path} - Status: {status} - Duration: {duration:.3f}s")
    
    return response

# Include API router
app.include_router(api_router, prefix="/api")

# Health check endpoint
@app.get("/health", summary="Health check")
async def health_check():
    """
    Health check endpoint with diagnostics.
    """
    from app.models.vector_store import vector_store_manager
    
    # Check Milvus connection
    milvus_status = "unknown"
    milvus_info = {}
    try:
        logger.info("Health check: Testing Milvus connection")
        vector_store_manager.connect()
        collections = vector_store_manager.list_collections()
        milvus_status = "connected"
        milvus_info = {
            "collections": collections,
            "connection": "active"
        }
        logger.info(f"Health check: Milvus connection successful. Collections: {collections}")
    except Exception as e:
        milvus_status = "error"
        milvus_info = {"error": str(e)}
        logger.error(f"Health check: Milvus connection failed: {e}")
    
    # Get embedding model status
    from app.core.embedding_manager import embedding_manager
    embedding_status = "unknown"
    try:
        if embedding_manager.german_model and embedding_manager.english_model:
            embedding_status = "loaded"
            logger.info("Health check: Embedding models are loaded")
        else:
            embedding_status = "not_loaded"
            logger.warning("Health check: Embedding models are not loaded")
    except Exception as e:
        embedding_status = "error"
        logger.error(f"Health check: Error checking embedding models: {e}")
    
    return {
        "status": "healthy",
        "components": {
            "milvus": {
                "status": milvus_status,
                "info": milvus_info
            },
            "embeddings": {
                "status": embedding_status
            }
        }
    }


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