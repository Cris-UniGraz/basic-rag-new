from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, status, Request
from typing import List, Dict, Any, Optional
import time
import asyncio
from loguru import logger
import json

from app.core.config import settings
from app.core.metrics import REQUESTS_TOTAL, REQUEST_DURATION, track_active_tasks, ERROR_COUNTER
from app.core.async_metadata_processor import async_metadata_processor, MetadataType
from app.schemas.document import ChatMessage, ChatResponse
from app.services.rag_service import RAGService, create_rag_service
from app.services.llm_service import llm_service
from app.core.coroutine_manager import coroutine_manager
from app.core.embedding_manager import embedding_manager
from pymilvus import utility

router = APIRouter()

# Create traditional RAG service as fallback
rag_service = create_rag_service(llm_service)


def get_persistent_rag_service(request: Request):
    """
    Get the PersistentRAGService from the application state.
    Returns None if not available (fallback to traditional service).
    """
    try:
        if hasattr(request.app.state, 'persistent_rag_service'):
            return request.app.state.persistent_rag_service
        return None
    except Exception as e:
        logger.warning(f"Failed to get persistent RAG service: {e}")
        return None


def determine_service_mode(persistent_service, startup_status):
    """
    Determine which service mode to use based on availability and health.
    """
    if not persistent_service:
        return "fallback", "No persistent service available"
    
    if not startup_status:
        return "fallback", "No startup status available"
    
    startup_mode = startup_status.get("startup_mode", "basic")
    
    if startup_mode == "full" and startup_status.get("persistent_rag_service", False):
        return "persistent_full", "Persistent service with full features"
    elif startup_mode == "degraded" and startup_status.get("persistent_rag_service", False):
        return "persistent_degraded", "Persistent service with reduced features"
    elif startup_mode in ["degraded", "basic"]:
        return "fallback", f"Startup mode is {startup_mode}"
    else:
        return "fallback", "Unknown startup mode"


@router.post("/chat", response_model=ChatResponse, summary="Chat with RAG")
@track_active_tasks("chat_request")
async def chat(
    messages: List[ChatMessage] = None,
    return_documents: bool = Query(False, description="Whether to return retrieved documents"),
    collection_name: str = Query(None, description="Collection name to use for retrieval"),
    background_tasks: BackgroundTasks = None,
    request: Request = None,  # Add request parameter to access raw body
):
    """
    Chat with the RAG system.
    
    - **messages**: List of chat messages
    - **return_documents**: Whether to return retrieved documents
    - **collection_name**: Collection name to use for retrieval
    """
    start_time = time.time()
    
    # Log received request data asíncronamente
    async_metadata_processor.log_async(
        "DEBUG", 
        f"Chat request received - return_documents: {return_documents}, collection_name: {collection_name}",
        {"return_documents": return_documents, "collection_name": collection_name}
    )
    
    # Check if we need to parse the request body
    if messages is None and request:
        try:
            async_metadata_processor.log_async("INFO", "No messages provided directly, attempting to parse request body")
            
            # Get raw body content
            body_bytes = await request.body()
            body_str = body_bytes.decode('utf-8')
            async_metadata_processor.log_async("INFO", "Raw request body received", {"body_length": len(body_str)})
            
            # Parse body as JSON
            body_data = json.loads(body_str)
            async_metadata_processor.log_async("INFO", "Request body parsed successfully", {"fields": list(body_data.keys())})
            
            # Check if we have a messages field in the body
            if "messages" in body_data and isinstance(body_data["messages"], list):
                # Convert raw messages to ChatMessage objects
                raw_messages = body_data["messages"]
                messages = []
                for msg in raw_messages:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        messages.append(ChatMessage(
                            role=msg["role"],
                            content=msg["content"],
                        ))
                async_metadata_processor.log_async("INFO", f"Successfully parsed {len(messages)} messages from request body", {"message_count": len(messages)})
            else:
                async_metadata_processor.log_async("WARNING", "No 'messages' field found in request body or it's not a list")
        except Exception as e:
            async_metadata_processor.log_async("ERROR", f"Error parsing request body: {e}", {"error": str(e)}, priority=3)
    
    try:
        # Log the messages received (safely handling potential large content)
        messages_log = []
        if messages:
            for i, msg in enumerate(messages):
                content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                messages_log.append({"index": i, "role": msg.role, "content_preview": content_preview})
            logger.debug(f"Messages received: {json.dumps(messages_log)}")
        else:
            logger.warning("No messages available to log")
    except Exception as e:
        logger.error(f"Error logging messages: {str(e)}")
    
    try:
        # Increment request counter
        REQUESTS_TOTAL.labels(endpoint="/api/chat", status="processing").inc()
        
        # Log validation start
        async_metadata_processor.log_async("DEBUG", "Starting request validation", {})
        
        # Log messages array length
        async_metadata_processor.log_async("DEBUG", "Validating messages array", {"message_count": len(messages) if messages else 0})
        
        # Extract user query from the last message
        if not messages:
            async_metadata_processor.log_async("WARNING", "No messages provided in request", priority=2)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="No messages provided"
            )
            
        # Check if the last message is from user
        last_role = messages[-1].role if messages else 'no messages'
        async_metadata_processor.log_async("DEBUG", "Checking last message role", {"role": last_role})
        if messages[-1].role != "user":
            async_metadata_processor.log_async("WARNING", f"Last message role is not 'user': {messages[-1].role}", {"role": messages[-1].role}, priority=2)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Last message must be from user"
            )
        
        user_query = messages[-1].content
        async_metadata_processor.log_async("DEBUG", "User query extracted", {"query_length": len(user_query), "query_preview": user_query[:100] + "..." if len(user_query) > 100 else user_query})
        
        if not user_query or len(user_query.strip()) == 0:
            async_metadata_processor.log_async("WARNING", "Empty user query received", priority=2)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Query cannot be empty"
            )
        
        # Format chat history
        chat_history = []
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages) and messages[i].role == "user" and messages[i+1].role == "assistant":
                chat_history.append((messages[i].content, messages[i+1].content))
        
        async_metadata_processor.log_async("DEBUG", "Formatted chat history", {"history_exchanges": len(chat_history)})
        
        # Phase 3.2: Use Persistent RAG Service when available
        collection_name_to_use = collection_name or settings.COLLECTION_NAME
        logger.info(f"Using collection: {collection_name_to_use}")
        
        # Get persistent RAG service and determine service mode
        persistent_service = get_persistent_rag_service(request)
        startup_status = getattr(request.app.state, 'startup_status', None)
        service_mode, mode_reason = determine_service_mode(persistent_service, startup_status)
        
        logger.info(f"Service mode determined: {service_mode} - {mode_reason}")
        async_metadata_processor.log_async("INFO", 
            "Chat request service mode determined",
            {
                "service_mode": service_mode,
                "reason": mode_reason,
                "collection": collection_name_to_use,
                "persistent_service_available": persistent_service is not None
            })
        
        retriever = None
        service_to_use = None
        
        if service_mode.startswith("persistent"):
            # Use persistent RAG service
            try:
                logger.debug("Using persistent RAG service for retriever...")
                
                # Get persistent retriever with priority based on service mode
                priority = 1 if service_mode == "persistent_full" else 2
                retriever = await persistent_service.get_persistent_retriever(
                    collection_name_to_use, 
                    settings.MAX_CHUNKS_CONSIDERED,
                    priority=priority
                )
                
                if retriever:
                    service_to_use = persistent_service
                    logger.info(f"✓ Successfully obtained persistent retriever for collection: {collection_name_to_use}")
                    
                    async_metadata_processor.log_async("INFO", 
                        "Using persistent retriever",
                        {
                            "collection": collection_name_to_use,
                            "service_mode": service_mode,
                            "priority": priority
                        })
                else:
                    logger.warning("Persistent service failed to provide retriever, falling back...")
                    service_mode = "fallback"
                    
            except Exception as e:
                logger.error(f"Error getting persistent retriever: {e}")
                # Enhanced error handling with circuit breaker concept for resilience
                async_metadata_processor.log_async("ERROR", 
                    "Failed to get persistent retriever",
                    {
                        "error": str(e),
                        "collection": collection_name_to_use
                    }, priority=3)
                service_mode = "fallback"
        
        # Fallback to traditional RAG service if needed
        if service_mode == "fallback" or not retriever:
            logger.info("Using traditional RAG service as fallback...")
            
            try:
                # Ensure traditional RAG service is initialized
                await rag_service.ensure_initialized()
                
                # Get traditional retriever
                retriever = await rag_service.get_retriever(collection_name_to_use, settings.MAX_CHUNKS_CONSIDERED)
                service_to_use = rag_service
                
                if retriever:
                    logger.info(f"✓ Successfully obtained traditional retriever for collection: {collection_name_to_use}")
                    async_metadata_processor.log_async("INFO", 
                        "Using traditional retriever as fallback",
                        {"collection": collection_name_to_use})
                
            except Exception as e:
                logger.error(f"Failed to initialize traditional RAG service: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG service initialization failed. Please try again later."
                )
        
        # Final validation
        if not retriever:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_name_to_use}' not found. Please upload documents first."
            )
        
        # Process query with the selected service
        logger.debug(f"Processing query with service: {type(service_to_use).__name__}")
        try:
            # Use persistent service method if available, otherwise use traditional method
            if service_to_use == persistent_service and hasattr(persistent_service, 'process_query_with_persistent_retrievers'):
                logger.info("Processing query with persistent RAG service")
                try:
                    result = await asyncio.wait_for(
                        persistent_service.process_query_with_persistent_retrievers(
                            user_query,
                            collection_name_to_use,
                            chat_history,
                            settings.MAX_CHUNKS_CONSIDERED
                        ),
                        timeout=settings.CHAT_REQUEST_TIMEOUT
                    )
                    
                    # Add service mode info to result
                    result['service_mode'] = service_mode
                    result['using_persistent_service'] = True
                    
                except asyncio.TimeoutError:
                    logger.error(f"Persistent RAG request timed out after {settings.CHAT_REQUEST_TIMEOUT} seconds")
                    raise HTTPException(
                        status_code=status.HTTP_408_REQUEST_TIMEOUT,
                        detail="Es tut mir leid, ich bin auf einen Fehler gestoßen: Timeout"
                    )
            else:
                # Use traditional RAG service
                logger.info("Processing query with traditional RAG service")
                try:
                    result = await asyncio.wait_for(
                        rag_service.process_query(
                            user_query,
                            retriever,
                            chat_history
                        ),
                        timeout=settings.CHAT_REQUEST_TIMEOUT
                    )
                    
                    # Add service mode info to result
                    result['service_mode'] = 'fallback'
                    result['using_persistent_service'] = False
                    
                except asyncio.TimeoutError:
                    logger.error(f"Traditional RAG request timed out after {settings.CHAT_REQUEST_TIMEOUT} seconds")
                    raise HTTPException(
                        status_code=status.HTTP_408_REQUEST_TIMEOUT,
                        detail="Es tut mir leid, ich bin auf einen Fehler gestoßen: Timeout"
                    )
            
            # Log pipeline metrics if available  
            if settings.ASYNC_PIPELINE_PHASE_LOGGING:
                # Determine if query was answered successfully
                from_cache = result.get('from_cache', False)
                response_text = result.get('response', '')
                sources = result.get('sources', [])
                documents = result.get('documents', [])
                
                # Query is considered answered if:
                # 1. Response came from cache (exact or semantic match)
                # 2. Response is generated and has relevant documents with sufficient reranking score
                # 3. Response is not empty and not an error message
                query_was_answered = False
                
                if from_cache:
                    # Cache hit means query was answered
                    query_was_answered = True
                elif response_text and response_text.strip():
                    # Check if we have documents with sufficient reranking score
                    if documents:
                        has_relevant_docs = any(
                            doc.metadata.get('reranking_score', 0) >= settings.MIN_RERANKING_SCORE 
                            for doc in documents if hasattr(doc, 'metadata') and doc.metadata
                        )
                        query_was_answered = has_relevant_docs
                    elif sources:
                        # If we have sources, consider it answered
                        query_was_answered = True
                    else:
                        # Check if response is not a default error message
                        error_phrases = [
                            "Leider konnte ich", 
                            "keine relevanten Informationen",
                            "Zeitüberschreitung",
                            "tut mir leid",
                            "auf einen Fehler gestoßen"
                        ]
                        query_was_answered = not any(phrase in response_text for phrase in error_phrases)
                
                if 'pipeline_metrics' in result:
                    metrics = result['pipeline_metrics']
                    async_metadata_processor.log_async("INFO", 
                        "Async pipeline performance summary",
                        {
                            "total_time": metrics.get('total_time', 0),
                            "phase_breakdown": {
                                "cache_optimization": metrics.get('phase1_time', 0),
                                "query_generation": metrics.get('phase2_time', 0),
                                "retrieval": metrics.get('phase3_time', 0),
                                "processing_reranking": metrics.get('phase4_time', 0),
                                "response_preparation": metrics.get('phase5_time', 0),
                                "llm_generation": metrics.get('phase6_time', 0)
                            },
                            "query": user_query,  # Complete query without truncation
                            "query_was_answered": query_was_answered
                        })
                else:
                    # No pipeline metrics available - log basic information
                    processing_time_so_far = time.time() - start_time
                    async_metadata_processor.log_async("INFO", 
                        "Async pipeline performance summary",
                        {
                            "total_time": processing_time_so_far,
                            "phase_breakdown": {
                                "cache_optimization": 0.0,
                                "query_generation": 0.0,
                                "retrieval": 0.0,
                                "processing_reranking": 0.0,
                                "response_preparation": 0.0,
                                "llm_generation": 0.0
                            },
                            "query": user_query,  # Complete query without truncation
                            "query_was_answered": query_was_answered,
                            "note": "No detailed pipeline metrics available"
                        })
            
            logger.debug("Advanced async pipeline completed successfully")
        except Exception as e:
            logger.error(f"Error during RAG processing: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            ERROR_COUNTER.labels(error_type=type(e).__name__, component="rag_service").inc()
            
            # Log error in pipeline metrics format
            if settings.ASYNC_PIPELINE_PHASE_LOGGING:
                processing_time_error = time.time() - start_time
                async_metadata_processor.log_async("INFO", 
                    "Async pipeline performance summary",
                    {
                        "total_time": processing_time_error,
                        "phase_breakdown": {
                            "cache_optimization": 0.0,
                            "query_generation": 0.0,
                            "retrieval": 0.0,
                            "processing_reranking": 0.0,
                            "response_preparation": 0.0,
                            "llm_generation": 0.0
                        },
                        "query": user_query,  # Complete query without truncation
                        "query_was_answered": False,  # Error means query was not answered
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
            
            # Provide user-friendly error messages
            if "CancelledError" in str(e) or "cancelled" in str(e).lower():
                detail = "Es tut mir leid, bei der Bearbeitung Ihrer Anfrage ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut."
            else:
                detail = "Es tut mir leid, bei der Bearbeitung Ihrer Anfrage ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut."
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=detail
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Record request duration asíncronamente
        async_metadata_processor.record_metric_async(
            "request_duration_seconds",
            processing_time,
            {"endpoint": "/api/chat"},
            "histogram"
        )
        
        # Extract relevant information
        response = result.get('response', '')
        sources = result.get('sources', [])
        from_cache = result.get('from_cache', False)
        documents = result.get('documents', [])
        using_persistent_service = result.get('using_persistent_service', False)
        result_service_mode = result.get('service_mode', 'unknown')
        
        if not response:
            async_metadata_processor.log_async("WARNING", "RAG service returned empty response", priority=2)
            response = "Leider konnte ich in den verfügbaren Dokumenten keine relevanten Informationen zu Ihrer Frage finden.\n\nFür weitere Informationen siehe: https://intranet.uni-graz.at/"
        
        # Format response
        chat_response = ChatResponse(
            response=response,
            sources=sources,
            from_cache=from_cache,
            processing_time=processing_time,
            documents=documents if return_documents else None
        )
        
        async_metadata_processor.log_async("DEBUG", "Response generated successfully", {
            "processing_time": processing_time, 
            "sources_count": len(sources), 
            "from_cache": from_cache,
            "service_mode": result_service_mode,
            "using_persistent_service": using_persistent_service
        })
        
        # Update metrics asíncronamente
        async_metadata_processor.record_metric_async(
            "requests_total",
            1,
            {"endpoint": "/api/chat", "status": "success"},
            "counter"
        )
        
        # Registrar rendimiento asíncronamente
        async_metadata_processor.record_performance_async(
            "chat_request",
            processing_time,
            True,
            {
                "from_cache": from_cache,
                "sources_count": len(sources),
                "query_length": len(user_query),
                "service_mode": result_service_mode,
                "using_persistent_service": using_persistent_service,
                "collection": collection_name_to_use
            }
        )
        
        # Clean up resources in background
        if background_tasks:
            background_tasks.add_task(coroutine_manager.cleanup)

        return chat_response
        
    except HTTPException as he:
        # Log detailed information about HTTP exceptions asíncronamente
        async_metadata_processor.log_async("ERROR", f"HTTP Exception {he.status_code}: {he.detail}", {
            "status_code": he.status_code,
            "detail": str(he.detail),
            "endpoint": "/api/chat"
        }, priority=3)
        
        async_metadata_processor.record_metric_async(
            "requests_total",
            1,
            {"endpoint": "/api/chat", "status": "error"},
            "counter"
        )
        raise
    except Exception as e:
        # Record error asíncronamente
        async_metadata_processor.log_async("ERROR", f"Unexpected error in chat endpoint: {e}", {
            "error": str(e),
            "error_type": type(e).__name__,
            "endpoint": "/api/chat"
        }, priority=3)
        
        async_metadata_processor.record_metric_async(
            "requests_total",
            1,
            {"endpoint": "/api/chat", "status": "error"},
            "counter"
        )
        
        async_metadata_processor.record_metric_async(
            "errors_total",
            1,
            {"error_type": "UnhandledException", "component": "chat_endpoint"},
            "counter"
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error inesperado: {str(e)}. Por favor contacta al administrador."
        )
    finally:
        # Log request duration regardless of success/failure asíncronamente
        duration = time.time() - start_time
        async_metadata_processor.log_async("INFO", "Chat request completed", {"duration": duration, "endpoint": "/api/chat"})
