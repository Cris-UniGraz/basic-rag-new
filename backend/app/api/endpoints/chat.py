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

# Create RAG service
rag_service = create_rag_service(llm_service)


@router.post("/chat", response_model=ChatResponse, summary="Chat with RAG")
@track_active_tasks("chat_request")
async def chat(
    messages: List[ChatMessage] = None,
    language: str = Query("german", description="Response language (german or english)"),
    return_documents: bool = Query(False, description="Whether to return retrieved documents"),
    collection_name: str = Query(None, description="Collection name to use for retrieval"),
    background_tasks: BackgroundTasks = None,
    request: Request = None,  # Add request parameter to access raw body
):
    """
    Chat with the RAG system.
    
    - **messages**: List of chat messages
    - **language**: Response language (german or english)
    - **return_documents**: Whether to return retrieved documents
    """
    start_time = time.time()
    
    # Log received request data asíncronamente
    async_metadata_processor.log_async(
        "DEBUG", 
        f"Chat request received - language: {language}, return_documents: {return_documents}, collection_name: {collection_name}",
        {"language": language, "return_documents": return_documents, "collection_name": collection_name}
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
        
        # Validate language
        async_metadata_processor.log_async("DEBUG", f"Validating language: {language}", {"language": language})
        if language.lower() not in ["german", "english"]:
            async_metadata_processor.log_async("WARNING", f"Invalid language provided: {language}", {"language": language}, priority=2)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Language must be 'german' or 'english', received: {language}"
            )
        
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
        
        # Ensure RAG service is initialized
        logger.debug("Ensuring RAG service is initialized")
        try:
            await rag_service.ensure_initialized()
            logger.debug("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service initialization failed. Please try again later."
            )
        
        # Initialize retrievers in parallel for better performance
        logger.debug("Initializing retrievers in parallel using optimized method")
        try:
            root_collection_name = collection_name or settings.COLLECTION_NAME
            logger.info(f"Using root collection: {root_collection_name}")
            
            # Use the new parallel initialization method
            initialization_result = await rag_service.initialize_retrievers_parallel(
                collection_name=root_collection_name,
                top_k=settings.MAX_CHUNKS_CONSIDERED,
                max_concurrency=settings.MAX_CONCURRENT_TASKS
            )
            
            retrievers = initialization_result["retrievers"]
            metadata = initialization_result["metadata"]
            
            # Log initialization summary
            logger.info(f"Parallel retriever initialization completed: "
                       f"{metadata['successful_retrievers']} successful, "
                       f"{metadata['failed_retrievers']} failed, "
                       f"in {metadata['initialization_time']:.2f}s")
            
            # Check if we have at least one retriever
            if not retrievers:
                logger.error(f"No retrievers could be initialized for collection root '{root_collection_name}'")
                
                # Provide detailed error information from metadata
                error_details = []
                for task_detail in metadata.get("task_details", []):
                    if not task_detail.get("exists", False):
                        error_details.append(f"Collection '{task_detail.get('collection')}' does not exist")
                    elif task_detail.get("initialization_error"):
                        error_details.append(f"Failed to initialize {task_detail.get('language')} retriever: {task_detail.get('initialization_error')}")
                
                error_message = f"No valid collections found for '{root_collection_name}'. "
                if error_details:
                    error_message += "Details: " + "; ".join(error_details[:3])  # Limit to first 3 errors
                error_message += " Please make sure you have uploaded documents to collections with suffixes '_de' or '_en'."
                
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=error_message
                )
                
            # Extract retrievers for use in processing
            german_retriever = retrievers.get("german")
            english_retriever = retrievers.get("english")
            
            logger.info(f"Successfully initialized retrievers - German: {german_retriever is not None}, English: {english_retriever is not None}")
            
            # Log additional performance metrics if available
            if metadata.get("initialization_time", 0) > 0:
                async_metadata_processor.log_async("INFO", 
                    "Parallel retriever initialization performance summary",
                    {
                        "total_time": metadata["initialization_time"],
                        "successful_retrievers": metadata["successful_retrievers"],
                        "failed_retrievers": metadata["failed_retrievers"],
                        "collection_root": root_collection_name,
                        "languages_available": list(retrievers.keys())
                    })
        except Exception as e:
            logger.error(f"Failed to initialize retrievers: {e}")
            ERROR_COUNTER.labels(error_type="RetrievalError", component="embeddings").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize retrieval models. Please try again later."
            )
        
        # Process query with advanced pipeline selection
        logger.debug("Processing query with RAG service")
        try:
            # Handle the case where one or both retrievers might be None
            if german_retriever is None and english_retriever is None:
                logger.error("Both retrievers are None, cannot process query")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No valid retrievers available. Please upload documents first."
                )
                
            # Process query with available retrievers using advanced async pipeline
            logger.info(f"Processing query with available retrievers - German: {german_retriever is not None}, English: {english_retriever is not None}")
            
            # Select processing method based on configuration
            if settings.ENABLE_ASYNC_PIPELINE:
                logger.info("Using advanced async pipeline for query processing")
                try:
                    result = await asyncio.wait_for(
                        rag_service.process_queries_with_async_pipeline(
                            user_query,
                            german_retriever,
                            english_retriever,
                            chat_history,
                            language
                        ),
                        timeout=settings.CHAT_REQUEST_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Chat request timed out after {settings.CHAT_REQUEST_TIMEOUT} seconds")
                    raise HTTPException(
                        status_code=status.HTTP_408_REQUEST_TIMEOUT,
                        detail="Es tut mir leid, ich bin auf einen Fehler gestoßen: Timeout"
                    )
                
                # Log pipeline metrics if available
                if 'pipeline_metrics' in result and settings.ASYNC_PIPELINE_PHASE_LOGGING:
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
                            "query": user_query[:50] + "..." if len(user_query) > 50 else user_query
                        })
                
                logger.debug("Advanced async pipeline completed successfully")
            else:
                logger.info("Using legacy sequential pipeline for query processing")
                try:
                    result = await asyncio.wait_for(
                        rag_service.process_queries_and_combine_results(
                            user_query,
                            german_retriever,
                            english_retriever,
                            chat_history,
                            language
                        ),
                        timeout=settings.CHAT_REQUEST_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Chat request timed out after {settings.CHAT_REQUEST_TIMEOUT} seconds")
                    raise HTTPException(
                        status_code=status.HTTP_408_REQUEST_TIMEOUT,
                        detail="Es tut mir leid, ich bin auf einen Fehler gestoßen: Timeout"
                    )
                logger.debug("Legacy pipeline completed successfully")
        except Exception as e:
            logger.error(f"Error during RAG processing: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            ERROR_COUNTER.labels(error_type=type(e).__name__, component="rag_service").inc()
            
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
        
        if not response:
            async_metadata_processor.log_async("WARNING", "RAG service returned empty response", priority=2)
            response = "Leider konnte ich in den verfügbaren Dokumenten keine relevanten Informationen zu Ihrer Frage finden."
        
        # Format response
        chat_response = ChatResponse(
            response=response,
            sources=sources,
            from_cache=from_cache,
            processing_time=processing_time,
            documents=documents if return_documents else None
        )
        
        async_metadata_processor.log_async("DEBUG", "Response generated successfully", {"processing_time": processing_time, "sources_count": len(sources), "from_cache": from_cache})
        
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
                "language": language,
                "from_cache": from_cache,
                "sources_count": len(sources),
                "query_length": len(user_query)
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
