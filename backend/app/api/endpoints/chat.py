from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, status, Request
from typing import List, Dict, Any, Optional
import time
import asyncio
from loguru import logger
import json

from app.core.config import settings
from app.core.metrics import REQUESTS_TOTAL, REQUEST_DURATION, track_active_tasks, ERROR_COUNTER
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
    
    # Log received request data
    logger.debug(f"Chat request received - language: {language}, return_documents: {return_documents}, collection_name: {collection_name}")
    
    # Check if we need to parse the request body
    if messages is None and request:
        try:
            logger.info("No messages provided directly, attempting to parse request body")
            
            # Get raw body content
            body_bytes = await request.body()
            body_str = body_bytes.decode('utf-8')
            logger.info(f"Raw request body: {body_str}")
            
            # Parse body as JSON
            body_data = json.loads(body_str)
            logger.info(f"Parsed request body: {body_data}")
            
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
                logger.info(f"Successfully parsed {len(messages)} messages from request body")
            else:
                logger.warning("No 'messages' field found in request body or it's not a list")
        except Exception as e:
            logger.error(f"Error parsing request body: {e}", exc_info=True)
    
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
        logger.debug(f"Validating language: {language}")
        if language.lower() not in ["german", "english"]:
            logger.warning(f"Invalid language provided: {language}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Language must be 'german' or 'english', received: {language}"
            )
        
        # Log messages array length
        logger.debug(f"Validating messages array, length: {len(messages) if messages else 0}")
        
        # Extract user query from the last message
        if not messages:
            logger.warning("No messages provided in request")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="No messages provided"
            )
            
        # Check if the last message is from user
        logger.debug(f"Checking last message role: {messages[-1].role if messages else 'no messages'}")
        if messages[-1].role != "user":
            logger.warning(f"Last message role is not 'user': {messages[-1].role}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Last message must be from user"
            )
        
        user_query = messages[-1].content
        logger.debug(f"User query extracted: {user_query[:100]}...")
        
        if not user_query or len(user_query.strip()) == 0:
            logger.warning("Empty user query received")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Query cannot be empty"
            )
        
        # Format chat history
        chat_history = []
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages) and messages[i].role == "user" and messages[i+1].role == "assistant":
                chat_history.append((messages[i].content, messages[i+1].content))
        
        logger.debug(f"Formatted chat history with {len(chat_history)} exchanges")
        
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
        
        # Initialize retrievers if needed
        logger.debug("Initializing retrievers")
        try:
            # Determine root collection name and add language-specific suffixes
            root_collection_name = collection_name or settings.COLLECTION_NAME
            german_collection = f"{root_collection_name}_de"
            english_collection = f"{root_collection_name}_en"
            
            logger.info(f"Using root collection: {root_collection_name}")
            logger.info(f"Using collections - German: {german_collection}, English: {english_collection}")
            
            retrievers = {}
            
            # Try to get German retriever
            try:
                logger.debug("Getting German retriever")
                # Check if collection exists before trying to create/load it
                if utility.has_collection(german_collection):
                    german_retriever = await rag_service.get_retriever(
                        settings.get_sources_path("de"),
                        embedding_manager.german_model,
                        german_collection,
                        top_k=settings.MAX_CHUNKS_CONSIDERED,
                        language="german",
                        max_concurrency=settings.MAX_CONCURRENT_TASKS
                    )
                    retrievers["german"] = german_retriever
                else:
                    logger.warning(f"German collection '{german_collection}' does not exist, skipping")
            except Exception as e:
                logger.error(f"Error initializing German retriever: {e}")
                
            # Try to get English retriever
            try:
                logger.debug("Getting English retriever")
                # Check if collection exists before trying to create/load it
                if utility.has_collection(english_collection):
                    english_retriever = await rag_service.get_retriever(
                        settings.get_sources_path("en"),
                        embedding_manager.english_model,
                        english_collection,
                        top_k=settings.MAX_CHUNKS_CONSIDERED,
                        language="english",
                        max_concurrency=settings.MAX_CONCURRENT_TASKS
                    )
                    retrievers["english"] = english_retriever
                else:
                    logger.warning(f"English collection '{english_collection}' does not exist, skipping")
            except Exception as e:
                logger.error(f"Error initializing English retriever: {e}")
                
            # Check if we have at least one retriever
            if not retrievers:
                logger.error(f"No retrievers could be initialized for collection root '{root_collection_name}'")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No valid collections found for '{root_collection_name}'. Please make sure you have uploaded documents to collections with suffixes '_de' or '_en'."
                )
                
            # Assign retrievers (use empty retrievers for missing languages)
            german_retriever = retrievers.get("german")
            english_retriever = retrievers.get("english")
            logger.debug("Retrievers initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize retrievers: {e}")
            ERROR_COUNTER.labels(error_type="RetrievalError", component="embeddings").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize retrieval models. Please try again later."
            )
        
        # Process query
        logger.debug("Processing query with RAG service")
        try:
            # Handle the case where one or both retrievers might be None
            if german_retriever is None and english_retriever is None:
                logger.error("Both retrievers are None, cannot process query")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No valid retrievers available. Please upload documents first."
                )
                
            # Process query with available retrievers
            logger.info(f"Processing query with available retrievers - German: {german_retriever is not None}, English: {english_retriever is not None}")
            result = await rag_service.process_queries_and_combine_results(
                user_query,
                german_retriever,
                english_retriever,
                chat_history,
                language
            )
            logger.debug("Query processed successfully")
        except asyncio.TimeoutError:
            logger.error("Request timed out when processing the query")
            ERROR_COUNTER.labels(error_type="TimeoutError", component="rag_service").inc()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request timed out. Please try a simpler query or try again later."
            )
        except Exception as e:
            logger.error(f"Error during RAG processing: {e}")
            ERROR_COUNTER.labels(error_type="ProcessingError", component="rag_service").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing your query: {str(e)}"
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Record request duration
        REQUEST_DURATION.labels(endpoint="/api/chat").observe(processing_time)
        
        # Extract relevant information
        response = result.get('response', '')
        sources = result.get('sources', [])
        from_cache = result.get('from_cache', False)
        documents = result.get('documents', [])
        
        if not response:
            logger.warning("RAG service returned empty response")
            response = "Leider konnte ich in den verfügbaren Dokumenten keine relevanten Informationen zu Ihrer Frage finden."
        
        # Format response
        chat_response = ChatResponse(
            response=response,
            sources=sources,
            from_cache=from_cache,
            processing_time=processing_time,
            documents=documents if return_documents else None
        )
        
        logger.debug(f"Response generated successfully in {processing_time:.2f} seconds")
        
        # Update metrics
        REQUESTS_TOTAL.labels(endpoint="/api/chat", status="success").inc()
        
        # Clean up resources in background
        if background_tasks:
            background_tasks.add_task(coroutine_manager.cleanup)

        return chat_response
        
    except HTTPException as he:
        # Log detailed information about HTTP exceptions
        logger.error(f"HTTP Exception {he.status_code}: {he.detail}")
        REQUESTS_TOTAL.labels(endpoint="/api/chat", status="error").inc()
        raise
    except Exception as e:
        # Record error
        logger.exception(f"Unexpected error in chat endpoint: {e}")
        REQUESTS_TOTAL.labels(endpoint="/api/chat", status="error").inc()
        ERROR_COUNTER.labels(error_type="UnhandledException", component="chat_endpoint").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error inesperado: {str(e)}. Por favor contacta al administrador."
        )
    finally:
        # Log request duration regardless of success/failure
        duration = time.time() - start_time
        logger.info(f"Chat request processed in {duration:.2f} seconds")
