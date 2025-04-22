from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import os
import time
import asyncio
import uuid
from pathlib import Path

from app.core.config import settings
from app.core.metrics import REQUESTS_TOTAL, REQUEST_DURATION, DOCUMENT_PROCESSING_DURATION
from app.schemas.document import DocumentSchema, DocumentSearchResult
from app.utils.files import save_upload_file, get_file_info
from app.utils.loaders import load_documents, FileType
from app.services.rag_service import RAGService, create_rag_service
from app.services.llm_service import llm_service
from app.core.embedding_manager import embedding_manager
from app.models.vector_store import vector_store_manager
from app.core.cache import track_upload_progress, get_upload_progress

from loguru import logger

# Create RAG service
rag_service = create_rag_service(llm_service)

router = APIRouter()


@router.post("/upload", response_model=Dict[str, Any], summary="Upload documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    language: str = Form("german", description="Document language (german or english)"),
    collection_name: str = Form(..., description="Collection name (required)"),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload documents to the system.
    
    - **files**: Documents to upload
    - **language**: Document language (german or english)
    - **collection_name**: Optional collection name (defaults to settings.COLLECTION_NAME)
    """
    start_time = time.time()
    
    try:
        # Increment request counter
        REQUESTS_TOTAL.labels(endpoint="/api/documents/upload", status="processing").inc()
        
        # Validate language
        if language.lower() not in ["german", "english"]:
            raise HTTPException(status_code=400, detail="Language must be 'german' or 'english'")
        
        # Append language suffix to collection name if needed
        language_sufix = get_language_sufix(language)
        if not collection_name.endswith(f"_{language_sufix}"):
            collection_name = f"{collection_name}_{language_sufix}"
            logger.info(f"Using collection name with language suffix: {collection_name}")
        
        # Get destination directory
        dest_dir = settings.get_sources_path(language.lower())
        os.makedirs(dest_dir, exist_ok=True)
        
        # Save uploaded files
        saved_paths = []
        for file in files:
            file_path = await save_upload_file(file, dest_dir, file.filename)
            saved_paths.append(file_path)
        
        # Generate a unique task ID for tracking progress
        task_id = str(uuid.uuid4())
        
        # Initialize progress at 0%
        track_upload_progress(task_id, 0, "started", "Starting document processing")
        
        # Process documents in background
        background_tasks.add_task(
            process_uploaded_documents,
            saved_paths,
            language,
            collection_name,
            task_id
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Record request duration
        REQUEST_DURATION.labels(endpoint="/api/documents/upload").observe(processing_time)
        
        # Update metrics
        REQUESTS_TOTAL.labels(endpoint="/api/documents/upload", status="success").inc()
        
        return {
            "message": f"Uploaded {len(files)} documents. Processing started in background.",
            "uploaded_files": [Path(path).name for path in saved_paths],
            "collection_name": collection_name,
            "language": language,
            "processing_time": processing_time,
            "task_id": task_id  # Return task ID for tracking progress
        }
        
    except Exception as e:
        # Record error
        REQUESTS_TOTAL.labels(endpoint="/api/documents/upload", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {str(e)}")


@router.get("/progress/{task_id}", response_model=Dict[str, Any], summary="Get upload progress")
async def get_document_progress(task_id: str):
    """
    Get the current progress of a document upload task.
    
    - **task_id**: The unique ID of the upload task
    """
    try:
        # Get progress data
        progress_data = get_upload_progress(task_id)
        
        return progress_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting progress: {str(e)}")


@router.get("/collections", response_model=List[Dict[str, Any]], summary="List collections")
async def list_collections():
    """
    List all available collections.
    """
    
    try:
        # Increment request counter
        REQUESTS_TOTAL.labels(endpoint="/api/documents/collections", status="processing").inc()
        
        logger.info("Request received to list collections")
        
        # Try to ensure connection to Milvus
        try:
            logger.info("Connecting to vector store manager...")
            vector_store_manager.connect()
            logger.info("Vector store manager connection successful")
        except Exception as conn_error:
            logger.error(f"Error connecting to vector store manager: {conn_error}")
            # Continue anyway to see if we can still list collections
        
        # Get collections
        logger.info("Attempting to list collections...")
        try:
            collections = vector_store_manager.list_collections()
            logger.info(f"Successfully retrieved {len(collections)} collections")
        except Exception as list_error:
            logger.error(f"Error listing collections: {list_error}")
            raise
        
        # Get statistics for each collection
        result = []
        logger.info("Getting statistics for each collection...")
        
        for collection_name in collections:
            logger.info(f"Getting stats for collection '{collection_name}'")
            try:
                stats = vector_store_manager.get_collection_stats(collection_name)
                result.append({
                    "name": collection_name,
                    "count": stats.get("count", 0),
                    "exists": stats.get("exists", True)
                })
                logger.info(f"Stats for '{collection_name}': count={stats.get('count', 0)}, exists={stats.get('exists', True)}")
            except Exception as stats_error:
                logger.error(f"Error getting stats for collection '{collection_name}': {stats_error}")
                # Add collection with error
                result.append({
                    "name": collection_name,
                    "count": 0,
                    "exists": True,
                    "error": str(stats_error)
                })
        
        # Update metrics
        REQUESTS_TOTAL.labels(endpoint="/api/documents/collections", status="success").inc()
        logger.info(f"Returning {len(result)} collections")
        
        return result
        
    except Exception as e:
        # Record error
        logger.error(f"Error in list_collections endpoint: {str(e)}", exc_info=True)
        REQUESTS_TOTAL.labels(endpoint="/api/documents/collections", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")


@router.delete("/collections/{collection_name}", response_model=Dict[str, Any], summary="Delete collection")
async def delete_collection(collection_name: str):
    """
    Delete a collection.
    
    - **collection_name**: Name of the collection to delete
    """
    try:
        # Increment request counter
        REQUESTS_TOTAL.labels(endpoint="/api/documents/collections", status="processing").inc()
        
        # Delete collection - handle both root name and direct name formats
        success = False
        
        # First try the exact name
        logger.info(f"Trying to delete collection with exact name: '{collection_name}'")
        try:
            if vector_store_manager.delete_collection(collection_name):
                success = True
        except Exception as e:
            logger.warning(f"Error deleting collection '{collection_name}': {e}")
            
        # Try with language suffixes if exact name failed
        if not success:
            logger.info(f"Trying to delete collections with language suffixes")
            german_collection = f"{collection_name}_de"
            english_collection = f"{collection_name}_en"
            
            try:
                if vector_store_manager.delete_collection(german_collection):
                    logger.info(f"Deleted German collection: {german_collection}")
                    success = True
            except Exception as e:
                logger.warning(f"Error deleting German collection '{german_collection}': {e}")
                
            try:
                if vector_store_manager.delete_collection(english_collection):
                    logger.info(f"Deleted English collection: {english_collection}")
                    success = True
            except Exception as e:
                logger.warning(f"Error deleting English collection '{english_collection}': {e}")
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        # Update metrics
        REQUESTS_TOTAL.labels(endpoint="/api/documents/collections", status="success").inc()
        
        return {"message": f"Collection '{collection_name}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        # Record error
        REQUESTS_TOTAL.labels(endpoint="/api/documents/collections", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


@router.get("/search", response_model=DocumentSearchResult, summary="Search documents")
async def search_documents(
    query: str = Query(..., description="Search query"),
    collection_name: Optional[str] = Query(None, description="Collection name"),
    language: str = Query("german", description="Document language (german or english)"),
    top_k: int = Query(5, description="Number of results to return"),
):
    """
    Search for documents.
    
    - **query**: Search query
    - **collection_name**: Collection name (defaults to settings.COLLECTION_NAME)
    - **language**: Document language (german or english)
    - **top_k**: Number of results to return
    """
    start_time = time.time()
    
    try:
        # Increment request counter
        REQUESTS_TOTAL.labels(endpoint="/api/documents/search", status="processing").inc()
        
        # Validate language
        if language.lower() not in ["german", "english"]:
            raise HTTPException(status_code=400, detail="Language must be 'german' or 'english'")
        
        # Append language suffix to collection name if needed
        language_sufix = get_language_sufix(language)
        if not collection_name.endswith(f"_{language_sufix}"):
            collection_name = f"{collection_name}_{language_sufix}"
            logger.info(f"Using collection name with language suffix: {collection_name}")
        
        # Get embedding model based on language
        embedding_model = (
            embedding_manager.german_model if language.lower() == "german" 
            else embedding_manager.english_model
        )
        
        # Get vector store
        vector_store = vector_store_manager.get_collection(collection_name, embedding_model)
        
        if not vector_store:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        # Search for documents
        docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
        
        # Format results
        documents = []
        for doc, score in docs_with_scores:
            # Add similarity score to metadata
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["similarity_score"] = float(score)
            
            # Create document schema
            documents.append(DocumentSchema(
                content=doc.page_content,
                metadata=doc.metadata
            ))
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Record request duration
        REQUEST_DURATION.labels(endpoint="/api/documents/search").observe(processing_time)
        
        # Update metrics
        REQUESTS_TOTAL.labels(endpoint="/api/documents/search", status="success").inc()
        
        return DocumentSearchResult(
            documents=documents,
            total=len(documents),
            reranked=False,
            from_cache=False,
            search_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Record error
        REQUESTS_TOTAL.labels(endpoint="/api/documents/search", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")


async def process_uploaded_documents(
    file_paths: List[str],
    language: str,
    collection_name: str,
    task_id: str
):
    """
    Process uploaded documents in the background.
    
    Args:
        file_paths: List of file paths
        language: Document language
        collection_name: Collection name
        task_id: Unique ID for tracking progress
    """
    try:
        # Get embedding model based on language
        embedding_model = (
            embedding_manager.german_model if language.lower() == "german" 
            else embedding_manager.english_model
        )
        
        # Update progress to 10%
        track_upload_progress(task_id, 10, "processing", "Loading and parsing documents")
        
        # Load documents
        start_time = time.time()
        documents = load_documents(file_paths)
        loading_time = time.time() - start_time
        
        # Update progress to 30%
        track_upload_progress(task_id, 30, "processing", "Processing and splitting documents")
        
        # Record document processing time
        DOCUMENT_PROCESSING_DURATION.labels(file_type="batch").observe(loading_time)
        
        # Split documents
        split_docs = await rag_service.split_documents(documents)
        
        # Update progress to 50%
        track_upload_progress(task_id, 50, "processing", "Creating vector embeddings")
        
        # Add to vector store
        vector_store = vector_store_manager.get_collection(collection_name, embedding_model)
        
        if vector_store:
            # Add to existing collection - use DONT_KEEP_COLLECTIONS from env
            vector_store_manager.add_documents(
                collection_name,
                split_docs,
                embedding_model
            )
        else:
            # Create new collection - use DONT_KEEP_COLLECTIONS from env
            vector_store_manager.create_collection(
                split_docs,
                embedding_model,
                collection_name
            )
        
        # Update progress to 80%
        track_upload_progress(task_id, 80, "processing", "Finalizing document storage")
        
        # Add parent documents
        parent_collection_name = f"{collection_name}_parents"
        await rag_service.create_parent_retriever(
            vector_store_manager.get_collection(collection_name, embedding_model),
            parent_collection_name,
            settings.MAX_CHUNKS_CONSIDERED,
            docs=documents
        )
        
        # Update progress to 100% (completed)
        track_upload_progress(task_id, 100, "completed", "Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing uploaded documents: {e}")
        # Update progress with error
        track_upload_progress(task_id, -1, "error", f"Error processing documents: {str(e)}")
        # Log error but don't raise (background task)

# Function to obtain the correct language code
def get_language_sufix(language: str) -> str:
    language_mapping = {
        "german": "de",
        "english": "en"
    }
    return language_mapping.get(language.lower(), "unknown")