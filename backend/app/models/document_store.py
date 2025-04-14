from typing import List, Dict, Any, Optional, Union, Tuple
from loguru import logger
from langchain_community.storage import MongoDBStore, RedisStore
from langchain_core.documents import Document
import time
from pymongo import MongoClient
import json
import hashlib
from redis import Redis
from tqdm import tqdm

from app.core.config import settings
from app.core.metrics import ERROR_COUNTER


class DocumentStoreManager:
    """
    Manages document storage for parent documents in MongoDB and Redis.
    
    Features:
    - Automatic connection management
    - Caching with Redis for frequently accessed documents
    - Batch processing for efficient document storage
    """
    
    def __init__(self):
        """Initialize the document store manager."""
        self._mongo_stores: Dict[str, MongoDBStore] = {}
        self._redis_store: Optional[RedisStore] = None
        self._mongo_client: Optional[MongoClient] = None
        self._redis_client: Optional[Redis] = None
    
    def _connect_mongo(self) -> MongoClient:
        """
        Connect to MongoDB.
        
        Returns:
            MongoDB client
        """
        if self._mongo_client is None:
            try:
                self._mongo_client = MongoClient(settings.MONGODB_CONNECTION_STRING)
                logger.info("Connected to MongoDB")
            except Exception as e:
                ERROR_COUNTER.labels(
                    error_type="ConnectionError",
                    component="mongodb"
                ).inc()
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise RuntimeError(f"Failed to connect to MongoDB: {str(e)}")
        
        return self._mongo_client
    
    def _connect_redis(self) -> Redis:
        """
        Connect to Redis.
        
        Returns:
            Redis client
        """
        if self._redis_client is None:
            try:
                self._redis_client = Redis.from_url(
                    settings.REDIS_URL, 
                    decode_responses=False
                )
                logger.info("Connected to Redis")
            except Exception as e:
                ERROR_COUNTER.labels(
                    error_type="ConnectionError",
                    component="redis"
                ).inc()
                logger.error(f"Failed to connect to Redis: {e}")
                raise RuntimeError(f"Failed to connect to Redis: {str(e)}")
        
        return self._redis_client
    
    def get_mongo_store(self, collection_name: str) -> MongoDBStore:
        """
        Get a MongoDB document store.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            MongoDB document store
        """
        if collection_name in self._mongo_stores:
            return self._mongo_stores[collection_name]
        
        try:
            # Connect to MongoDB
            self._connect_mongo()
            
            # Create store
            store = MongoDBStore(
                connection_string=settings.MONGODB_CONNECTION_STRING,
                db_name=settings.MONGODB_DATABASE_NAME,
                collection_name=collection_name
            )
            
            # Cache store
            self._mongo_stores[collection_name] = store
            
            return store
            
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type="MongoDBError",
                component="get_store"
            ).inc()
            logger.error(f"Error getting MongoDB store for '{collection_name}': {e}")
            raise RuntimeError(f"Failed to get MongoDB store: {str(e)}")
    
    def get_redis_store(self) -> RedisStore:
        """
        Get a Redis document store for caching.
        
        Returns:
            Redis document store
        """
        if self._redis_store is not None:
            return self._redis_store
        
        try:
            # Connect to Redis
            self._connect_redis()
            
            # Create store
            store = RedisStore(
                redis_url=settings.REDIS_URL,
                namespace="parent_documents",
                ttl=settings.CACHE_TTL
            )
            
            # Cache store
            self._redis_store = store
            
            return store
            
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type="RedisError",
                component="get_store"
            ).inc()
            logger.error(f"Error getting Redis store: {e}")
            raise RuntimeError(f"Failed to get Redis store: {str(e)}")
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[Document],
        batch_size: int = 100,
        cache_in_redis: bool = True
    ) -> None:
        """
        Add documents to MongoDB and optionally cache in Redis.
        
        Args:
            collection_name: Name of the collection
            documents: Documents to add
            batch_size: Number of documents to process in each batch
            cache_in_redis: Whether to cache documents in Redis
        """
        # Get stores
        mongo_store = self.get_mongo_store(collection_name)
        redis_store = self.get_redis_store() if cache_in_redis else None
        
        try:
            logger.info(f"Adding {len(documents)} documents to collection '{collection_name}'")
            
            # Process documents in batches
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for batch_idx in tqdm(range(total_batches), desc="Adding document batches"):
                # Extract batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(documents))
                batch = documents[start_idx:end_idx]
                
                # Process each document in the batch
                for i, doc in enumerate(batch):
                    # Ensure document has metadata
                    if doc.metadata is None:
                        doc.metadata = {}
                    
                    # Add document ID if not present
                    if "doc_id" not in doc.metadata:
                        doc.metadata["doc_id"] = f"{batch_idx}_{i}_{int(time.time())}"
                    
                    # Add document to MongoDB
                    mongo_store.mset([(doc.metadata["doc_id"], doc)])
                    
                    # Optionally cache in Redis
                    if redis_store:
                        redis_store.mset([(doc.metadata["doc_id"], doc)])
                
                # Log progress for large collections
                if batch_idx > 0 and batch_idx % 10 == 0:
                    logger.info(f"Added {batch_idx * batch_size}/{len(documents)} documents")
            
            logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")
            
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type="StorageError",
                component="add_documents"
            ).inc()
            logger.error(f"Error adding documents to collection '{collection_name}': {e}")
            raise RuntimeError(f"Failed to add documents to collection: {str(e)}")
    
    def get_document(
        self,
        collection_name: str,
        doc_id: str,
        use_cache: bool = True
    ) -> Optional[Document]:
        """
        Get a document from storage.
        
        Args:
            collection_name: Name of the collection
            doc_id: Document ID
            use_cache: Whether to check Redis cache first
            
        Returns:
            Document or None if not found
        """
        # Try Redis cache first if enabled
        if use_cache and self._redis_store:
            try:
                doc = self._redis_store.mget([doc_id])
                if doc and doc[0]:
                    logger.debug(f"Retrieved document {doc_id} from Redis cache")
                    return doc[0]
            except Exception as e:
                logger.warning(f"Error retrieving from Redis cache: {e}")
        
        # Try MongoDB
        try:
            mongo_store = self.get_mongo_store(collection_name)
            doc = mongo_store.mget([doc_id])
            
            if doc and doc[0]:
                logger.debug(f"Retrieved document {doc_id} from MongoDB")
                
                # Add to Redis cache if enabled
                if use_cache and self._redis_store:
                    try:
                        self._redis_store.mset([(doc_id, doc[0])])
                    except Exception as e:
                        logger.warning(f"Error caching in Redis: {e}")
                
                return doc[0]
                
            logger.debug(f"Document {doc_id} not found in collection '{collection_name}'")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def get_documents(
        self,
        collection_name: str,
        doc_ids: List[str],
        use_cache: bool = True
    ) -> Dict[str, Document]:
        """
        Get multiple documents by their IDs.
        
        Args:
            collection_name: Name of the collection
            doc_ids: List of document IDs
            use_cache: Whether to use Redis cache
            
        Returns:
            Dictionary mapping document IDs to Documents
        """
        if not doc_ids:
            return {}
        
        result = {}
        missing_ids = set(doc_ids)
        
        # Try Redis cache first if enabled
        if use_cache and self._redis_store:
            try:
                docs = self._redis_store.mget(doc_ids)
                
                # Process results
                for i, doc in enumerate(docs):
                    if doc:
                        result[doc_ids[i]] = doc
                        missing_ids.remove(doc_ids[i])
                
                logger.debug(f"Retrieved {len(result)} documents from Redis cache")
                
                # If all found in cache, return early
                if not missing_ids:
                    return result
                    
            except Exception as e:
                logger.warning(f"Error retrieving from Redis cache: {e}")
        
        # Get remaining documents from MongoDB
        if missing_ids:
            try:
                mongo_store = self.get_mongo_store(collection_name)
                docs = mongo_store.mget(list(missing_ids))
                
                # Process results
                for i, doc in enumerate(docs):
                    if doc:
                        doc_id = list(missing_ids)[i]
                        result[doc_id] = doc
                        
                        # Add to Redis cache if enabled
                        if use_cache and self._redis_store:
                            try:
                                self._redis_store.mset([(doc_id, doc)])
                            except Exception as e:
                                logger.warning(f"Error caching in Redis: {e}")
                
                logger.debug(f"Retrieved {len(docs)} documents from MongoDB")
                
            except Exception as e:
                logger.error(f"Error retrieving documents from MongoDB: {e}")
        
        return result
    
    def delete_document(self, collection_name: str, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            collection_name: Name of the collection
            doc_id: Document ID
            
        Returns:
            True if document was deleted, False otherwise
        """
        try:
            # Delete from MongoDB
            mongo_store = self.get_mongo_store(collection_name)
            mongo_store.delete(doc_id)
            
            # Delete from Redis if available
            if self._redis_store:
                try:
                    self._redis_store.delete(doc_id)
                except Exception as e:
                    logger.warning(f"Error deleting from Redis cache: {e}")
            
            logger.debug(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection was deleted, False otherwise
        """
        try:
            # Connect to MongoDB
            client = self._connect_mongo()
            
            # Drop collection
            db = client[settings.MONGODB_DATABASE_NAME]
            db.drop_collection(collection_name)
            
            # Remove from cache
            if collection_name in self._mongo_stores:
                del self._mongo_stores[collection_name]
            
            logger.info(f"Deleted collection '{collection_name}'")
            return True
            
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type="MongoDBError",
                component="delete_collection"
            ).inc()
            logger.error(f"Error deleting collection '{collection_name}': {e}")
            return False
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Connect to MongoDB
            client = self._connect_mongo()
            
            # Get collection stats
            db = client[settings.MONGODB_DATABASE_NAME]
            count = db[collection_name].count_documents({})
            
            return {
                "exists": True,
                "count": count,
            }
            
        except Exception as e:
            logger.error(f"Error getting stats for collection '{collection_name}': {e}")
            return {"exists": False, "error": str(e)}
    
    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        try:
            # Connect to MongoDB
            client = self._connect_mongo()
            
            # Get collection names
            db = client[settings.MONGODB_DATABASE_NAME]
            return db.list_collection_names()
            
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []


# Global instance
document_store_manager = DocumentStoreManager()