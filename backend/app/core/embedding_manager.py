import os
# Temporarily comment out torch until we can install it
# import torch
from typing import Dict, Optional, Any, List, Union
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import gc
import numpy as np

from .config import settings
from .metrics import measure_time, EMBEDDING_CREATION_DURATION
from .cache import cache_result


class EmbeddingManager:
    """
    Manages embedding models and provides caching and failover capabilities.
    
    Features:
    - Singleton pattern to ensure only one instance exists
    - Lazy loading of models to reduce startup time
    - Model caching to improve performance
    - Memory management to prevent OOM errors
    - Metrics collection for performance monitoring
    - Automatic retries for model loading and embedding generation
    """
    
    _instance = None
    _models: Dict[str, Embeddings] = {}
    _device_map: Dict[str, str] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            logger.info("Initializing EmbeddingManager")
            self._init_device_map()
            self._initialized = True
    
    def _init_device_map(self) -> None:
        """Initialize device mappings for different models."""
        # Temporarily use CPU only
        default_device = "cpu"
        
        logger.info(f"Default device for embeddings: {default_device}")
        
        # Assign default device to models
        self._device_map = {
            settings.GERMAN_EMBEDDING_MODEL_NAME: default_device,
            settings.ENGLISH_EMBEDDING_MODEL_NAME: default_device,
        }
    
    def initialize_models(
        self, 
        german_model_name: Optional[str] = None, 
        english_model_name: Optional[str] = None
    ) -> None:
        """
        Initialize embedding models for German and English languages.
        
        Args:
            german_model_name: Name of the German embedding model
            english_model_name: Name of the English embedding model
        """
        german_model_name = german_model_name or settings.GERMAN_EMBEDDING_MODEL_NAME
        english_model_name = english_model_name or settings.ENGLISH_EMBEDDING_MODEL_NAME
        
        # Load models if they're not already loaded
        if german_model_name not in self._models:
            self._models[german_model_name] = self._load_embedding_model(german_model_name)
            logger.info(f"Loaded German embedding model: {german_model_name}")
        
        if english_model_name not in self._models:
            self._models[english_model_name] = self._load_embedding_model(english_model_name)
            logger.info(f"Loaded English embedding model: {english_model_name}")
    
    @property
    def german_model(self) -> Embeddings:
        """Get the German embedding model, loading it if necessary."""
        model_name = settings.GERMAN_EMBEDDING_MODEL_NAME
        if model_name not in self._models:
            self._models[model_name] = self._load_embedding_model(model_name)
            logger.info(f"Loaded German embedding model: {model_name}")
        return self._models[model_name]
    
    @property
    def english_model(self) -> Embeddings:
        """Get the English embedding model, loading it if necessary."""
        model_name = settings.ENGLISH_EMBEDDING_MODEL_NAME
        if model_name not in self._models:
            self._models[model_name] = self._load_embedding_model(model_name)
            logger.info(f"Loaded English embedding model: {model_name}")
        return self._models[model_name]
    
    def get_model(self, model_name: str) -> Embeddings:
        """
        Get a specified embedding model, loading it if necessary.
        
        Args:
            model_name: Name of the embedding model to get
            
        Returns:
            The requested embedding model
            
        Raises:
            ValueError: If the model name is invalid
        """
        if model_name not in self._models:
            self._models[model_name] = self._load_embedding_model(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        return self._models[model_name]
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _load_embedding_model(self, model_name: str) -> Embeddings:
        """
        Load an embedding model with retry logic.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            An instance of the embedding model
            
        Raises:
            ValueError: If the specified device is not supported
            RuntimeError: If the model cannot be loaded
        """
        try:
            # Try to free memory before loading a new model
            self._manage_memory()
            
            if model_name == "azure_openai":
                # Azure OpenAI embeddings
                embedding_model = AzureOpenAIEmbeddings(
                    azure_deployment=settings.AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    model=settings.AZURE_OPENAI_EMBEDDING_MODEL
                )
                logger.info(f"Loaded Azure OpenAI embedding model: {settings.AZURE_OPENAI_EMBEDDING_MODEL}")
            else:
                # For now, just return Azure OpenAI embeddings for any model
                logger.warning(f"HuggingFace models are disabled, using Azure OpenAI instead for {model_name}")
                embedding_model = AzureOpenAIEmbeddings(
                    azure_deployment=settings.AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    model=settings.AZURE_OPENAI_EMBEDDING_MODEL
                )
                
                # Original HuggingFace code - temporarily disabled
                # Get device from map or use default
                # device = self._device_map.get(model_name, "cpu")
                # 
                # # Hugging Face embeddings
                # model_kwargs = {"device": device}
                # encode_kwargs = {"normalize_embeddings": True}  # For cosine similarity
                # 
                # embedding_model = HuggingFaceBgeEmbeddings(
                #     model_name=model_name,
                #     model_kwargs=model_kwargs,
                #     encode_kwargs=encode_kwargs,
                # )
                # logger.info(f"Loaded Hugging Face embedding model: {model_name} on {device}")
            
            return embedding_model
            
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {e}")
            # Try to free memory and reload
            self._manage_memory(force=True)
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")
    
    def _manage_memory(self, force: bool = False) -> None:
        """
        Manage memory to prevent OOM errors.
        
        Args:
            force: Whether to force memory cleanup regardless of usage
        """
        # Simplified memory management without torch
        if force:
            logger.info("Cleaning up memory")
            
            # Run garbage collection
            gc.collect()
    
    @cache_result(prefix="embed_texts")
    @measure_time(EMBEDDING_CREATION_DURATION, {"model": "default"})
    def embed_texts(
        self, 
        texts: List[str], 
        model_name: Optional[str] = None,
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with batching.
        
        Args:
            texts: List of texts to embed
            model_name: Name of the model to use (defaults to German model)
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Get the specified model or default to German
        if not model_name:
            model = self.german_model
            model_name = settings.GERMAN_EMBEDDING_MODEL_NAME
        else:
            model = self.get_model(model_name)
        
        # Use batching to avoid OOM errors with large input
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Log progress for large batches
            if len(texts) > batch_size and i % (batch_size * 10) == 0 and i > 0:
                logger.debug(f"Embedded {i}/{len(texts)} texts")
        
        return all_embeddings
    
    @cache_result(prefix="embed_query")
    def embed_query(
        self, 
        query: str, 
        model_name: Optional[str] = None
    ) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Text to embed
            model_name: Name of the model to use (defaults to German model)
            
        Returns:
            Embedding vector
        """
        # Get the specified model or default to German
        if not model_name:
            model = self.german_model
            model_name = settings.GERMAN_EMBEDDING_MODEL_NAME
        else:
            model = self.get_model(model_name)
        
        # Create metrics label
        labels = {"model": model_name}
        
        # Generate and time the embedding
        with measure_time(EMBEDDING_CREATION_DURATION, labels):
            embedding = model.embed_query(query)
        
        return embedding
    
    def clear_models(self) -> None:
        """Unload all models and clear the cache to free memory."""
        self._models.clear()
        self._manage_memory(force=True)
        logger.info("Cleared all embedding models")


# Global instance
embedding_manager = EmbeddingManager()