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

    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_API_LLM_DEPLOYMENT_ID: str
    AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID: str
    AZURE_OPENAI_LLM_MODEL: str
    AZURE_OPENAI_EMBEDDING_MODEL: str
    
    # Azure Cohere API
    AZURE_COHERE_ENDPOINT: str
    AZURE_COHERE_API_KEY: str

    # Embedding Models Configuration
    EMBEDDING_MODEL_NAME: str
    GERMAN_EMBEDDING_MODEL_NAME: str
    ENGLISH_EMBEDDING_MODEL_NAME: str

    # Chunking Parameters
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=16)
    PARENT_CHUNK_SIZE: int = Field(default=4096)
    PARENT_CHUNK_OVERLAP: int = Field(default=32)
    PAGE_OVERLAP: int = Field(default=16)

    # Reranking Configuration
    RERANKING_TYPE: str = Field(default="cohere")
    GERMAN_COHERE_RERANKING_MODEL: str
    ENGLISH_COHERE_RERANKING_MODEL: str
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
    DEFAULT_LANGUAGE: str = Field(default="german")

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def get_sources_path(self, language: Optional[str] = None):
        """Get the path to source documents, with optional language folder."""
        base_path = Path(self.SOURCES_PATH)
        if language:
            return str(base_path / language.lower())
        return str(base_path)


settings = Settings()