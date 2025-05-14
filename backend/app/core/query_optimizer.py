from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json
import numpy as np
from datetime import datetime, timedelta
import logging
from langchain.schema import Document
import asyncio

from app.core.metrics_manager import MetricsManager
from app.core.config import settings

class QueryOptimizer:
    """
    Sistema de caché avanzado y optimizador de consultas para RAG.
    
    Características:
    - Caché de respuestas LLM para consultas similares
    - Almacenamiento y recuperación eficiente de consultas
    - Caché de embeddings para evitar recálculos
    - Detección de consultas similares mediante embeddings
    - Limpieza automática del caché basada en tiempo
    - Métricas detalladas de rendimiento
    
    Implementa un patrón Singleton para compartir el estado del caché
    a través de múltiples instancias.
    """
    _instance = None

    def __new__(cls):
        """Implementa el patrón Singleton."""
        if cls._instance is None:
            cls._instance = super(QueryOptimizer, cls).__new__(cls)
            # Inicializar todos los atributos aquí usando la configuración
            cls._instance.llm_cache = {}
            cls._instance.max_cache_size = settings.ADVANCED_CACHE_MAX_SIZE
            cls._instance.max_history_size = 10
            cls._instance.metrics = MetricsManager()
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance.query_history = {}
            cls._instance.embedding_cache = {}
            cls._instance.similarity_threshold = settings.ADVANCED_CACHE_SIMILARITY_THRESHOLD
            cls._instance.enabled = settings.ADVANCED_CACHE_ENABLED
            cls._instance.ttl_hours = settings.ADVANCED_CACHE_TTL_HOURS
        return cls._instance
    
    def __init__(self):
        """Constructor vacío para evitar reinicialización por Singleton."""
        pass

    def _generate_query_hash(self, query: str) -> str:
        """
        Genera un hash único para una consulta normalizada.
        
        Args:
            query: La consulta del usuario
            
        Returns:
            Hash MD5 de la consulta normalizada
        """
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _store_llm_response(self, query: str, response: str, language: str, sources: List[Dict] = None):
        """
        Almacena una respuesta LLM en el caché.
        
        Args:
            query: Consulta original
            response: Respuesta generada por el LLM
            language: Idioma de la consulta/respuesta
            sources: Fuentes utilizadas para generar la respuesta
        """
        # Verificar si el caché avanzado está habilitado
        if not self.enabled:
            return
            
        query_hash = self._generate_query_hash(query)
        
        validated_sources = []
        
        if sources and isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict):
                    validated_source = {
                        'source': source.get('source', 'Unknown Source'),
                        'page': source.get('page', 'N/A'),
                        'page_number': source.get('page_number', None),
                        'sheet_name': source.get('sheet_name', None),
                        'reranking_score': source.get('reranking_score', 0)
                    }
                    validated_sources.append(validated_source)

        self.llm_cache[query_hash] = {
            'response': response,
            'timestamp': datetime.now(),
            'language': language,
            'sources': validated_sources,
            'original_query': query
        }

        # Limpiar caché automáticamente cuando sea necesario
        self._auto_cleanup_if_needed()

    def get_llm_response(self, query: str, language: str) -> Optional[Dict]:
        """
        Recupera una respuesta del caché si existe y es válida.
        
        Args:
            query: Consulta del usuario
            language: Idioma de la consulta
            
        Returns:
            Entrada del caché si existe, None en caso contrario
        """
        # Verificar si el caché avanzado está habilitado
        if not self.enabled:
            return None
            
        query_hash = self._generate_query_hash(query)
       
        if query_hash in self.llm_cache:
            cache_entry = self.llm_cache[query_hash]

            # Verificar que la entrada no haya expirado y que el idioma coincida
            if (datetime.now() - cache_entry['timestamp'] < timedelta(hours=self.ttl_hours) and 
                cache_entry['language'] == language):
                self.logger.info(f"Cache hit for query: '{query}'")
                return cache_entry
        
        self.logger.debug(f"Cache miss for query: '{query}'")
        return None
    
    def _validate_document(self, document: Document) -> Document:
        """
        Valida y asegura que un documento tenga todos los metadatos necesarios.
        
        Args:
            document: Documento a validar
            
        Returns:
            Documento con metadatos completos
        """     
        if not hasattr(document, 'metadata') or document.metadata is None:
            document.metadata = {}
        
        # Asegurar campos mínimos requeridos
        required_fields = {
            'source': 'Unknown Source',
            'page': 'N/A',
            'doc_chunk': 0,
            'start_index': 0
        }
        
        for field, default_value in required_fields.items():
            if field not in document.metadata:
                document.metadata[field] = default_value
                
        return document

    def _store_query_result(self, query: str, result: Any, language: str):
        """
        Almacena el resultado de una consulta en el historial.
        
        Args:
            query: Consulta original
            result: Resultado de la consulta
            language: Idioma de la consulta
        """
        query_hash = self._generate_query_hash(query)
        
        # Extraer la respuesta y fuentes del resultado
        response = ''
        sources = []
        
        if isinstance(result, dict):
            response = result.get('response', '')
            sources = result.get('sources', [])
        
        cached_data = {
            'response': response,
            'sources': sources,
            'timestamp': datetime.now(),
            'language': language,
            'original_query': query,
            'usage_count': 0
        }
        
        self.query_history[query_hash] = cached_data
        
        # También almacenar en el caché LLM
        if response:
            self._store_llm_response(query, response, language, sources)


    def _get_cached_result(self, query: str, language: str) -> Optional[Any]:
        """
        Busca un resultado en el historial de consultas.
        
        Args:
            query: Consulta a buscar
            language: Idioma de la consulta
            
        Returns:
            Resultado cacheado o None si no existe
        """
        query_hash = self._generate_query_hash(query)
        if query_hash in self.query_history:
            entry = self.query_history[query_hash]
            if datetime.now() - entry['timestamp'] < timedelta(hours=24):
                entry['usage_count'] += 1
                return {
                    'response': entry.get('response', ''),
                    'sources': entry.get('sources', []),
                    'language': entry.get('language', language),
                    'original_query': entry.get('original_query', query),
                    'from_cache': True
                }
        return None
        
    def _store_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        """
        Almacena un embedding en el caché.
        
        Args:
            text: Texto original
            model_name: Nombre del modelo de embedding
            embedding: Vector de embedding
        """
        key = f"{text}:{model_name}"
        self.embedding_cache[key] = {
            'embedding': embedding,
            'timestamp': datetime.now()
        }
        
    def _get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """
        Recupera un embedding del caché.
        
        Args:
            text: Texto original
            model_name: Nombre del modelo de embedding
            
        Returns:
            Embedding cacheado o None si no existe
        """
        key = f"{text}:{model_name}"
        if key in self.embedding_cache:
            cached = self.embedding_cache[key]
            if datetime.now() - cached['timestamp'] < timedelta(hours=24):
                return cached['embedding']
        return None
        
    async def optimize_query(self, 
                        query: str, 
                        language: str,
                        embedding_model: Any) -> Dict[str, Any]:
        """
        Optimiza una consulta, cacheando y reutilizando resultados cuando es posible.
        
        Args:
            query: Consulta del usuario
            language: Idioma de la consulta
            embedding_model: Modelo de embedding a utilizar
            
        Returns:
            Resultado optimizado que incluye la fuente (caché o nuevo)
        """
        start_time = datetime.now()
        
        # Verificar caché primero
        cached_result = self._get_cached_result(query, language)
        if cached_result:
            self.metrics.metrics['cache_hits'] += 1
            return {'result': cached_result, 'source': 'cache'}
            
        # Obtener o generar embedding
        query_embedding = self._get_embedding(query, str(embedding_model))
        if query_embedding is None:
            # Determinar método correcto según el tipo de modelo
            if hasattr(embedding_model, 'aembed_query'):
                query_embedding = await embedding_model.aembed_query(query)
            else:
                query_embedding = embedding_model.embed_query(query)
            self._store_embedding(query, str(embedding_model), query_embedding)
            
        # Procesar la consulta
        result = {
            'original_query': query,
            'language': language,
            'embedding': query_embedding,
            'timestamp': datetime.now()
        }
        
        self._store_query_result(query, result, language)
        self.metrics.metrics['cache_misses'] += 1
        
        # Registrar métricas
        processing_time = (datetime.now() - start_time).total_seconds()
        self.metrics.metrics['optimization_time'].append(processing_time)
        self.metrics.metrics['query_optimizations'] += 1
        
        return {'result': result, 'source': 'new'}
    
    def cleanup_cache(self, max_age_hours: int = None):
        """
        Limpia entradas antiguas del caché basadas en su timestamp.
        
        Args:
            max_age_hours: Tiempo máximo de vida en horas para las entradas
                           Si es None, usa el valor de configuración
        
        Returns:
            Número de entradas eliminadas
        """
        if not self.enabled:
            return 0
            
        # Usar configuración si no se proporciona max_age_hours
        if max_age_hours is None:
            max_age_hours = self.ttl_hours
            
        current_time = datetime.now()
        keys_to_remove = []
        
        # Identificar entradas antiguas
        for query_hash, entry in self.llm_cache.items():
            age = current_time - entry['timestamp']
            if age > timedelta(hours=max_age_hours):
                keys_to_remove.append(query_hash)
        
        # Eliminar entradas antiguas
        for key in keys_to_remove:
            del self.llm_cache[key]
            
        # Registrar métricas de limpieza
        self.metrics.metrics['cache_cleanups'] += 1
        self.metrics.metrics['entries_removed'] = len(keys_to_remove)

        self.logger.info(f"Cache cleanup completed. Removed {len(keys_to_remove)} entries.")
        
        return len(keys_to_remove)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del estado actual del caché.
        
        Returns:
            Diccionario con estadísticas detalladas
        """
        current_time = datetime.now()
        stats = {
            'total_entries': len(self.llm_cache),
            'oldest_entry': None,
            'newest_entry': None,
            'avg_age_hours': 0
        }
        
        if self.llm_cache:
            ages = [(current_time - entry['timestamp']).total_seconds() / 3600 
                    for entry in self.llm_cache.values()]
            stats.update({
                'oldest_entry': max(ages),
                'newest_entry': min(ages),
                'avg_age_hours': sum(ages) / len(ages)
            })
        
        return stats

    def _auto_cleanup_if_needed(self):
        """
        Limpia automáticamente el caché si se excede el tamaño máximo.
        Primero intenta eliminar entradas antiguas, y si es necesario,
        elimina las entradas menos usadas para hacer espacio.
        """
        if len(self.llm_cache) <= self.max_cache_size:
            return
            
        # Primero eliminar entradas expiradas
        removed = self.cleanup_old_entries()
        
        # Si todavía es necesario, eliminar las entradas más antiguas
        if len(self.llm_cache) > self.max_cache_size:
            entries_to_remove = len(self.llm_cache) - self.max_cache_size
            
            # Ordenar por timestamp y eliminar las más antiguas
            oldest_keys = sorted(
                self.llm_cache.keys(), 
                key=lambda k: self.llm_cache[k]['timestamp']
            )[:entries_to_remove]
            
            for key in oldest_keys:
                del self.llm_cache[key]
                
            self.logger.info(f"Removed {len(oldest_keys)} oldest entries to maintain cache size limit")
            
    def cleanup_old_entries(self):
        """
        Limpia todas las entradas antiguas del caché.
        
        Returns:
            Número de entradas eliminadas
        """
        if not self.enabled:
            return 0
            
        current_time = datetime.now()
        keys_to_remove = [
            key for key, entry in self.llm_cache.items()
            if (current_time - entry['timestamp']) > timedelta(hours=self.ttl_hours)
        ]
        for key in keys_to_remove:
            del self.llm_cache[key]
            
        if keys_to_remove:
            self.logger.info(f"Removed {len(keys_to_remove)} expired entries from cache")
            
        return len(keys_to_remove)