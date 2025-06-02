from typing import List, Dict, Any, Optional, Tuple, Union, cast
import hashlib
import json
import numpy as np
from datetime import datetime, timedelta
import logging
from langchain.schema import Document
import asyncio
import re
from collections import defaultdict
import math

# Intenta importar sklearn, pero proporciona una alternativa si no está disponible
try:
    from sklearn.metrics.pairwise import cosine_similarity
    USE_SKLEARN = True
except ImportError:
    USE_SKLEARN = False
    # Se implementará una versión nativa de similitud de coseno

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
            cls._instance.max_history_size = settings.QUERY_HISTORY_SIZE
            cls._instance.metrics = MetricsManager()
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance.query_history = {}
            cls._instance.embedding_cache = {}
            cls._instance.query_embeddings = {}  # Almacena embeddings de consultas anteriores
            cls._instance.similarity_threshold = settings.ADVANCED_CACHE_SIMILARITY_THRESHOLD
            cls._instance.query_similarity_threshold = settings.QUERY_SIMILARITY_THRESHOLD
            cls._instance.enabled = settings.ADVANCED_CACHE_ENABLED
            cls._instance.ttl_hours = settings.ADVANCED_CACHE_TTL_HOURS
            cls._instance.query_optimization_enabled = settings.QUERY_OPTIMIZATION_ENABLED
            cls._instance.semantic_caching_enabled = settings.SEMANTIC_CACHING_ENABLED
            cls._instance.apply_query_rewriting = settings.APPLY_QUERY_REWRITING
            
            # Limpiar respuestas de error existentes al inicializar
            cls._instance.clean_error_responses_from_cache()
            # Limpiar entradas con chunk_content inválido
            cls._instance.clean_invalid_chunk_content_from_cache()
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
        
        # MODIFICADO: Verificar que hay respuesta válida y fuentes encontradas
        if not response or not response.strip():
            self.logger.info(f"No se guardó en caché la consulta '{query[:50]}...' porque la respuesta está vacía")
            return
        
        # NUEVO: No cachear respuestas de error
        if self._is_error_response(response):
            self.logger.info(f"No se guardó en caché la consulta '{query[:50]}...' porque es una respuesta de error")
            return
            
        query_hash = self._generate_query_hash(query)
        
        validated_sources = []
        
        # Verificar si existen documentos relevantes (con puntaje >= MIN_RERANKING_SCORE)
        has_relevant_docs = False
        
        if sources and isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict):
                    validated_source = {
                        'source': source.get('source', 'Unknown Source'),
                        'page': source.get('page', 'N/A'),
                        'page_number': source.get('page_number', None),
                        'sheet_name': source.get('sheet_name', None),
                        'reranking_score': source.get('reranking_score', 0),
                        'chunk_content': source.get('chunk_content', '')  # FIXED: Store the actual chunk content
                    }
                    validated_sources.append(validated_source)
                    
                    # Comprobar si este documento es relevante según su puntaje de reranking
                    if validated_source['reranking_score'] >= settings.MIN_RERANKING_SCORE:
                        has_relevant_docs = True
        
        # MODIFICADO: Solo almacenar en caché si hay respuesta Y fuentes relevantes
        if has_relevant_docs and validated_sources:
            self.llm_cache[query_hash] = {
                'response': response,
                'timestamp': datetime.now(),
                'language': language,
                'sources': validated_sources,
                'original_query': query
            }

            # Limpiar caché automáticamente cuando sea necesario
            self._auto_cleanup_if_needed()
            self.logger.info(f"Respuesta almacenada en caché para la consulta: '{query[:50]}...' con {len(validated_sources)} fuentes")
        else:
            if not validated_sources:
                self.logger.info(f"No se guardó en caché la consulta '{query[:50]}...' porque no se encontraron fuentes")
            else:
                self.logger.info(f"No se guardó en caché la consulta '{query[:50]}...' porque no contiene documentos relevantes")

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
            self.logger.debug(f"Cache disabled - returning None for query: '{query}'")
            return None
            
        query_hash = self._generate_query_hash(query)
        self.logger.debug(f"Looking for query hash: {query_hash} (query: '{query}', language: {language})")
        self.logger.debug(f"Current cache has {len(self.llm_cache)} entries")
       
        if query_hash in self.llm_cache:
            cache_entry = self.llm_cache[query_hash]
            self.logger.debug(f"Found cache entry in llm_cache with language: {cache_entry.get('language')}, timestamp: {cache_entry.get('timestamp')}")

            # Verificar que la entrada no haya expirado y que el idioma coincida
            time_check = datetime.now() - cache_entry['timestamp'] < timedelta(hours=self.ttl_hours)
            language_check = cache_entry['language'] == language
            
            self.logger.debug(f"LLM Cache - Time check: {time_check}, Language check: {language_check}")
            
            if time_check and language_check:
                response_length = len(cache_entry.get('response', ''))
                self.logger.info(f"LLM Cache hit for query: '{query}' - Response length: {response_length}")
                return cache_entry
            else:
                self.logger.debug(f"LLM Cache entry found but failed validation - time_valid: {time_check}, language_match: {language_check}")
        else:
            self.logger.debug(f"Query hash {query_hash} not found in llm_cache")
        
        
        self.logger.debug(f"Cache miss for query: '{query}' in both caches")
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
        
    def _store_embedding(self, text: str, model_name: str, embedding: Union[np.ndarray, List]):
        """
        Almacena un embedding en el caché.
        
        Args:
            text: Texto original
            model_name: Nombre del modelo de embedding
            embedding: Vector de embedding
        """
        # Convertir a numpy array si es una lista
        if isinstance(embedding, list):
            embedding = np.array(embedding)
            
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
        
    def _compute_cosine_similarity_native(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Implementación nativa de similitud de coseno sin dependencias externas.
        
        Args:
            vec1: Primer vector
            vec2: Segundo vector
            
        Returns:
            Similitud de coseno entre 0 y 1
        """
        # Convertir a numpy array si son listas
        if isinstance(vec1, list):
            vec1 = np.array(vec1)
        if isinstance(vec2, list):
            vec2 = np.array(vec2)
            
        # Asegurar que los vectores sean unidimensionales
        if len(vec1.shape) > 1:
            vec1 = vec1.flatten()
        if len(vec2.shape) > 1:
            vec2 = vec2.flatten()
            
        # Calcular producto punto
        dot_product = np.dot(vec1, vec2)
        
        # Calcular las normas
        norm_vec1 = np.sqrt(np.sum(vec1 * vec1))
        norm_vec2 = np.sqrt(np.sum(vec2 * vec2))
        
        # Evitar división por cero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        # Calcular similitud
        similarity = dot_product / (norm_vec1 * norm_vec2)
        
        # Asegurar que el resultado esté en el rango [0, 1]
        return max(0.0, min(float(similarity), 1.0))
        
    def _compute_similarity(self, query_embedding: Union[np.ndarray, List], stored_embedding: Union[np.ndarray, List]) -> float:
        """
        Calcula la similitud del coseno entre dos embeddings.
        
        Args:
            query_embedding: Embedding de la consulta actual
            stored_embedding: Embedding de una consulta almacenada
            
        Returns:
            Valor de similitud entre 0 y 1
        """
        # Convertir a numpy array si son listas
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        if isinstance(stored_embedding, list):
            stored_embedding = np.array(stored_embedding)
            
        if USE_SKLEARN:
            # Usar sklearn si está disponible
            # Asegurar que los embeddings tengan dimensiones correctas para similitud de coseno
            q_emb = query_embedding.reshape(1, -1) if len(query_embedding.shape) == 1 else query_embedding
            s_emb = stored_embedding.reshape(1, -1) if len(stored_embedding.shape) == 1 else stored_embedding
            
            similarity = cosine_similarity(q_emb, s_emb)[0][0]
            return float(similarity)
        else:
            # Usar implementación nativa como alternativa
            return self._compute_cosine_similarity_native(query_embedding, stored_embedding)
    
    def _find_similar_query(self, query_embedding: Union[np.ndarray, List], language: str) -> Optional[Dict]:
        """
        Busca una consulta semánticamente similar en el historial.
        
        Args:
            query_embedding: Embedding de la consulta actual
            language: Idioma de la consulta
            
        Returns:
            Consulta similar si existe, None en caso contrario
        """
        # Convertir a numpy array si es una lista
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        if not self.semantic_caching_enabled:
            return None
            
        best_match = None
        highest_similarity = 0.0
        
        for query_hash, data in self.query_embeddings.items():
            # Solo considerar consultas en el mismo idioma
            if data['language'] != language:
                continue
                
            # Verificar si la entrada ha expirado
            if datetime.now() - data['timestamp'] > timedelta(hours=self.ttl_hours):
                continue
                
            similarity = self._compute_similarity(query_embedding, data['embedding'])
            
            # Registrar la similitud para métricas
            self.metrics.metrics['query_similarity_scores'].append(similarity)
            
            if similarity > highest_similarity and similarity >= self.query_similarity_threshold:
                # Verificar que existe una respuesta en caché para esta consulta
                cached_result = self._get_cached_result(data['query'], language)
                
                # También verificar en el caché LLM
                llm_cached_result = self.get_llm_response(data['query'], language)
                
                # Solo considerar como coincidencia si tiene una respuesta cacheada válida (no de error)
                response_to_check = None
                if llm_cached_result and llm_cached_result.get('response'):
                    response_to_check = llm_cached_result.get('response')
                elif cached_result and cached_result.get('response'):
                    response_to_check = cached_result.get('response')
                
                # Verificar que la respuesta no es un mensaje de error
                if response_to_check and not self._is_error_response(response_to_check):
                    highest_similarity = similarity
                    best_match = {
                        'query': data['query'],
                        'similarity': similarity,
                        'query_hash': query_hash
                    }
        
        if best_match:
            self.logger.info(f"Found similar query with similarity {best_match['similarity']:.4f}: '{best_match['query']}' and valid cached response")
            
        return best_match
    
    def _normalize_query(self, query: str) -> str:
        """
        Normaliza una consulta para mejorar coincidencias.
        
        Args:
            query: Consulta original
            
        Returns:
            Consulta normalizada
        """
        # Eliminar caracteres especiales y convertir a minúsculas
        normalized = re.sub(r'[^\w\s]', ' ', query.lower())
        # Eliminar espacios múltiples
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _is_error_response(self, response: str) -> bool:
        """
        Verifica si una respuesta es un mensaje de error.
        
        Args:
            response: Respuesta a verificar
            
        Returns:
            True si es un mensaje de error, False en caso contrario
        """
        if not response:
            return False
            
        error_messages = [
            "Leider konnte ich in den verfügbaren Dokumenten keine relevanten Informationen",
            "I'm sorry, I couldn't find relevant information",
            "keine relevanten Informationen",
            "no relevant information"
        ]
        
        return any(error_msg in response for error_msg in error_messages)
    
    def _store_query_embedding(self, query: str, embedding: Union[np.ndarray, List], language: str):
        """
        Almacena el embedding de una consulta para comparaciones futuras.
        
        Args:
            query: Consulta original
            embedding: Vector de embedding
            language: Idioma de la consulta
        """
        # Convertir a numpy array si es una lista
        if isinstance(embedding, list):
            embedding = np.array(embedding)
            
        query_hash = self._generate_query_hash(query)
        
        self.query_embeddings[query_hash] = {
            'query': query,
            'embedding': embedding,
            'language': language,
            'timestamp': datetime.now()
        }
        
        # Limitar el tamaño del historial de embeddings
        if len(self.query_embeddings) > self.max_history_size:
            # Eliminar la entrada más antigua
            oldest_key = min(
                self.query_embeddings.keys(), 
                key=lambda k: self.query_embeddings[k]['timestamp']
            )
            del self.query_embeddings[oldest_key]
    
    async def optimize_query(self, 
                        query: str, 
                        language: str,
                        embedding_model: Any) -> Dict[str, Any]:
        """
        Optimiza una consulta, analizando similitud semántica con consultas anteriores,
        cacheando y reutilizando resultados cuando es posible.
        
        Args:
            query: Consulta del usuario
            language: Idioma de la consulta
            embedding_model: Modelo de embedding a utilizar
            
        Returns:
            Resultado optimizado que incluye la fuente (caché o nuevo)
        """
        start_time = datetime.now()
        
        # Verificar si la optimización está habilitada
        if not self.query_optimization_enabled:
            return {'result': {'original_query': query}, 'source': 'new'}
        
        # Verificar caché por coincidencia exacta primero (usar llm_cache, no query_history)
        cached_result = self.get_llm_response(query, language)
        if cached_result:
            self.metrics.metrics['cache_hits'] += 1
            self.logger.info(f"Exact cache hit for query: '{query}' - Response length: {len(cached_result.get('response', ''))}")
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
        
        # Buscar consultas semánticamente similares
        similar_query = self._find_similar_query(query_embedding, language)
        if similar_query and self.semantic_caching_enabled:
            # Verificar si hay una respuesta en caché LLM para la consulta similar (preferir llm_cache)
            llm_cached = self.get_llm_response(similar_query['query'], language)
            
            # Si no hay en llm_cache, verificar en el historial 
            similar_cached = None
            if not llm_cached or not llm_cached.get('response'):
                similar_cached = self._get_cached_result(similar_query['query'], language)
                
            # Usar cualquiera de las dos cachés que tenga respuesta válida
            if (llm_cached and llm_cached.get('response')) or (similar_cached and similar_cached.get('response')):
                self.metrics.metrics['cache_hits'] += 1
                self.logger.info(f"Semantic cache hit with similarity {similar_query['similarity']:.4f}")
                
                # Registrar la similitud para métricas
                self.metrics.metrics['query_similarity_scores'].append(similar_query['similarity'])
                
                # Usar la caché que tenga respuesta válida
                cache_result = llm_cached if (llm_cached and llm_cached.get('response')) else similar_cached
                
                # Añadir información de similitud al resultado
                cache_result['semantic_match'] = {
                    'original_query': query,
                    'matched_query': similar_query['query'],
                    'similarity': similar_query['similarity']
                }
                
                response_length = len(cache_result.get('response', '')) if cache_result.get('response') else 0
                self.logger.info(f"Returning cached response for similar query with content length: {response_length}")
                if response_length == 0:
                    self.logger.warning(f"Semantic cache result has empty response! Cache result keys: {list(cache_result.keys()) if cache_result else 'None'}")
                
                return {'result': cache_result, 'source': 'semantic_cache'}
        
        # Almacenar el embedding para futuras comparaciones
        self._store_query_embedding(query, query_embedding, language)
            
        # Procesar la consulta nueva
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
    
    def clean_error_responses_from_cache(self):
        """
        Limpia respuestas de error del caché existente.
        """
        keys_to_remove = []
        
        for query_hash, entry in self.llm_cache.items():
            response = entry.get('response', '')
            if self._is_error_response(response):
                keys_to_remove.append(query_hash)
        
        for key in keys_to_remove:
            del self.llm_cache[key]
            self.logger.info(f"Eliminada respuesta de error del caché para hash: {key}")
        
        if keys_to_remove:
            self.logger.info(f"Limpieza completada: eliminadas {len(keys_to_remove)} respuestas de error del caché")
        else:
            self.logger.info("No se encontraron respuestas de error en el caché")
        
        return len(keys_to_remove)
        
    def clean_invalid_chunk_content_from_cache(self):
        """
        Limpia entradas del caché que tienen chunk_content inválido (solo nombres de archivo).
        """
        if not self.enabled:
            return 0
            
        keys_to_remove = []
        
        for key, value in self.llm_cache.items():
            sources = value.get('sources', [])
            has_invalid_chunks = False
            
            for source in sources:
                chunk_content = source.get('chunk_content', '')
                # Si el chunk_content es solo un nombre de archivo (sin contenido real)
                if chunk_content and chunk_content.endswith(('.pdf', '.docx', '.doc', '.xlsx', '.txt')):
                    # Verificar si es SOLO el nombre del archivo sin contenido adicional
                    # Un chunk válido debería tener más de 50 caracteres y contener espacios
                    if len(chunk_content) < 100 or chunk_content.count(' ') < 5:
                        has_invalid_chunks = True
                        break
            
            if has_invalid_chunks:
                keys_to_remove.append(key)
        
        # Eliminar las claves identificadas
        for key in keys_to_remove:
            del self.llm_cache[key]
            self.logger.info(f"Eliminada entrada con chunk_content inválido del caché para hash: {key}")
            
        if keys_to_remove:
            self.logger.info(f"Limpieza completada: eliminadas {len(keys_to_remove)} entradas con chunk_content inválido")
        else:
            self.logger.info("No se encontraron entradas con chunk_content inválido en el caché")
            
        return len(keys_to_remove)