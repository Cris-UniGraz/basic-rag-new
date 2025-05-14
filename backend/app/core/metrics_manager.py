from datetime import datetime, timedelta
import logging
import json
import os
import numpy as np
import csv
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import defaultdict
import threading
import time

class MetricsManager:
    """
    Sistema avanzado de métricas para monitorear el rendimiento del RAG.
    
    Esta clase implementa el patrón Singleton para asegurar una única 
    instancia que registra todas las métricas del sistema.
    
    Características:
    - Seguimiento de tasas de acierto/fallo de caché
    - Monitoreo de tiempos de respuesta
    - Conteo de operaciones por tipo
    - Registro de errores
    - Métricas de optimización de consultas
    - Estadísticas de recuperación de documentos
    - Métricas de reranking
    - Análisis de calidad de respuesta
    - Persistencia y exportación de métricas
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetricsManager, cls).__new__(cls)
                cls._instance.reset_metrics()
                cls._instance.logger = logging.getLogger(__name__)
                # Inicializar el thread de guardado automático si está configurado
                if hasattr(cls._instance, '_auto_save') and cls._instance._auto_save:
                    cls._instance._start_auto_save()
            return cls._instance
    
    def reset_metrics(self):
        """Reinicia todas las métricas a sus valores iniciales."""
        self.metrics = {
            # Métricas de caché
            'cache_hits': 0,            # Número de aciertos en caché
            'cache_misses': 0,          # Número de fallos en caché
            'exact_cache_hits': 0,      # Aciertos exactos en caché
            'semantic_cache_hits': 0,   # Aciertos semánticos en caché
            'cache_hit_rate': 0,        # Tasa de aciertos de caché
            'cache_cleanups': 0,        # Número de limpiezas de caché realizadas
            'entries_removed': 0,       # Entradas removidas durante limpiezas
            
            # Métricas de API y rendimiento
            'rate_limits': 0,           # Contador de límites de tasa alcanzados
            'api_calls': defaultdict(int),  # Llamadas a APIs por tipo
            'api_tokens': defaultdict(int), # Tokens consumidos por servicio
            'response_times': [],       # Lista de tiempos de respuesta
            'operation_counts': defaultdict(int),  # Conteo por tipo de operación
            'operation_times': defaultdict(list),  # Tiempos por tipo de operación
            
            # Métricas de errores
            'errors': defaultdict(int), # Conteo de errores por tipo
            'error_details': [],        # Detalles de errores ocurridos
            
            # Métricas de procesamiento
            'processing_times': [],     # Tiempos de procesamiento general
            'document_counts': [],      # Número de documentos procesados por consulta
            'document_sources': defaultdict(int),  # Fuentes de documentos recuperados
            'document_scores': [],      # Puntuaciones de relevancia de documentos
            
            # Métricas de consultas
            'query_lengths': [],        # Longitud de las consultas
            'query_languages': defaultdict(int),  # Idiomas de las consultas
            'query_optimizations': 0,   # Número de optimizaciones realizadas
            'optimization_time': [],    # Tiempos de optimización
            'query_similarity_scores': [], # Puntuaciones de similitud entre consultas
            'query_examples': [],       # Ejemplos de consultas (para análisis)
            
            # Métricas de embedding
            'embedding_times': [],      # Tiempos de generación de embeddings
            'embedding_dimensions': [],  # Dimensiones de embeddings generados
            'embedding_calls': 0,       # Número de llamadas a servicio de embedding
            
            # Métricas de reranking
            'reranking_times': [],      # Tiempos de reranking
            'reranking_scores': [],     # Puntuaciones de reranking
            'reranking_improvements': [], # Mejoras por reranking (cambios en orden)
            
            # Métricas LLM
            'llm_completion_times': [], # Tiempos de generación de respuestas
            'llm_tokens_input': [],     # Tokens de entrada a LLM
            'llm_tokens_output': [],    # Tokens de salida de LLM
            'llm_calls': 0,             # Número de llamadas a LLM
            
            # Métricas de recursos
            'memory_usage': [],         # Uso de memoria en momentos clave
            'cpu_usage': [],            # Uso de CPU en momentos clave
            
            # Métricas de usuarios
            'user_sessions': defaultdict(int),  # Sesiones por usuario
            'user_queries': defaultdict(int),   # Consultas por usuario
            'session_durations': []     # Duración de sesiones
        }
        
        # Inicializar valores adicionales
        self.history = {
            'queries': [],        # Historial de consultas para análisis
            'responses': [],      # Historial de respuestas para análisis
            'performances': []    # Historial de rendimiento
        }
        
        # Configuración
        self._max_history_items = 100
        self._auto_save = False
        self._auto_save_interval = 3600  # 1 hora por defecto
        self._metrics_dir = "logs/metrics"
        
        # Timestamp de inicio
        self.start_time = datetime.now()
        
        # Contadores de períodos
        self.period_start_time = self.start_time
        self.period_metrics = {}
    
    def configure(self, 
                 max_history: int = None, 
                 auto_save: bool = None,
                 auto_save_interval: int = None,
                 metrics_dir: str = None):
        """
        Configura las opciones del gestor de métricas.
        
        Args:
            max_history: Número máximo de items en el historial
            auto_save: Si debe guardar métricas automáticamente
            auto_save_interval: Intervalo en segundos para guardado automático
            metrics_dir: Directorio donde guardar las métricas
        """
        if max_history is not None:
            self._max_history_items = max_history
        
        if auto_save is not None:
            old_auto_save = self._auto_save
            self._auto_save = auto_save
            
            # Si cambiamos de False a True, iniciar el thread
            if not old_auto_save and auto_save:
                self._start_auto_save()
        
        if auto_save_interval is not None:
            self._auto_save_interval = auto_save_interval
            
        if metrics_dir is not None:
            self._metrics_dir = metrics_dir
            os.makedirs(self._metrics_dir, exist_ok=True)
    
    def _start_auto_save(self):
        """Inicia un thread para guardar métricas periódicamente."""
        def auto_save_worker():
            while self._auto_save:
                time.sleep(self._auto_save_interval)
                try:
                    self.save_metrics()
                except Exception as e:
                    self.logger.error(f"Error guardando métricas automáticamente: {e}")
        
        thread = threading.Thread(target=auto_save_worker, daemon=True)
        thread.start()
    
    def log_operation(self, operation_type: str, duration: float, success: bool, details: Dict = None):
        """
        Registra una operación realizada en el sistema.
        
        Args:
            operation_type: Tipo de operación (ej. "embedding", "retrieval", "reranking")
            duration: Duración de la operación en segundos
            success: Si la operación fue exitosa
            details: Detalles adicionales sobre la operación
        """
        self.metrics['operation_counts'][operation_type] += 1
        self.metrics['response_times'].append(duration)
        self.metrics['operation_times'][operation_type].append(duration)
        
        if not success:
            self.metrics['errors'][operation_type] += 1
            if details:
                error_info = {
                    'type': operation_type,
                    'time': datetime.now().isoformat(),
                    'duration': duration,
                    'details': details
                }
                self.metrics['error_details'].append(error_info)
    
    def log_api_call(self, api_type: str, success: bool, tokens: int = 0, duration: float = 0, rate_limited: bool = False):
        """
        Registra una llamada a API externa.
        
        Args:
            api_type: Tipo de API (ej. "openai", "cohere", "azure")
            success: Si la llamada fue exitosa
            tokens: Número de tokens consumidos
            duration: Duración de la llamada
            rate_limited: Si se alcanzó el límite de tasa
        """
        self.metrics['api_calls'][api_type] += 1
        
        if tokens > 0:
            self.metrics['api_tokens'][api_type] += tokens
        
        if duration > 0:
            self.metrics['response_times'].append(duration)
        
        if rate_limited:
            self.metrics['rate_limits'] += 1
        
        if not success:
            self.metrics['errors'][f'api_{api_type}'] += 1
    
    def log_embedding(self, model: str, query_length: int, duration: float, dimensions: int = 0):
        """
        Registra una operación de embedding.
        
        Args:
            model: Modelo de embedding utilizado
            query_length: Longitud del texto embebido
            duration: Duración de la operación
            dimensions: Dimensiones del embedding generado
        """
        self.metrics['embedding_calls'] += 1
        self.metrics['embedding_times'].append(duration)
        self.metrics['operation_counts']['embedding'] += 1
        
        if dimensions > 0:
            self.metrics['embedding_dimensions'].append(dimensions)
        
        # Actualizar también la API correspondiente
        api_type = model.split('/')[0] if '/' in model else model
        self.metrics['api_calls'][api_type] += 1
    
    def log_llm_call(self, model: str, input_tokens: int, output_tokens: int, duration: float, success: bool = True):
        """
        Registra una llamada al LLM.
        
        Args:
            model: Modelo LLM utilizado
            input_tokens: Tokens de entrada
            output_tokens: Tokens de salida generados
            duration: Duración de la operación
            success: Si la llamada fue exitosa
        """
        self.metrics['llm_calls'] += 1
        self.metrics['llm_completion_times'].append(duration)
        self.metrics['llm_tokens_input'].append(input_tokens)
        self.metrics['llm_tokens_output'].append(output_tokens)
        
        # Actualizar también la API correspondiente
        api_type = model.split('/')[0] if '/' in model else model
        self.metrics['api_calls'][api_type] += 1
        self.metrics['api_tokens'][api_type] += (input_tokens + output_tokens)
        
        if not success:
            self.metrics['errors']['llm'] += 1
    
    def log_retrieval(self, 
                     query: str, 
                     num_docs: int, 
                     duration: float, 
                     sources: List[str] = None,
                     language: str = None,
                     search_type: str = "vector"):
        """
        Registra una operación de recuperación de documentos.
        
        Args:
            query: Consulta utilizada
            num_docs: Número de documentos recuperados
            duration: Duración de la operación
            sources: Fuentes de los documentos
            language: Idioma de la consulta
            search_type: Tipo de búsqueda ("vector", "keyword", "hybrid")
        """
        self.metrics['operation_counts']['retrieval'] += 1
        self.metrics['document_counts'].append(num_docs)
        self.metrics['operation_times']['retrieval'].append(duration)
        
        if language:
            self.metrics['query_languages'][language] += 1
        
        if query:
            self.metrics['query_lengths'].append(len(query))
            
            # Guardar algunas consultas de ejemplo (limitado)
            if len(self.metrics['query_examples']) < self._max_history_items:
                self.metrics['query_examples'].append({
                    'query': query,
                    'time': datetime.now().isoformat(),
                    'language': language,
                    'num_docs': num_docs
                })
        
        if sources:
            for source in sources:
                if isinstance(source, dict) and 'source' in source:
                    source_name = source['source']
                elif isinstance(source, str):
                    source_name = source
                else:
                    source_name = 'unknown'
                    
                self.metrics['document_sources'][source_name] += 1
    
    def log_reranking(self, 
                     model: str, 
                     original_count: int, 
                     filtered_count: int, 
                     duration: float,
                     scores: List[float] = None):
        """
        Registra una operación de reranking.
        
        Args:
            model: Modelo de reranking utilizado
            original_count: Número de documentos antes del reranking
            filtered_count: Número de documentos después del reranking
            duration: Duración de la operación
            scores: Puntuaciones de reranking
        """
        self.metrics['operation_counts']['reranking'] += 1
        self.metrics['reranking_times'].append(duration)
        
        # Calcular la mejora (reducción de documentos)
        if original_count > 0:
            improvement = (original_count - filtered_count) / original_count
            self.metrics['reranking_improvements'].append(improvement)
        
        # Registrar las puntuaciones
        if scores:
            self.metrics['reranking_scores'].extend(scores)
            
            # También almacenar en document_scores para análisis general
            self.metrics['document_scores'].extend(scores)
    
    def log_query_optimization(self, processing_time: float, was_cached: bool, cache_type: str = None):
        """
        Registra una optimización de consulta.
        
        Args:
            processing_time: Tiempo que tomó optimizar la consulta
            was_cached: Si la respuesta provino del caché
            cache_type: Tipo de caché usado ("exact", "semantic", None)
        """
        self.metrics['query_optimizations'] += 1
        self.metrics['optimization_time'].append(processing_time)
        
        if was_cached:
            self.metrics['cache_hits'] += 1
            if cache_type == "exact":
                self.metrics['exact_cache_hits'] += 1
            elif cache_type == "semantic":
                self.metrics['semantic_cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
        
        # Actualizar tasa de aciertos de caché
        total_queries = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_queries > 0:
            self.metrics['cache_hit_rate'] = self.metrics['cache_hits'] / total_queries
    
    def log_rag_query(self, 
                     query: str, 
                     processing_time: float, 
                     num_sources: int,
                     from_cache: bool,
                     language: str = None,
                     user_id: str = None,
                     session_id: str = None):
        """
        Registra una consulta completa de RAG.
        
        Args:
            query: Consulta realizada
            processing_time: Tiempo total de procesamiento
            num_sources: Número de fuentes utilizadas
            from_cache: Si la respuesta vino del caché
            language: Idioma de la consulta
            user_id: ID del usuario (si aplica)
            session_id: ID de la sesión (si aplica)
        """
        self.metrics['processing_times'].append(processing_time)
        
        if query:
            self.metrics['query_lengths'].append(len(query))
        
        if language:
            self.metrics['query_languages'][language] += 1
        
        # Registrar en cache hits/misses
        if from_cache:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
            
        # Actualizar tasa de aciertos de caché
        total_queries = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_queries > 0:
            self.metrics['cache_hit_rate'] = self.metrics['cache_hits'] / total_queries
            
        # Registrar información de usuario/sesión si está disponible
        if user_id:
            self.metrics['user_queries'][user_id] += 1
            
        if session_id:
            self.metrics['user_sessions'][session_id] += 1
            
        # Guardar en historial (limitado)
        if len(self.history['queries']) < self._max_history_items:
            entry = {
                'query': query,
                'time': datetime.now().isoformat(),
                'processing_time': processing_time,
                'num_sources': num_sources,
                'from_cache': from_cache,
                'language': language
            }
            self.history['queries'].append(entry)
            
        # Añadir al histórico de rendimiento
        perf_entry = {
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'from_cache': from_cache,
            'operation': 'rag_query'
        }
        self.history['performances'].append(perf_entry)
    
    def log_error(self, error_type: str, details: str = None, component: str = None):
        """
        Registra un error en el sistema.
        
        Args:
            error_type: Tipo de error
            details: Detalles del error
            component: Componente donde ocurrió el error
        """
        err_key = f"{component}_{error_type}" if component else error_type
        self.metrics['errors'][err_key] += 1
        
        if details:
            error_info = {
                'type': error_type,
                'component': component,
                'time': datetime.now().isoformat(),
                'details': details
            }
            self.metrics['error_details'].append(error_info)
    
    def log_memory_usage(self, memory_mb: float, cpu_percent: float = None, operation: str = None):
        """
        Registra el uso de recursos del sistema.
        
        Args:
            memory_mb: Uso de memoria en MB
            cpu_percent: Porcentaje de CPU utilizado (0-100)
            operation: Operación que se estaba realizando
        """
        memory_data = {
            'memory_mb': memory_mb,
            'timestamp': datetime.now().isoformat()
        }
        
        if cpu_percent is not None:
            memory_data['cpu_percent'] = cpu_percent
            self.metrics['cpu_usage'].append(cpu_percent)
            
        if operation:
            memory_data['operation'] = operation
            
        self.metrics['memory_usage'].append(memory_data)
    
    def start_period(self):
        """Inicia un período de medición para métricas temporales."""
        self.period_start_time = datetime.now()
        self.period_metrics = {
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'api_calls': self.metrics['api_calls'].copy(),
            'errors': self.metrics['errors'].copy(),
            'llm_calls': self.metrics['llm_calls'],
            # Agregar otros contadores que queramos comparar
        }
    
    def get_period_metrics(self) -> Dict[str, Any]:
        """
        Genera métricas para el período desde que se llamó start_period().
        
        Returns:
            Diccionario con las métricas del período
        """
        period_duration = (datetime.now() - self.period_start_time).total_seconds()
        
        # Calcular diferencias para este período
        period_data = {
            'duration': period_duration,
            'cache_hits': self.metrics['cache_hits'] - self.period_metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'] - self.period_metrics['cache_misses'],
            'llm_calls': self.metrics['llm_calls'] - self.period_metrics['llm_calls'],
        }
        
        # Calcular diferencias en diccionarios anidados
        period_data['api_calls'] = {}
        for api, count in self.metrics['api_calls'].items():
            period_data['api_calls'][api] = count - self.period_metrics['api_calls'].get(api, 0)
            
        period_data['errors'] = {}
        for err, count in self.metrics['errors'].items():
            period_data['errors'][err] = count - self.period_metrics['errors'].get(err, 0)
        
        return period_data
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Genera un resumen de las métricas actuales.
        
        Returns:
            Diccionario con estadísticas resumidas
        """
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        # Calcular promedios para listas de valores
        avg_response_time = np.mean(self.metrics['response_times']) if self.metrics['response_times'] else 0
        avg_processing_time = np.mean(self.metrics['processing_times']) if self.metrics['processing_times'] else 0
        avg_embedding_time = np.mean(self.metrics['embedding_times']) if self.metrics['embedding_times'] else 0
        avg_llm_time = np.mean(self.metrics['llm_completion_times']) if self.metrics['llm_completion_times'] else 0
        
        # Calcular totales y tasas
        total_operations = sum(self.metrics['operation_counts'].values())
        total_errors = sum(self.metrics['errors'].values())
        error_rate = total_errors / total_operations if total_operations > 0 else 0
        
        # Calcular estadísticas de tokens
        total_tokens_input = sum(self.metrics['llm_tokens_input']) if self.metrics['llm_tokens_input'] else 0
        total_tokens_output = sum(self.metrics['llm_tokens_output']) if self.metrics['llm_tokens_output'] else 0
        
        # Calcular estadísticas de cache
        cache_hit_rate = self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
        semantic_hits_pct = self.metrics['semantic_cache_hits'] / self.metrics['cache_hits'] if self.metrics['cache_hits'] > 0 else 0
        
        # Construir el resumen
        return {
            'uptime_seconds': total_time,
            'uptime_formatted': str(timedelta(seconds=int(total_time))),
            'start_time': self.start_time.isoformat(),
            'current_time': datetime.now().isoformat(),
            
            # Estadísticas de operaciones
            'total_operations': total_operations,
            'operations_breakdown': dict(self.metrics['operation_counts']),
            'operations_per_second': total_operations / total_time if total_time > 0 else 0,
            
            # Estadísticas de caché
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'exact_cache_hits': self.metrics['exact_cache_hits'],
            'semantic_cache_hits': self.metrics['semantic_cache_hits'],
            'semantic_hits_percentage': semantic_hits_pct,
            
            # Estadísticas de tiempos
            'average_response_time': avg_response_time,
            'average_processing_time': avg_processing_time,
            'average_embedding_time': avg_embedding_time,
            'average_llm_time': avg_llm_time,
            
            # Estadísticas API y LLM
            'total_api_calls': dict(self.metrics['api_calls']),
            'total_tokens': {
                'input': total_tokens_input,
                'output': total_tokens_output,
                'total': total_tokens_input + total_tokens_output
            },
            'tokens_by_service': dict(self.metrics['api_tokens']),
            
            # Estadísticas de errores
            'total_errors': total_errors,
            'error_rate': error_rate,
            'errors_breakdown': dict(self.metrics['errors']),
            'rate_limit_hits': self.metrics['rate_limits'],
            
            # Estadísticas de consultas y documentos
            'query_languages': dict(self.metrics['query_languages']),
            'document_sources': dict(self.metrics['document_sources']),
            'average_documents_per_query': np.mean(self.metrics['document_counts']) if self.metrics['document_counts'] else 0,
        }
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """
        Genera estadísticas detalladas para análisis.
        
        Returns:
            Diccionario con estadísticas detalladas y distribuciones
        """
        # Función auxiliar para calcular percentiles
        def get_percentiles(data_list, percentiles=[50, 90, 95, 99]):
            if not data_list:
                return {f"p{p}": 0 for p in percentiles}
            return {f"p{p}": float(np.percentile(data_list, p)) for p in percentiles}
        
        # Calcular estadísticas para tiempos de respuesta
        response_times_stats = {
            'count': len(self.metrics['response_times']),
            'mean': float(np.mean(self.metrics['response_times'])) if self.metrics['response_times'] else 0,
            'median': float(np.median(self.metrics['response_times'])) if self.metrics['response_times'] else 0,
            'min': float(np.min(self.metrics['response_times'])) if self.metrics['response_times'] else 0,
            'max': float(np.max(self.metrics['response_times'])) if self.metrics['response_times'] else 0,
            'std': float(np.std(self.metrics['response_times'])) if self.metrics['response_times'] else 0,
            **get_percentiles(self.metrics['response_times'])
        }
        
        # Estadísticas para tiempos de procesamiento
        processing_times_stats = {
            'count': len(self.metrics['processing_times']),
            'mean': float(np.mean(self.metrics['processing_times'])) if self.metrics['processing_times'] else 0,
            'median': float(np.median(self.metrics['processing_times'])) if self.metrics['processing_times'] else 0,
            'min': float(np.min(self.metrics['processing_times'])) if self.metrics['processing_times'] else 0,
            'max': float(np.max(self.metrics['processing_times'])) if self.metrics['processing_times'] else 0,
            'std': float(np.std(self.metrics['processing_times'])) if self.metrics['processing_times'] else 0,
            **get_percentiles(self.metrics['processing_times'])
        }
        
        # Estadísticas por tipo de operación
        operation_stats = {}
        for op_type, times in self.metrics['operation_times'].items():
            if times:
                operation_stats[op_type] = {
                    'count': len(times),
                    'mean': float(np.mean(times)),
                    'median': float(np.median(times)),
                    'min': float(np.min(times)),
                    'max': float(np.max(times)),
                    'std': float(np.std(times)),
                    **get_percentiles(times)
                }
        
        # Combinar todo en un único diccionario de estadísticas
        return {
            'response_times': response_times_stats,
            'processing_times': processing_times_stats,
            'operation_times': operation_stats,
            'document_counts': {
                'count': len(self.metrics['document_counts']),
                'mean': float(np.mean(self.metrics['document_counts'])) if self.metrics['document_counts'] else 0,
                'median': float(np.median(self.metrics['document_counts'])) if self.metrics['document_counts'] else 0,
                'min': float(np.min(self.metrics['document_counts'])) if self.metrics['document_counts'] else 0,
                'max': float(np.max(self.metrics['document_counts'])) if self.metrics['document_counts'] else 0
            },
            'query_lengths': {
                'count': len(self.metrics['query_lengths']),
                'mean': float(np.mean(self.metrics['query_lengths'])) if self.metrics['query_lengths'] else 0,
                'median': float(np.median(self.metrics['query_lengths'])) if self.metrics['query_lengths'] else 0,
                'min': float(np.min(self.metrics['query_lengths'])) if self.metrics['query_lengths'] else 0,
                'max': float(np.max(self.metrics['query_lengths'])) if self.metrics['query_lengths'] else 0
            },
            'document_scores': {
                'count': len(self.metrics['document_scores']),
                'mean': float(np.mean(self.metrics['document_scores'])) if self.metrics['document_scores'] else 0,
                'median': float(np.median(self.metrics['document_scores'])) if self.metrics['document_scores'] else 0,
                'min': float(np.min(self.metrics['document_scores'])) if self.metrics['document_scores'] else 0,
                'max': float(np.max(self.metrics['document_scores'])) if self.metrics['document_scores'] else 0,
                **get_percentiles(self.metrics['document_scores'])
            },
            'llm_tokens': {
                'input': {
                    'count': len(self.metrics['llm_tokens_input']),
                    'mean': float(np.mean(self.metrics['llm_tokens_input'])) if self.metrics['llm_tokens_input'] else 0,
                    'median': float(np.median(self.metrics['llm_tokens_input'])) if self.metrics['llm_tokens_input'] else 0,
                    'min': float(np.min(self.metrics['llm_tokens_input'])) if self.metrics['llm_tokens_input'] else 0,
                    'max': float(np.max(self.metrics['llm_tokens_input'])) if self.metrics['llm_tokens_input'] else 0,
                    'total': sum(self.metrics['llm_tokens_input']) if self.metrics['llm_tokens_input'] else 0
                },
                'output': {
                    'count': len(self.metrics['llm_tokens_output']),
                    'mean': float(np.mean(self.metrics['llm_tokens_output'])) if self.metrics['llm_tokens_output'] else 0,
                    'median': float(np.median(self.metrics['llm_tokens_output'])) if self.metrics['llm_tokens_output'] else 0,
                    'min': float(np.min(self.metrics['llm_tokens_output'])) if self.metrics['llm_tokens_output'] else 0,
                    'max': float(np.max(self.metrics['llm_tokens_output'])) if self.metrics['llm_tokens_output'] else 0,
                    'total': sum(self.metrics['llm_tokens_output']) if self.metrics['llm_tokens_output'] else 0
                }
            }
        }
    
    def save_metrics(self, filename: str = None) -> str:
        """
        Guarda las métricas actuales en un archivo JSON.
        
        Args:
            filename: Nombre del archivo a usar (si no se proporciona, se genera uno)
            
        Returns:
            Ruta al archivo guardado
        """
        # Crear directorio de métricas si no existe
        os.makedirs(self._metrics_dir, exist_ok=True)
        
        # Generar nombre de archivo basado en la fecha si no se proporciona uno
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        # Ruta completa al archivo
        filepath = os.path.join(self._metrics_dir, filename)
        
        # Preparar datos para guardar (convertir a formato serializable)
        metrics_copy = self._prepare_data_for_export(self.metrics)
        summary = self.get_summary()
        detailed_stats = self.get_detailed_statistics()
        
        # Datos a guardar
        data = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'detailed_stats': detailed_stats,
            'metrics': metrics_copy
        }
        
        # Guardar como JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Métricas guardadas en {filepath}")
        return filepath
    
    def export_to_csv(self, directory: str = None) -> Dict[str, str]:
        """
        Exporta métricas seleccionadas a archivos CSV para análisis.
        
        Args:
            directory: Directorio donde guardar los archivos CSV
            
        Returns:
            Diccionario con las rutas a los archivos generados
        """
        # Establecer directorio
        if not directory:
            directory = os.path.join(self._metrics_dir, 'csv_export')
        
        os.makedirs(directory, exist_ok=True)
        
        # Timestamp para nombres de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Diccionario para almacenar las rutas de archivos
        csv_files = {}
        
        # Exportar tiempos de operación
        operation_times_path = os.path.join(directory, f"operation_times_{timestamp}.csv")
        with open(operation_times_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['operation_type', 'duration'])
            
            for op_type, times in self.metrics['operation_times'].items():
                for duration in times:
                    writer.writerow([op_type, duration])
        
        csv_files['operation_times'] = operation_times_path
        
        # Exportar consultas de ejemplo
        if self.metrics['query_examples']:
            queries_path = os.path.join(directory, f"queries_{timestamp}.csv")
            with open(queries_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['query', 'time', 'language', 'num_docs'])
                
                for q in self.metrics['query_examples']:
                    writer.writerow([
                        q.get('query', ''),
                        q.get('time', ''),
                        q.get('language', ''),
                        q.get('num_docs', 0)
                    ])
            
            csv_files['queries'] = queries_path
        
        # Exportar histórico de rendimiento
        if self.history['performances']:
            perf_path = os.path.join(directory, f"performance_history_{timestamp}.csv")
            with open(perf_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'processing_time', 'from_cache', 'operation'])
                
                for p in self.history['performances']:
                    writer.writerow([
                        p.get('timestamp', ''),
                        p.get('processing_time', 0),
                        p.get('from_cache', False),
                        p.get('operation', '')
                    ])
            
            csv_files['performance'] = perf_path
            
        # Exportar conteos de errores
        if self.metrics['errors']:
            errors_path = os.path.join(directory, f"errors_{timestamp}.csv")
            with open(errors_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['error_type', 'count'])
                
                for error_type, count in self.metrics['errors'].items():
                    writer.writerow([error_type, count])
            
            csv_files['errors'] = errors_path
        
        self.logger.info(f"Exportados {len(csv_files)} archivos CSV a {directory}")
        return csv_files
    
    def _prepare_data_for_export(self, data):
        """
        Prepara datos para exportación, convirtiendo a formatos serializables.
        
        Args:
            data: Los datos a preparar
            
        Returns:
            Datos en formato serializable
        """
        if isinstance(data, dict):
            return {k: self._prepare_data_for_export(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_data_for_export(item) for item in data]
        elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(data)
        elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
            return float(data)
        elif isinstance(data, (np.ndarray,)):
            return self._prepare_data_for_export(data.tolist())
        elif isinstance(data, (defaultdict)):
            return self._prepare_data_for_export(dict(data))
        else:
            return data