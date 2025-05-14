from datetime import datetime
import logging
from typing import Dict, Any
from collections import defaultdict

class MetricsManager:
    """
    Sistema de métricas para monitorear el rendimiento del RAG.
    
    Esta clase implementa el patrón Singleton para asegurar una única 
    instancia que registra todas las métricas del sistema.
    
    Características:
    - Seguimiento de tasas de acierto/fallo de caché
    - Monitoreo de tiempos de respuesta
    - Conteo de operaciones por tipo
    - Registro de errores
    - Métricas de optimización de consultas
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsManager, cls).__new__(cls)
            cls._instance.reset_metrics()
            cls._instance.logger = logging.getLogger(__name__)
        return cls._instance
    
    def reset_metrics(self):
        """Reinicia todas las métricas a sus valores iniciales."""
        self.metrics = {
            'cache_hits': 0,            # Número de aciertos en caché
            'cache_misses': 0,          # Número de fallos en caché
            'rate_limits': 0,           # Contador de límites de tasa alcanzados
            'api_calls': 0,             # Número total de llamadas a APIs externas
            'response_times': [],       # Lista de tiempos de respuesta
            'operation_counts': defaultdict(int),  # Conteo por tipo de operación
            'errors': defaultdict(int), # Conteo de errores por tipo
            'processing_times': [],     # Tiempos de procesamiento general
            'document_counts': [],      # Número de documentos procesados por consulta
            'query_lengths': [],        # Longitud de las consultas
            'query_optimizations': 0,   # Número de optimizaciones realizadas
            'optimization_time': [],    # Tiempos de optimización
            'cache_hit_rate': 0,        # Tasa de aciertos de caché
            'query_similarity_scores': [], # Puntuaciones de similitud entre consultas
            'cache_cleanups': 0,        # Número de limpiezas de caché realizadas
            'entries_removed': 0        # Entradas removidas durante limpiezas
        }
        self.start_time = datetime.now()

    def log_operation(self, operation_type: str, duration: float, success: bool):
        """
        Registra una operación realizada en el sistema.
        
        Args:
            operation_type: Tipo de operación (ej. "embedding", "retrieval", "reranking")
            duration: Duración de la operación en segundos
            success: Si la operación fue exitosa
        """
        self.metrics['operation_counts'][operation_type] += 1
        self.metrics['response_times'].append(duration)
        if not success:
            self.metrics['errors'][operation_type] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Genera un resumen de las métricas actuales.
        
        Returns:
            Diccionario con estadísticas resumidas
        """
        total_time = (datetime.now() - self.start_time).total_seconds()
        avg_response_time = sum(self.metrics['response_times']) / len(self.metrics['response_times']) if self.metrics['response_times'] else 0
        
        return {
            'total_operations': sum(self.metrics['operation_counts'].values()),
            'cache_hit_rate': self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0,
            'average_response_time': avg_response_time,
            'error_rate': sum(self.metrics['errors'].values()) / sum(self.metrics['operation_counts'].values()) if sum(self.metrics['operation_counts'].values()) > 0 else 0,
            'total_time': total_time,
            'total_api_calls': self.metrics['api_calls'],
            'rate_limit_hits': self.metrics['rate_limits']
        }
    
    def log_query_optimization(self, processing_time: float, was_cached: bool):
        """
        Registra una optimización de consulta.
        
        Args:
            processing_time: Tiempo que tomó optimizar la consulta
            was_cached: Si la respuesta provino del caché
        """
        self.metrics['optimization_time'].append(processing_time)
        if was_cached:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
        
        # Actualizar tasa de aciertos de caché
        total_queries = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_queries > 0:
            self.metrics['cache_hit_rate'] = self.metrics['cache_hits'] / total_queries