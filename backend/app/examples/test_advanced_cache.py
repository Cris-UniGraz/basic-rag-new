#!/usr/bin/env python
"""
Script para probar el sistema de caché avanzado implementado en el proyecto basic-rag-new.
Este script demuestra cómo el sistema puede:
1. Almacenar y recuperar respuestas del caché
2. Manejar consultas similares
3. Proporcionar métricas de rendimiento
"""

import asyncio
import time
from loguru import logger
import sys
import os

# Configurar logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Añadir la ruta del proyecto al path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.core.query_optimizer import QueryOptimizer
from app.core.metrics_manager import MetricsManager
from app.core.embedding_manager import embedding_manager
from app.services.rag_service import RAGService
from app.services.llm_service import create_llm_service


async def test_cache_functionality():
    """Prueba el funcionamiento básico del caché."""
    # Inicializar servicios
    logger.info("Inicializando servicios...")
    
    # Crear LLM service
    llm_service = create_llm_service()
    
    # Crear RAG service
    rag_service = RAGService(llm_service)
    
    # Inicializar embedding manager
    embedding_manager.initialize_models(
        "intfloat/multilingual-e5-large", 
        "intfloat/multilingual-e5-large"
    )
    
    # Obtener métricas y optimizador
    metrics = MetricsManager()
    optimizer = QueryOptimizer()
    
    # Test 1: Almacenar una respuesta en caché
    logger.info("Test 1: Almacenando respuesta en caché...")
    test_query = "¿Cuál es el proceso para renovar una UNIGRAzcard?"
    test_response = "Para renovar tu UNIGRAzcard, debes acudir a un lector de actualización o a una terminal en línea. La tarjeta debe renovarse cada 30 días para mantener su validez."
    test_sources = [
        {"source": "documento1.pdf", "page": "5"}
    ]
    
    optimizer._store_llm_response(test_query, test_response, "spanish", test_sources)
    logger.info(f"Respuesta almacenada en caché. Tamaño del caché: {len(optimizer.llm_cache)}")
    
    # Test 2: Recuperar respuesta del caché
    logger.info("Test 2: Recuperando respuesta del caché...")
    cached_result = optimizer.get_llm_response(test_query, "spanish")
    
    if cached_result:
        logger.info(f"Caché hit! Respuesta recuperada: {cached_result['response'][:50]}...")
    else:
        logger.error("Caché miss! No se encontró la respuesta en el caché.")
    
    # Test 3: Probar con consulta ligeramente diferente
    logger.info("Test 3: Probando con consulta similar...")
    similar_query = "¿Cómo puedo renovar mi UNIGRAzcard?"
    
    # Simular embedding para la consulta similar
    embedding_model = embedding_manager.get_model_for_language("spanish")
    await optimizer.optimize_query(similar_query, "spanish", embedding_model)
    
    # Test 4: Probar la limpieza automática del caché
    logger.info("Test 4: Probando limpieza automática del caché...")
    # Forzar limpieza manual
    removed = optimizer.cleanup_old_entries()
    logger.info(f"Se eliminaron {removed} entradas antiguas del caché")
    
    # Test 5: Métricas del caché
    logger.info("Test 5: Métricas del sistema de caché")
    logger.info(f"Métricas actuales:\n")
    logger.info(f"Cache hits: {metrics.metrics['cache_hits']}")
    logger.info(f"Cache misses: {metrics.metrics['cache_misses']}")
    logger.info(f"Operaciones de optimización: {metrics.metrics['query_optimizations']}")
    
    # Test 6: Estadísticas del caché
    logger.info("Test 6: Estadísticas del caché")
    stats = optimizer.get_cache_stats()
    logger.info(f"Estadísticas del caché:\n{stats}")
    
    logger.info("¡Tests completados!")


if __name__ == "__main__":
    logger.info("Iniciando pruebas del sistema de caché avanzado...")
    asyncio.run(test_cache_functionality())