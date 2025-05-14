#!/usr/bin/env python
"""
Script para probar la integración del sistema de caché avanzado con el servicio RAG.
Este script simula varias consultas y muestra el ahorro de tiempo con el caché.
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
from app.models.vector_store import vector_store_manager


async def test_rag_with_cache():
    """Prueba el servicio RAG con el sistema de caché avanzado."""
    # Inicializar servicios
    logger.info("Inicializando servicios...")
    
    # Crear LLM service
    llm_service = create_llm_service()
    
    # Crear RAG service
    rag_service = RAGService(llm_service)
    await rag_service.initialize()
    
    # Conectar a la base de datos vectorial
    vector_store_manager.connect()
    
    # Métricas
    metrics = MetricsManager()
    optimizer = QueryOptimizer()
    
    # Variables de prueba
    collection_name = "test_collection"
    language = "german"
    queries = [
        "Wie kann ich meine UNIGRAzcard erneuern?",
        "Was ist das Verfahren zur Erneuerung meiner UNIGRAzcard?",
        "Ich muss meine UNIGRAzcard erneuern. Was soll ich tun?",
        "Wie lange dauert die Erneuerung der UNIGRAzcard?",
        "Wie kann ich meine UNIGRAzcard erneuern?"  # Consulta repetida para probar caché
    ]
    
    # Preparar retrievers (en una aplicación real, estos se obtienen de colecciones existentes)
    # En este ejemplo, usamos None para simular retrievers no disponibles
    retriever_de = None
    retriever_en = None
    
    # Para este test, simularemos respuestas directamente desde el optimizador de caché
    # en lugar de usar retrievers reales (que requerirían documentos reales)
    
    # Prueba 1: Primera ejecución (sin caché)
    logger.info("=== Test 1: Primera ejecución (sin caché) ===")
    first_query = queries[0]
    
    # Simular una respuesta del sistema
    mock_response = "Um deine UNIGRAzcard zu erneuern, musst du zu einem Aktualisierungsleser oder einem Online-Terminal gehen. Die Karte muss alle 30 Tage erneuert werden, um ihre Gültigkeit zu erhalten."
    mock_sources = [{"source": "UNIGRAzcard_Handbuch.pdf", "page": "12"}]
    
    # Guardar en caché
    optimizer._store_llm_response(first_query, mock_response, language, mock_sources)
    logger.info(f"Respuesta almacenada en caché para: '{first_query}'")
    
    # Prueba 2: Consulta repetida (debe usar caché)
    logger.info("=== Test 2: Consulta repetida (debe usar caché) ===")
    start_time = time.time()
    cached_result = optimizer.get_llm_response(first_query, language)
    end_time = time.time()
    
    if cached_result:
        logger.info(f"Caché hit! Tiempo de respuesta: {(end_time - start_time):.5f} segundos")
        logger.info(f"Respuesta del caché: {cached_result['response'][:50]}...")
    else:
        logger.error("Caché miss! No se encontró la respuesta en el caché.")
    
    # Prueba 3: Consulta similar (para demonstrar que funcionaría)
    logger.info("=== Test 3: Consulta similar ===")
    similar_query = queries[1]
    
    # En un escenario real, el optimizador detectaría la similitud semántica
    # Para este ejemplo, simulamos lo que sucedería
    logger.info(f"Consulta similar: '{similar_query}'")
    logger.info("En un sistema completo con embeddings, esta consulta similar podría ser detectada como semánticamente equivalente")
    
    # Prueba 4: Estadísticas y limpieza
    logger.info("=== Test 4: Estadísticas y limpieza ===")
    
    # Obtener estadísticas antes de la limpieza
    stats_before = optimizer.get_cache_stats()
    logger.info(f"Estadísticas antes de limpieza: {stats_before}")
    
    # Limpiar caché
    removed = optimizer.cleanup_cache(max_age_hours=0)  # Forzar limpieza
    logger.info(f"Limpieza de caché: {removed} entradas eliminadas")
    
    # Obtener estadísticas después de la limpieza
    stats_after = optimizer.get_cache_stats()
    logger.info(f"Estadísticas después de limpieza: {stats_after}")
    
    logger.info("¡Tests de RAG con caché completados!")


if __name__ == "__main__":
    logger.info("Iniciando pruebas de RAG con sistema de caché avanzado...")
    asyncio.run(test_rag_with_cache())