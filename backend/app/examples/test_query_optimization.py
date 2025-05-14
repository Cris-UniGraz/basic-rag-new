#!/usr/bin/env python
"""
Script para probar la optimización de consultas y caché semántico.
Este script demuestra cómo funciona el sistema de optimización cuando
recibe consultas similares pero no idénticas.
"""

import asyncio
import time
from loguru import logger
import sys
import os
import numpy as np

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
from app.core.config import settings


async def test_semantic_query_matching():
    """Prueba la funcionalidad de coincidencia semántica entre consultas."""
    # Inicializar servicios
    logger.info("Inicializando servicios...")
    
    # Configurar optimizer y metrics
    optimizer = QueryOptimizer()
    metrics = MetricsManager()
    
    # Inicializar embedding manager
    embedding_manager.initialize_models(
        "intfloat/multilingual-e5-large", 
        "intfloat/multilingual-e5-large"
    )
    
    # Preparar modelo de embeddings según idioma
    language = "spanish"
    embedding_model = embedding_manager.get_model_for_language(language)
    
    # Colección de consultas semánticamente similares para probar
    similar_queries = [
        "¿Cómo puedo renovar mi UNIGRAzcard?",
        "¿Cuál es el proceso para renovar una UNIGRAzcard?",
        "¿Dónde debo ir para renovar mi UNIGRAzcard?",
        "Necesito renovar mi UNIGRAzcard, ¿qué debo hacer?",
        "¿Con qué frecuencia hay que renovar la UNIGRAzcard?",
        "Mi UNIGRAzcard está caducada, ¿cómo la renuevo?"
    ]
    
    # Colección de consultas diferentes para contrastes
    different_queries = [
        "¿Dónde está la biblioteca central?",
        "¿Cuáles son los horarios de atención?",
        "¿Quién es el rector de la universidad?",
        "¿Cómo me inscribo en un curso?"
    ]
    
    # Test 1: Generar embedding para la primera consulta como referencia
    logger.info("=== Test 1: Generando embedding de referencia ===")
    reference_query = similar_queries[0]
    
    # Crear respuesta simulada para la consulta de referencia
    test_response = "Para renovar tu UNIGRAzcard, debes acudir a un lector de actualización o a una terminal en línea. La tarjeta debe renovarse cada 30 días para mantener su validez."
    test_sources = [{"source": "UNIGRAzcard_Handbuch.pdf", "page": "12"}]
    
    # Guardar en caché
    optimizer._store_llm_response(reference_query, test_response, language, test_sources)
    
    # Generar y almacenar embedding para la consulta de referencia
    optimizer_result = await optimizer.optimize_query(reference_query, language, embedding_model)
    logger.info(f"Consulta de referencia: '{reference_query}'")
    logger.info(f"Embedding almacenado con éxito: {optimizer_result['source']}")
    
    # Test 2: Probar similitud con otras consultas similares
    logger.info("\n=== Test 2: Probando similitud con consultas semánticamente cercanas ===")
    for i, query in enumerate(similar_queries[1:], 1):
        logger.info(f"\nConsulta similar #{i}: '{query}'")
        
        start_time = time.time()
        result = await optimizer.optimize_query(query, language, embedding_model)
        end_time = time.time()
        
        # Analizar resultado
        if result['source'] == 'semantic_cache':
            similarity = result['result'].get('semantic_match', {}).get('similarity', 0)
            logger.info(f"✅ COINCIDENCIA SEMÁNTICA detectada con similitud: {similarity:.4f}")
            logger.info(f"Tiempo de procesamiento: {(end_time - start_time):.5f} segundos")
            logger.info(f"Consulta coincidente: '{result['result'].get('semantic_match', {}).get('matched_query', '')}'")
        else:
            logger.info(f"❌ No se detectó coincidencia semántica. Fuente: {result['source']}")
            logger.info(f"Tiempo de procesamiento: {(end_time - start_time):.5f} segundos")
    
    # Test 3: Probar con consultas diferentes (no deberían coincidir)
    logger.info("\n=== Test 3: Probando con consultas diferentes (no deberían coincidir) ===")
    for i, query in enumerate(different_queries, 1):
        logger.info(f"\nConsulta diferente #{i}: '{query}'")
        
        start_time = time.time()
        result = await optimizer.optimize_query(query, language, embedding_model)
        end_time = time.time()
        
        # Analizar resultado
        if result['source'] == 'semantic_cache':
            similarity = result['result'].get('semantic_match', {}).get('similarity', 0)
            logger.info(f"⚠️ Falso positivo! Similitud: {similarity:.4f}")
        else:
            logger.info(f"✅ Correctamente identificada como consulta diferente. Fuente: {result['source']}")
        
        logger.info(f"Tiempo de procesamiento: {(end_time - start_time):.5f} segundos")
    
    # Test 4: Probar umbrales de similitud
    logger.info("\n=== Test 4: Probando diferentes umbrales de similitud ===")
    test_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    test_query = "¿Dónde puedo actualizar mi UNIGRAzcard?"
    
    # Generar embedding para la consulta de prueba
    if hasattr(embedding_model, 'aembed_query'):
        test_embedding = await embedding_model.aembed_query(test_query)
    else:
        test_embedding = embedding_model.embed_query(test_query)
    
    # Recuperar el embedding de referencia
    reference_embedding = optimizer.query_embeddings.get(
        optimizer._generate_query_hash(reference_query), {}
    ).get('embedding')
    
    if reference_embedding is not None:
        logger.info(f"Consulta de prueba: '{test_query}'")
        logger.info(f"Consulta de referencia: '{reference_query}'")
        
        # Calcular similitud real
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(
            test_embedding.reshape(1, -1), 
            reference_embedding.reshape(1, -1)
        )[0][0]
        
        logger.info(f"Similitud real entre consultas: {similarity:.4f}")
        
        # Probar diferentes umbrales
        for threshold in test_thresholds:
            # Guardar umbral original
            original_threshold = optimizer.query_similarity_threshold
            
            # Establecer nuevo umbral para la prueba
            optimizer.query_similarity_threshold = threshold
            
            # Ejecutar prueba
            result = await optimizer.optimize_query(test_query, language, embedding_model)
            
            # Verificar resultado
            if result['source'] == 'semantic_cache':
                logger.info(f"Umbral {threshold:.2f}: COINCIDENCIA ✅")
            else:
                logger.info(f"Umbral {threshold:.2f}: NO COINCIDENCIA ❌")
            
            # Restaurar umbral original
            optimizer.query_similarity_threshold = original_threshold
    else:
        logger.error("No se pudo recuperar el embedding de referencia")
    
    # Test 5: Estadísticas finales
    logger.info("\n=== Test 5: Estadísticas finales ===")
    
    # Obtener estadísticas de caché
    cache_stats = optimizer.get_cache_stats()
    logger.info(f"Estadísticas de caché: {cache_stats}")
    
    # Obtener métricas
    cache_hits = metrics.metrics.get('cache_hits', 0)
    cache_misses = metrics.metrics.get('cache_misses', 0)
    hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
    
    logger.info(f"Cache hits: {cache_hits}")
    logger.info(f"Cache misses: {cache_misses}")
    logger.info(f"Hit rate: {hit_rate:.2%}")
    
    # Similitudes registradas
    similarities = metrics.metrics.get('query_similarity_scores', [])
    if similarities:
        avg_similarity = sum(similarities) / len(similarities)
        logger.info(f"Similitud promedio: {avg_similarity:.4f}")
        logger.info(f"Similitud mínima: {min(similarities):.4f}")
        logger.info(f"Similitud máxima: {max(similarities):.4f}")
    
    logger.info("¡Tests de optimización de consultas completados!")


if __name__ == "__main__":
    logger.info("Iniciando pruebas de optimización de consultas...")
    asyncio.run(test_semantic_query_matching())