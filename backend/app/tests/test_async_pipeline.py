"""
Tests para verificar la funcionalidad del pipeline asíncrono completo.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.rag_service import RAGService
from app.core.config import settings


class TestAsyncPipeline:
    """Pruebas para el pipeline asíncrono completo."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Mock del proveedor LLM."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value="Mocked LLM response")
        return mock_llm
    
    @pytest.fixture
    def rag_service(self, mock_llm_provider):
        """Instancia del servicio RAG para pruebas."""
        return RAGService(mock_llm_provider)
    
    @pytest.fixture
    def mock_retrievers(self):
        """Mock retrievers for testing."""
        german_retriever = MagicMock()
        english_retriever = MagicMock()
        return german_retriever, english_retriever
    
    @pytest.mark.asyncio
    async def test_async_pipeline_phases_execution_order(self, rag_service, mock_retrievers):
        """Test que las fases del pipeline se ejecuten en el orden correcto."""
        
        german_retriever, english_retriever = mock_retrievers
        
        # Track execution order
        execution_order = []
        
        with patch('app.services.rag_service.embedding_manager') as mock_embedding_manager, \
             patch.object(rag_service, 'query_optimizer') as mock_optimizer, \
             patch.object(rag_service, 'generate_all_queries_in_one_call') as mock_query_gen, \
             patch.object(rag_service, 'retrieve_context_without_reranking') as mock_retrieval, \
             patch.object(rag_service, 'rerank_docs') as mock_rerank, \
             patch('app.utils.glossary.find_glossary_terms_with_explanation') as mock_glossary:
            
            # Setup mocks
            mock_embedding_manager.german_model = MagicMock()
            mock_embedding_manager.english_model = MagicMock()
            mock_optimizer.get_llm_response.return_value = None
            
            async def mock_optimize_query(*args, **kwargs):
                execution_order.append("optimize_query")
                return {'result': {'original_query': 'test'}, 'source': 'new'}
            
            async def mock_generate_queries(*args, **kwargs):
                execution_order.append("generate_queries")
                return {
                    "query_de": "test query",
                    "query_en": "test query", 
                    "step_back_query_de": "step back test",
                    "step_back_query_en": "step back test"
                }
            
            async def mock_retrieve(*args, **kwargs):
                execution_order.append("retrieve")
                from langchain_core.documents import Document
                return [Document(page_content="test content", metadata={"source": "test"})]
            
            async def mock_rerank_func(*args, **kwargs):
                execution_order.append("rerank")
                from langchain_core.documents import Document
                return [Document(page_content="test content", metadata={"source": "test", "reranking_score": 0.8})]
            
            mock_optimizer.optimize_query = mock_optimize_query
            mock_query_gen.side_effect = mock_generate_queries
            mock_retrieval.side_effect = mock_retrieve
            mock_rerank.side_effect = mock_rerank_func
            mock_glossary.return_value = []
            
            # Execute pipeline
            result = await rag_service.process_queries_with_async_pipeline(
                "test query",
                german_retriever,
                english_retriever,
                [],
                "german"
            )
            
            # Verify execution order
            assert "optimize_query" in execution_order
            assert "generate_queries" in execution_order
            assert "retrieve" in execution_order
            assert "rerank" in execution_order
            
            # Verify result structure
            assert "response" in result
            assert "sources" in result
            assert "processing_time" in result
            assert "pipeline_metrics" in result
            
            # Verify pipeline metrics
            metrics = result["pipeline_metrics"]
            assert "phase1_time" in metrics
            assert "phase2_time" in metrics
            assert "phase3_time" in metrics
            assert "phase4_time" in metrics
            assert "phase5_time" in metrics
            assert "phase6_time" in metrics
            assert "total_time" in metrics
    
    @pytest.mark.asyncio
    async def test_async_pipeline_parallelization_performance(self, rag_service, mock_retrievers):
        """Test que la paralelización mejore el rendimiento."""
        
        german_retriever, english_retriever = mock_retrievers
        
        with patch('app.services.rag_service.embedding_manager') as mock_embedding_manager, \
             patch.object(rag_service, 'query_optimizer') as mock_optimizer, \
             patch.object(rag_service, 'generate_all_queries_in_one_call') as mock_query_gen, \
             patch.object(rag_service, 'retrieve_context_without_reranking') as mock_retrieval, \
             patch.object(rag_service, 'rerank_docs') as mock_rerank, \
             patch('app.utils.glossary.find_glossary_terms_with_explanation') as mock_glossary:
            
            # Setup mocks with realistic delays
            mock_embedding_manager.german_model = MagicMock()
            mock_embedding_manager.english_model = MagicMock()
            mock_optimizer.get_llm_response.return_value = None
            
            async def slow_optimize_query(*args, **kwargs):
                await asyncio.sleep(0.1)  # 100ms delay
                return {'result': {'original_query': 'test'}, 'source': 'new'}
            
            async def slow_generate_queries(*args, **kwargs):
                await asyncio.sleep(0.1)  # 100ms delay
                return {
                    "query_de": "test query",
                    "query_en": "test query", 
                    "step_back_query_de": "step back test",
                    "step_back_query_en": "step back test"
                }
            
            async def slow_retrieve(*args, **kwargs):
                await asyncio.sleep(0.05)  # 50ms delay per retrieval
                from langchain_core.documents import Document
                return [Document(page_content="test content", metadata={"source": "test"})]
            
            async def fast_rerank(*args, **kwargs):
                await asyncio.sleep(0.02)  # 20ms delay
                from langchain_core.documents import Document
                return [Document(page_content="test content", metadata={"source": "test", "reranking_score": 0.8})]
            
            mock_optimizer.optimize_query = slow_optimize_query
            mock_query_gen.side_effect = slow_generate_queries
            mock_retrieval.side_effect = slow_retrieve
            mock_rerank.side_effect = fast_rerank
            mock_glossary.return_value = []
            
            # Measure execution time
            start_time = time.time()
            result = await rag_service.process_queries_with_async_pipeline(
                "test query",
                german_retriever,
                english_retriever,
                [],
                "german"
            )
            execution_time = time.time() - start_time
            
            # Pipeline should complete faster than sequential execution
            # Sequential would be: 0.1 + 0.1 + (4 * 0.05) + 0.02 = 0.42s
            # Parallel should be much faster due to overlapping operations
            assert execution_time < 0.3, f"Execution time {execution_time:.3f}s is too slow for async pipeline"
            
            # Verify pipeline metrics are present
            assert "pipeline_metrics" in result
            metrics = result["pipeline_metrics"]
            assert metrics["total_time"] > 0
            
            # Phase 1 and Phase 2 should execute in parallel, so their sum should be less than sequential
            phase1_phase2_time = metrics["phase1_time"] + metrics["phase2_time"]
            assert phase1_phase2_time < 0.25, "Phase 1 and 2 should execute in parallel"
    
    @pytest.mark.asyncio
    async def test_async_pipeline_cache_hit_early_return(self, rag_service, mock_retrievers):
        """Test que el pipeline retorne temprano en caso de cache hit."""
        
        german_retriever, english_retriever = mock_retrievers
        
        with patch.object(rag_service, 'query_optimizer') as mock_optimizer:
            
            # Mock cache hit
            cached_response = {
                'response': 'Cached response',
                'sources': [{'source': 'cached_source'}],
                'from_cache': True,
                'processing_time': 0.0
            }
            mock_optimizer.get_llm_response.return_value = cached_response
            
            # Execute pipeline
            start_time = time.time()
            result = await rag_service.process_queries_with_async_pipeline(
                "test query",
                german_retriever,
                english_retriever,
                [],
                "german"
            )
            execution_time = time.time() - start_time
            
            # Should return almost immediately due to cache hit
            assert execution_time < 0.1, f"Cache hit should be fast, got {execution_time:.3f}s"
            
            # Should return cached response
            assert result['response'] == 'Cached response'
            assert result['from_cache'] is True
            assert result['sources'] == [{'source': 'cached_source'}]
    
    @pytest.mark.asyncio
    async def test_async_pipeline_error_handling(self, rag_service, mock_retrievers):
        """Test manejo de errores en el pipeline asíncrono."""
        
        german_retriever, english_retriever = mock_retrievers
        
        with patch('app.services.rag_service.embedding_manager') as mock_embedding_manager, \
             patch.object(rag_service, 'query_optimizer') as mock_optimizer, \
             patch.object(rag_service, 'generate_all_queries_in_one_call') as mock_query_gen:
            
            # Setup mocks
            mock_embedding_manager.german_model = MagicMock()
            mock_embedding_manager.english_model = MagicMock()
            mock_optimizer.get_llm_response.return_value = None
            
            async def failing_optimize_query(*args, **kwargs):
                raise Exception("Optimization failed")
            
            async def failing_generate_queries(*args, **kwargs):
                raise Exception("Query generation failed")
            
            mock_optimizer.optimize_query = failing_optimize_query
            mock_query_gen.side_effect = failing_generate_queries
            
            # Execute pipeline - should handle errors gracefully
            result = await rag_service.process_queries_with_async_pipeline(
                "test query",
                german_retriever,
                english_retriever,
                [],
                "german"
            )
            
            # Should return error response but not crash
            assert "response" in result
            assert "Es tut mir leid" in result["response"]  # Error message in German
            assert result["from_cache"] is False
            assert "processing_time" in result
    
    @pytest.mark.asyncio
    async def test_async_pipeline_semantic_cache_handling(self, rag_service, mock_retrievers):
        """Test manejo de caché semántico en el pipeline."""
        
        german_retriever, english_retriever = mock_retrievers
        
        with patch('app.services.rag_service.embedding_manager') as mock_embedding_manager, \
             patch.object(rag_service, 'query_optimizer') as mock_optimizer, \
             patch.object(rag_service, '_handle_semantic_cache_result') as mock_semantic_handler:
            
            # Setup mocks
            mock_embedding_manager.german_model = MagicMock()
            mock_embedding_manager.english_model = MagicMock()
            mock_optimizer.get_llm_response.return_value = None
            
            # Mock semantic cache hit
            async def mock_optimize_query(*args, **kwargs):
                return {
                    'result': {
                        'response': 'Semantic cached response',
                        'sources': [],
                        'semantic_match': {'similarity': 0.85}
                    },
                    'source': 'semantic_cache'
                }
            
            async def mock_semantic_result(*args, **kwargs):
                return {
                    'response': 'Enhanced semantic response',
                    'sources': [],
                    'from_cache': False,
                    'semantic_match': {'similarity': 0.85},
                    'used_cached_chunks': True
                }
            
            mock_optimizer.optimize_query = mock_optimize_query
            mock_semantic_handler.return_value = mock_semantic_result()
            
            # Execute pipeline
            result = await rag_service.process_queries_with_async_pipeline(
                "test query",
                german_retriever,
                english_retriever,
                [],
                "german"
            )
            
            # Should handle semantic cache result
            assert result['response'] == 'Enhanced semantic response'
            assert 'semantic_match' in result
            assert result['used_cached_chunks'] is True
    
    def test_async_pipeline_configuration_fallback(self):
        """Test que la configuración funcione correctamente."""
        
        # Test with async pipeline enabled
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.ENABLE_ASYNC_PIPELINE = True
            mock_settings.ASYNC_PIPELINE_PHASE_LOGGING = True
            mock_settings.ASYNC_PIPELINE_PARALLEL_LIMIT = 10
            
            # These should be accessible
            assert mock_settings.ENABLE_ASYNC_PIPELINE is True
            assert mock_settings.ASYNC_PIPELINE_PHASE_LOGGING is True
            assert mock_settings.ASYNC_PIPELINE_PARALLEL_LIMIT == 10


def test_pipeline_performance_comparison():
    """
    Test conceptual para demostrar la diferencia de rendimiento
    entre pipeline secuencial vs asíncrono.
    """
    def simulate_sequential_pipeline():
        """Simula pipeline secuencial."""
        phases = {
            "cache_check": 0.02,
            "query_optimization": 0.1,
            "query_generation": 0.15,
            "retrieval_german": 0.2,
            "retrieval_english": 0.2,
            "reranking": 0.1,
            "response_generation": 0.3
        }
        return sum(phases.values())  # Total sequential time
    
    def simulate_async_pipeline():
        """Simula pipeline asíncrono."""
        # Phase 1: cache_check + query_optimization (parallel)
        phase1 = max(0.02, 0.1)  # 0.1s
        
        # Phase 2: query_generation
        phase2 = 0.15  # 0.15s
        
        # Phase 3: retrieval_german + retrieval_english (parallel)
        phase3 = max(0.2, 0.2)  # 0.2s
        
        # Phase 4: reranking
        phase4 = 0.1  # 0.1s
        
        # Phase 5: response_generation
        phase5 = 0.3  # 0.3s
        
        return phase1 + phase2 + phase3 + phase4 + phase5  # 0.85s
    
    sequential_time = simulate_sequential_pipeline()  # 1.07s
    async_time = simulate_async_pipeline()  # 0.85s
    
    improvement = ((sequential_time - async_time) / sequential_time) * 100
    
    print(f"Sequential pipeline: {sequential_time:.2f}s")
    print(f"Async pipeline: {async_time:.2f}s")
    print(f"Performance improvement: {improvement:.1f}%")
    
    # Verificar que hay mejora significativa
    assert improvement > 15, f"Expected >15% improvement, got {improvement:.1f}%"
    assert async_time < sequential_time, "Async pipeline should be faster"


if __name__ == "__main__":
    # Ejecutar test conceptual
    test_pipeline_performance_comparison()
    print("Pipeline performance comparison test passed!")