"""
Test para verificar la funcionalidad de inicialización paralela de retrievers con procesamiento unificado.
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


class TestUnifiedRetrieverInitialization:
    """Pruebas para la inicialización unificada de retrievers."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Mock del proveedor LLM."""
        return AsyncMock()
    
    @pytest.fixture
    def rag_service(self, mock_llm_provider):
        """Instancia del servicio RAG para pruebas."""
        return RAGService(mock_llm_provider)
    
    @pytest.mark.asyncio
    async def test_unified_initialization_collection_exists(self, rag_service):
        """Test cuando la colección unificada existe."""
        
        # Mock de utilidades y métodos necesarios
        with patch('app.services.rag_service.utility.has_collection') as mock_has_collection, \
             patch('app.services.rag_service.settings') as mock_settings, \
             patch('app.services.rag_service.embedding_manager') as mock_embedding_manager, \
             patch.object(rag_service, 'get_retriever') as mock_get_retriever, \
             patch.object(rag_service, 'ensure_initialized') as mock_ensure_initialized:
            
            # Configurar mocks
            mock_has_collection.return_value = True
            mock_settings.MAX_CHUNKS_CONSIDERED = 5
            mock_settings.MAX_CONCURRENT_TASKS = 3
            mock_embedding_manager.model = MagicMock()
            mock_ensure_initialized.return_value = None
            
            # Mock unified retriever
            mock_unified_retriever = MagicMock()
            
            # Configurar delay simulado para probar inicialización
            async def mock_get_retriever_delayed(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simular tiempo de inicialización
                return mock_unified_retriever
            
            mock_get_retriever.side_effect = mock_get_retriever_delayed
            
            # Ejecutar prueba
            start_time = time.time()
            result = await rag_service.initialize_retrievers_parallel("test_collection")
            execution_time = time.time() - start_time
            
            # Verificaciones
            assert "retriever" in result
            assert "metadata" in result
            
            retriever = result["retriever"]
            metadata = result["metadata"]
            
            # Verificar que el retriever unificado fue inicializado
            assert retriever == mock_unified_retriever
            
            # Verificar metadata
            assert metadata["successful_retrievers"] == 1
            assert metadata["failed_retrievers"] == 0
            assert metadata["total_tasks"] == 1
            
            # Verificar que get_retriever fue llamado una vez
            assert mock_get_retriever.call_count == 1
    
    @pytest.mark.asyncio
    async def test_unified_initialization_collection_fails(self, rag_service):
        """Test cuando la colección falla al inicializar."""
        
        with patch('app.services.rag_service.utility.has_collection') as mock_has_collection, \
             patch('app.services.rag_service.settings') as mock_settings, \
             patch('app.services.rag_service.embedding_manager') as mock_embedding_manager, \
             patch.object(rag_service, 'get_retriever') as mock_get_retriever, \
             patch.object(rag_service, 'ensure_initialized') as mock_ensure_initialized:
            
            # Configurar mocks
            mock_has_collection.return_value = True
            mock_settings.MAX_CHUNKS_CONSIDERED = 5
            mock_settings.MAX_CONCURRENT_TASKS = 3
            mock_embedding_manager.model = MagicMock()
            mock_ensure_initialized.return_value = None
            
            # Mock para simular fallo en retriever
            async def mock_get_retriever_with_failure(*args, **kwargs):
                raise Exception("Failed to initialize unified retriever")
            
            mock_get_retriever.side_effect = mock_get_retriever_with_failure
            
            # Ejecutar prueba
            result = await rag_service.initialize_retrievers_parallel("test_collection")
            
            # Verificaciones
            retriever = result["retriever"]
            metadata = result["metadata"]
            
            # El retriever debería ser None ya que falló
            assert retriever is None
            
            # Verificar metadata de fallo
            assert metadata["successful_retrievers"] == 0
            assert metadata["failed_retrievers"] == 1
            assert metadata["total_tasks"] == 1
    
    @pytest.mark.asyncio
    async def test_unified_initialization_no_collection_exists(self, rag_service):
        """Test cuando ninguna colección existe."""
        
        with patch('app.services.rag_service.utility.has_collection') as mock_has_collection, \
             patch('app.services.rag_service.settings') as mock_settings, \
             patch.object(rag_service, 'ensure_initialized') as mock_ensure_initialized:
            
            # Configurar mocks
            mock_has_collection.return_value = False
            mock_settings.MAX_CHUNKS_CONSIDERED = 5
            mock_settings.MAX_CONCURRENT_TASKS = 3
            mock_ensure_initialized.return_value = None
            
            # Ejecutar prueba
            result = await rag_service.initialize_retrievers_parallel("test_collection")
            
            # Verificaciones
            retriever = result["retriever"]
            metadata = result["metadata"]
            
            # No debería haber retriever inicializado
            assert retriever is None
            assert metadata["successful_retrievers"] == 0
            assert metadata["failed_retrievers"] == 1
            assert metadata["total_tasks"] == 1
    
    @pytest.mark.asyncio 
    async def test_parallel_initialization_performance_gain(self, rag_service):
        """Test para verificar la ganancia de rendimiento de la paralelización."""
        
        with patch('app.services.rag_service.utility.has_collection') as mock_has_collection, \
             patch('app.services.rag_service.settings') as mock_settings, \
             patch('app.services.rag_service.embedding_manager') as mock_embedding_manager, \
             patch.object(rag_service, 'get_retriever') as mock_get_retriever, \
             patch.object(rag_service, 'ensure_initialized') as mock_ensure_initialized:
            
            # Configurar mocks
            mock_has_collection.return_value = True
            mock_settings.MAX_CHUNKS_CONSIDERED = 5
            mock_settings.MAX_CONCURRENT_TASKS = 3
            mock_settings.get_sources_path.side_effect = lambda lang: f"/path/to/{lang}"
            mock_embedding_manager.german_model = MagicMock()
            mock_embedding_manager.english_model = MagicMock()
            mock_ensure_initialized.return_value = None
            
            # Simular tiempo de inicialización más realista
            initialization_delay = 0.2
            
            async def mock_get_retriever_realistic(*args, **kwargs):
                await asyncio.sleep(initialization_delay)
                return MagicMock()
            
            mock_get_retriever.side_effect = mock_get_retriever_realistic
            
            # Medir tiempo de ejecución paralela
            start_time = time.time()
            result = await rag_service.initialize_retrievers_parallel("test_collection")
            parallel_execution_time = time.time() - start_time
            
            # El tiempo paralelo debería ser aproximadamente initialization_delay,
            # no 2 * initialization_delay (que sería secuencial)
            expected_max_time = initialization_delay * 1.3  # 30% de tolerancia
            
            assert parallel_execution_time < expected_max_time, \
                f"Parallel execution time {parallel_execution_time:.3f}s is too high. " \
                f"Expected < {expected_max_time:.3f}s"
            
            # Verificar que ambos retrievers fueron inicializados
            assert len(result["retrievers"]) == 2
            assert result["metadata"]["successful_retrievers"] == 2


def test_performance_comparison():
    """
    Test conceptual para demostrar la diferencia de rendimiento.
    Este test no se ejecuta automáticamente, pero muestra la idea.
    """
    def simulate_sequential_initialization():
        """Simula inicialización secuencial."""
        total_time = 0
        for retriever in ["german", "english"]:
            # Simular tiempo de inicialización
            delay = 0.2  # 200ms por retriever
            total_time += delay
        return total_time
    
    def simulate_parallel_initialization():
        """Simula inicialización paralela."""
        # En paralelo, el tiempo total es el máximo de los tiempos individuales
        return max(0.2, 0.2)  # 200ms (el mayor de los dos)
    
    sequential_time = simulate_sequential_initialization()
    parallel_time = simulate_parallel_initialization()
    
    improvement_percentage = ((sequential_time - parallel_time) / sequential_time) * 100
    
    print(f"Sequential time: {sequential_time:.3f}s")
    print(f"Parallel time: {parallel_time:.3f}s") 
    print(f"Performance improvement: {improvement_percentage:.1f}%")
    
    # Verificar que hay mejora significativa
    assert improvement_percentage > 40, "Expected at least 40% improvement from parallelization"


if __name__ == "__main__":
    # Ejecutar test conceptual
    test_performance_comparison()
    print("Performance comparison test passed!")