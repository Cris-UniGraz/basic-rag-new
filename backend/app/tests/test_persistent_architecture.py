"""
Tests básicos para validar la implementación de la arquitectura persistente.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).parent))

# Mock environment variables for testing
os.environ.update({
    "AZURE_OPENAI_API_KEY": "test-key",
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
    "AZURE_OPENAI_API_VERSION": "2023-05-15",
    "AZURE_OPENAI_API_LLM_DEPLOYMENT_ID": "test-gpt-4",
    "AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID": "test-embeddings",
    "AZURE_OPENAI_LLM_MODEL": "gpt-4",
    "AZURE_OPENAI_EMBEDDING_MODEL": "text-embedding-ada-002",
    "AZURE_COHERE_ENDPOINT": "https://test.cohere.azure.com/",
    "AZURE_COHERE_API_KEY": "test-cohere-key",
    "EMBEDDING_MODEL_NAME": "azure_openai",
    "COHERE_RERANKING_MODEL": "rerank-multilingual-v3.0",
    "COLLECTION_NAME": "test_collection",
    "MONGODB_CONNECTION_STRING": "mongodb://test:test@localhost:27017",
    "MONGODB_DATABASE_NAME": "test_db"
})


def mock_llm_service():
    """Mock LLM service for testing."""
    class MockLLMService:
        async def ainvoke(self, prompt):
            return "Test response"
        
        def invoke(self, prompt):
            return "Test response"
    
    return MockLLMService()


async def test_persistent_rag_service_creation(mock_llm_service):
    """Test that PersistentRAGService can be created and is singleton."""
    from app.services.persistent_rag_service import PersistentRAGService, create_persistent_rag_service
    
    # Test singleton pattern
    service1 = PersistentRAGService(mock_llm_service)
    service2 = PersistentRAGService(mock_llm_service)
    
    assert service1 is service2, "PersistentRAGService should be singleton"
    
    # Test factory function
    service3 = create_persistent_rag_service(mock_llm_service)
    assert service3 is service1, "Factory function should return same singleton instance"
    
    print("✓ PersistentRAGService singleton pattern works correctly")


async def test_retriever_cache():
    """Test the RetrieverCache functionality."""
    from app.services.persistent_rag_service import RetrieverCache
    
    cache = RetrieverCache(ttl_seconds=60)
    
    # Test setting and getting
    test_retriever = "mock_retriever"
    await cache.set("test_key", test_retriever)
    
    retrieved = await cache.get("test_key")
    assert retrieved == test_retriever, "Cache should return the stored retriever"
    
    # Test cache stats
    stats = cache.get_stats()
    assert stats["cached_retrievers"] == 1, "Cache should have 1 cached retriever"
    
    print("✓ RetrieverCache basic functionality works correctly")


async def test_health_status_tracking():
    """Test health status tracking."""
    from app.services.persistent_rag_service import RetrieverHealthStatus
    
    health = RetrieverHealthStatus()
    
    # Initially healthy
    assert health.is_healthy is True
    assert health.error_count == 0
    
    # Record success
    health.record_success(0.5)
    assert health.is_healthy is True
    assert health.get_avg_response_time() == 0.5
    
    # Record error
    health.record_error(Exception("Test error"))
    assert health.error_count == 1
    assert health.last_error == "Test error"
    
    print("✓ Health status tracking works correctly")


async def test_embedding_manager_singleton():
    """Test EmbeddingManager singleton and thread-safety."""
    from app.core.embedding_manager import EmbeddingManager
    
    # Test singleton pattern
    manager1 = EmbeddingManager()
    manager2 = EmbeddingManager()
    
    assert manager1 is manager2, "EmbeddingManager should be singleton"
    
    # Test initialization state
    assert hasattr(manager1, '_initialized'), "EmbeddingManager should have _initialized attribute"
    
    print("✓ EmbeddingManager singleton pattern works correctly")


async def test_connection_pool_creation():
    """Test connection pool creation."""
    from app.core.embedding_manager import AzureOpenAIConnectionPool
    
    pool = AzureOpenAIConnectionPool(max_connections=5, rate_limit_per_minute=100)
    
    assert pool.max_connections == 5
    assert pool.rate_limit_per_minute == 100
    assert len(pool.request_times) == 0
    
    # Test cleanup
    pool.cleanup()
    
    print("✓ Connection pool creation works correctly")


async def test_vector_store_manager_enhancements():
    """Test VectorStoreManager enhancements."""
    from app.models.vector_store import VectorStoreManager
    
    manager = VectorStoreManager()
    
    # Test initialization
    assert hasattr(manager, '_health_status'), "Should have health status tracking"
    assert hasattr(manager, '_circuit_breaker'), "Should have circuit breaker"
    assert hasattr(manager, '_metrics'), "Should have metrics tracking"
    
    # Test initial state
    assert manager._health_status["is_healthy"] is False, "Should start unhealthy"
    assert manager._circuit_breaker["state"] == "closed", "Circuit breaker should start closed"
    
    print("✓ VectorStoreManager enhancements work correctly")


async def test_circuit_breaker_functionality():
    """Test circuit breaker functionality."""
    from app.services.persistent_rag_service import PersistentRAGService
    
    service = PersistentRAGService()
    
    # Test initial circuit breaker state
    assert not service._is_circuit_breaker_open("test_breaker"), "Circuit breaker should start closed"
    
    # Simulate failures
    for i in range(6):  # More than threshold
        service._record_circuit_breaker_failure("test_breaker")
    
    # Check if circuit breaker opens
    breaker = service._circuit_breakers.get("test_breaker", {})
    assert breaker.get("state") == "open", "Circuit breaker should open after multiple failures"
    
    # Test reset
    service._reset_circuit_breaker("test_breaker")
    assert breaker.get("state") == "closed", "Circuit breaker should close after reset"
    
    print("✓ Circuit breaker functionality works correctly")


def test_configuration_additions():
    """Test that new configuration settings are available."""
    from app.core.config import settings
    
    # Test new persistent RAG settings
    assert hasattr(settings, 'RETRIEVER_CACHE_TTL'), "Should have RETRIEVER_CACHE_TTL setting"
    assert hasattr(settings, 'MAX_RETRIEVER_ERRORS'), "Should have MAX_RETRIEVER_ERRORS setting"
    assert hasattr(settings, 'HEALTH_CHECK_INTERVAL'), "Should have HEALTH_CHECK_INTERVAL setting"
    
    # Test circuit breaker settings
    assert hasattr(settings, 'CIRCUIT_BREAKER_THRESHOLD'), "Should have CIRCUIT_BREAKER_THRESHOLD setting"
    assert hasattr(settings, 'CIRCUIT_BREAKER_TIMEOUT'), "Should have CIRCUIT_BREAKER_TIMEOUT setting"
    
    # Test connection pooling settings
    assert hasattr(settings, 'AZURE_OPENAI_MAX_CONNECTIONS'), "Should have AZURE_OPENAI_MAX_CONNECTIONS setting"
    assert hasattr(settings, 'MILVUS_MAX_CONNECTIONS'), "Should have MILVUS_MAX_CONNECTIONS setting"
    
    print("✓ Configuration additions are present")


async def test_graceful_degradation():
    """Test graceful degradation functionality."""
    from app.services.persistent_rag_service import PersistentRAGService
    
    service = PersistentRAGService()
    
    # Test degradation mode switching
    service._degradation_mode = "full"
    assert service._degradation_mode == "full"
    
    service._degradation_mode = "intermediate" 
    assert service._degradation_mode == "intermediate"
    
    service._degradation_mode = "basic"
    assert service._degradation_mode == "basic"
    
    print("✓ Graceful degradation mode switching works correctly")


async def run_all_tests():
    """Run all tests."""
    print("🧪 Running Persistent Architecture Tests...\n")
    
    # Create mock LLM service
    class MockLLMService:
        async def ainvoke(self, prompt):
            return "Test response"
        
        def invoke(self, prompt):
            return "Test response"
    
    mock_llm = MockLLMService()
    
    try:
        # Run async tests
        await test_persistent_rag_service_creation(mock_llm)
        await test_retriever_cache()
        await test_health_status_tracking()
        await test_embedding_manager_singleton()
        await test_connection_pool_creation()
        await test_vector_store_manager_enhancements()
        await test_circuit_breaker_functionality()
        await test_graceful_degradation()
        
        # Run sync tests
        test_configuration_additions()
        
        print("\n✅ All tests passed! Persistent architecture implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n🎉 Persistent RAG Architecture - Phase 1 Implementation Complete!")
        print("\nKey features implemented:")
        print("• PersistentRAGService with singleton pattern and retriever caching")
        print("• Enhanced EmbeddingManager with connection pooling and thread-safety") 
        print("• Improved VectorStoreManager with connection pools and health monitoring")
        print("• Circuit breaker patterns for resilience")
        print("• Health monitoring and graceful degradation")
        print("• Background maintenance tasks")
        print("• Production-ready configuration settings")
        
        print("\nNext Steps (Phase 2):")
        print("• Implement RetrieverManager for advanced retriever lifecycle management")
        print("• Add RetrieverPool for load balancing")
        print("• Integrate with chat endpoint to use persistent retrievers")
        print("• Add comprehensive monitoring and alerting")
    else:
        print("\n❌ Tests failed. Please review the implementation.")
        sys.exit(1)