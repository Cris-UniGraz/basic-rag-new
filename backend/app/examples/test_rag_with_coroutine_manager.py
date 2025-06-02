"""
Real-world test for RAG integration with the enhanced CoroutineManager.

This script demonstrates how the enhanced CoroutineManager improves the performance
and reliability of the RAG system in a realistic scenario with:
- Multiple parallel retrievals across languages
- Query optimization and translation
- Chunked execution for efficient resource usage
- Advanced timeout and error handling
- Performance metrics
"""
import asyncio
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

# Add the project root to the Python path to make imports work
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.coroutine_manager import coroutine_manager
from app.core.config import settings
from app.services.rag_service import RAGService
from app.core.metrics_manager import MetricsManager
from app.core.query_optimizer import QueryOptimizer
from loguru import logger


# Configure logger to display on console
logger.remove()
logger.add(sys.stdout, level="INFO")


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    async def ainvoke(self, messages):
        """Simulate LLM response."""
        # Extract the query from messages
        query = None
        for message in messages:
            if isinstance(message, dict) and message.get("type") == "human":
                query = message.get("content", "")
                break
            elif hasattr(message, "content") and hasattr(message, "type"):
                if message.type == "human":
                    query = message.content
                    break
        
        # Simulate processing time based on message length
        processing_time = 0.1 + (len(str(messages)) * 0.001)
        await asyncio.sleep(processing_time)
        
        # Return a mock response
        return f"This is a simulated response to: {query}"


class MockEmbeddingModel:
    """Mock embedding model for testing."""
    
    def embed_documents(self, texts):
        """Simulate document embedding."""
        return [[random.random() for _ in range(384)] for _ in texts]
    
    def embed_query(self, text):
        """Simulate query embedding."""
        return [random.random() for _ in range(384)]


class MockRetriever:
    """Mock document retriever for testing."""
    
    def __init__(self, name: str, delay_range: tuple = (0.1, 0.5), error_rate: float = 0.1):
        self.name = name
        self.delay_range = delay_range
        self.error_rate = error_rate
    
    async def ainvoke(self, query_dict):
        """Simulate document retrieval with realistic timing and occasional errors."""
        from langchain_core.documents import Document
        
        # Extract query from the input dictionary
        query = query_dict.get("input", "")
        if not query:
            return []
        
        # Simulate processing delay
        delay = random.uniform(*self.delay_range)
        await asyncio.sleep(delay)
        
        # Simulate occasional errors
        if random.random() < self.error_rate:
            raise RuntimeError(f"Simulated error in {self.name} retriever")
        
        # Generate mock documents with metadata
        num_docs = random.randint(3, 8)
        docs = []
        
        for i in range(num_docs):
            doc = Document(
                page_content=f"This is document {i} from {self.name} retriever for query: {query}",
                metadata={
                    "source": f"mock_source_{i}",
                    "page_number": i,
                    "file_type": "txt",
                    "retriever": self.name
                }
            )
            docs.append(doc)
        
        return docs


async def test_parallel_retrieval():
    """
    Test the parallel retrieval capabilities of the RAG service with enhanced CoroutineManager.
    
    This test demonstrates how multiple retrieval operations are performed in parallel
    with chunking, error handling, and timeout management.
    """
    logger.info("=== Testing Parallel Retrieval with Enhanced CoroutineManager ===")
    
    # Initialize mock components
    llm_provider = MockLLMProvider()
    german_embedding = MockEmbeddingModel()
    english_embedding = MockEmbeddingModel()
    
    # Create a RAG service with the mock components
    rag_service = RAGService(llm_provider)
    rag_service.query_optimizer = QueryOptimizer()  # Use real optimizer
    rag_service.metrics_manager = MetricsManager()  # Use real metrics manager
    
    # Create mock retrievers with different characteristics
    retriever_de = MockRetriever("german", delay_range=(0.2, 0.6), error_rate=0.1)
    retriever_en = MockRetriever("english", delay_range=(0.3, 0.7), error_rate=0.15)
    
    # Test queries in German
    test_queries_de = [
        "Wie funktioniert die Einschreibung an der Universität?",
        "Wann ist die Bibliothek geöffnet?",
        "Welche Studiengänge gibt es an der Fakultät für Informatik?",
        "Wie kann ich einen Studienausweis beantragen?",
        "Wo finde ich Informationen zu Stipendien?",
    ]
    
    # Run test for each query
    all_results = []
    total_time_start = time.time()
    
    for query in test_queries_de:
        logger.info(f"Processing query: '{query}'")
        
        start_time = time.time()
        result = await rag_service.process_queries_with_async_pipeline(
            query=query,
            retriever_de=retriever_de,
            retriever_en=retriever_en,
            language="german"
        )
        
        duration = time.time() - start_time
        
        # Log result summary
        sources_count = len(result.get('sources', []))
        is_cached = result.get('from_cache', False)
        logger.info(f"Query processing completed in {duration:.2f}s")
        logger.info(f"Found {sources_count} sources, cached: {is_cached}")
        
        # Add to results
        all_results.append({
            'query': query,
            'duration': duration,
            'sources_count': sources_count,
            'is_cached': is_cached
        })
    
    # Log overall statistics
    total_time = time.time() - total_time_start
    cached_count = sum(1 for r in all_results if r['is_cached'])
    avg_duration = sum(r['duration'] for r in all_results) / len(all_results)
    
    logger.info("=== Parallel Retrieval Test Summary ===")
    logger.info(f"Total time: {total_time:.2f}s for {len(test_queries_de)} queries")
    logger.info(f"Average query time: {avg_duration:.2f}s")
    logger.info(f"Cache hit rate: {cached_count}/{len(test_queries_de)}")
    
    # Check coroutine manager metrics
    task_count = coroutine_manager.active_task_count
    logger.info(f"Active tasks remaining: {task_count}")
    
    # Clean up
    cleanup_result = await coroutine_manager.cleanup()
    logger.info(f"Cleanup result: {cleanup_result['status']}")
    logger.info(f"Cleaned up {cleanup_result['cleaned_tasks']} tasks")


async def test_heavy_load_retrieval():
    """
    Test the system under heavy load to demonstrate chunking and resource management.
    
    This test simulates multiple concurrent users making queries to the RAG system,
    demonstrating how the enhanced CoroutineManager prevents resource exhaustion.
    """
    logger.info("=== Testing Heavy Load Retrieval ===")
    
    # Initialize components
    llm_provider = MockLLMProvider()
    
    # Create a RAG service
    rag_service = RAGService(llm_provider)
    rag_service.query_optimizer = QueryOptimizer()
    rag_service.metrics_manager = MetricsManager()
    
    # Create retrievers with different performance characteristics
    retrievers = {
        "german_fast": MockRetriever("german_fast", delay_range=(0.1, 0.3), error_rate=0.05),
        "german_slow": MockRetriever("german_slow", delay_range=(0.5, 1.2), error_rate=0.1),
        "english_fast": MockRetriever("english_fast", delay_range=(0.1, 0.3), error_rate=0.05),
        "english_slow": MockRetriever("english_slow", delay_range=(0.5, 1.2), error_rate=0.15),
    }
    
    # List of sample queries
    queries = [
        "Wie funktioniert die Einschreibung?",
        "Wann sind die Öffnungszeiten?",
        "Wo finde ich den Hörsaal B?",
        "Wie viele ECTS brauche ich?",
        "Wer ist für die Studienberatung zuständig?",
        "Wie beantrage ich ein Urlaubssemester?",
        "Welche Fristen gibt es für die Prüfungsanmeldung?",
        "Wo kann ich Bücher ausleihen?",
        "Wie komme ich an einen Bibliotheksausweis?",
        "Welche Sprachkurse werden angeboten?",
    ]
    
    # Test parameters
    num_concurrent_users = 10
    queries_per_user = 3
    
    # Generate user sessions with randomized queries
    user_sessions = []
    for i in range(num_concurrent_users):
        user_queries = random.sample(queries, queries_per_user)
        retriever_pair = random.choice([
            ("german_fast", "english_fast"),
            ("german_fast", "english_slow"),
            ("german_slow", "english_fast"),
            ("german_slow", "english_slow"),
        ])
        
        user_sessions.append({
            "user_id": f"user_{i}",
            "queries": user_queries,
            "retriever_de": retrievers[retriever_pair[0]],
            "retriever_en": retrievers[retriever_pair[1]]
        })
    
    # Define function to process a single user session
    async def process_user_session(session):
        user_id = session["user_id"]
        results = []
        
        for query in session["queries"]:
            try:
                logger.info(f"User {user_id} querying: '{query}'")
                
                start_time = time.time()
                result = await rag_service.process_queries_with_async_pipeline(
                    query=query,
                    retriever_de=session["retriever_de"],
                    retriever_en=session["retriever_en"],
                    language="german"
                )
                
                duration = time.time() - start_time
                
                results.append({
                    "query": query,
                    "duration": duration,
                    "sources_count": len(result.get("sources", [])),
                    "from_cache": result.get("from_cache", False)
                })
                
                logger.info(f"User {user_id} query completed in {duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Error processing query for user {user_id}: {e}")
                results.append({
                    "query": query,
                    "error": str(e)
                })
        
        return {
            "user_id": user_id,
            "results": results
        }
    
    # Process all user sessions in parallel
    logger.info(f"Starting load test with {num_concurrent_users} concurrent users, {queries_per_user} queries each")
    start_time = time.time()
    
    # Use gather_coroutines to process all sessions with chunking
    tasks = [process_user_session(session) for session in user_sessions]
    all_session_results = await coroutine_manager.gather_coroutines(
        *tasks,
        chunk_size=5,  # Process in chunks of 5 users
        task_prefix="user-session",
        task_type="load-test",
        return_exceptions=True
    )
    
    total_time = time.time() - start_time
    
    # Analyze results
    valid_results = [r for r in all_session_results if isinstance(r, dict)]
    error_count = sum(1 for r in all_session_results if isinstance(r, Exception))
    
    total_queries = 0
    successful_queries = 0
    cached_queries = 0
    query_times = []
    
    for session_result in valid_results:
        for query_result in session_result.get("results", []):
            total_queries += 1
            if "error" not in query_result:
                successful_queries += 1
                query_times.append(query_result.get("duration", 0))
                if query_result.get("from_cache", False):
                    cached_queries += 1
    
    # Print summary
    logger.info("=== Heavy Load Test Summary ===")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Users: {len(valid_results)}/{num_concurrent_users} successful")
    logger.info(f"Queries: {successful_queries}/{total_queries} successful, {cached_queries} from cache")
    
    if query_times:
        avg_query_time = sum(query_times) / len(query_times)
        logger.info(f"Average query time: {avg_query_time:.2f}s")
    
    # Get coroutine manager metrics
    logger.info(f"Active tasks remaining: {coroutine_manager.active_task_count}")
    
    # Clean up
    cleanup_result = await coroutine_manager.cleanup()
    logger.info(f"Cleanup result: {cleanup_result['status']}")
    logger.info(f"Cleaned up {cleanup_result['cleaned_tasks']} tasks")


async def test_error_recovery():
    """
    Test error recovery capabilities of the RAG service with enhanced CoroutineManager.
    
    This test simulates various error scenarios including:
    - Retrievers that fail completely
    - Timeouts during document retrieval
    - Partial failures in multi-stage processing
    
    It demonstrates how the enhanced CoroutineManager handles these errors gracefully.
    """
    logger.info("=== Testing Error Recovery ===")
    
    # Initialize components
    llm_provider = MockLLMProvider()
    
    # Create a RAG service
    rag_service = RAGService(llm_provider)
    rag_service.query_optimizer = QueryOptimizer()
    rag_service.metrics_manager = MetricsManager()
    
    # Create special error-prone retrievers
    error_retriever = MockRetriever("error_prone", delay_range=(0.1, 0.3), error_rate=1.0)  # Always fails
    timeout_retriever = MockRetriever("timeout_prone", delay_range=(1.5, 3.0), error_rate=0.1)  # Often times out
    normal_retriever = MockRetriever("normal", delay_range=(0.1, 0.3), error_rate=0.1)  # Mostly succeeds
    
    # Test different combinations
    test_cases = [
        {
            "name": "Both retrievers fail",
            "retriever_de": error_retriever,
            "retriever_en": error_retriever,
            "query": "Wie funktioniert die Einschreibung?",
            "timeout": 1.0
        },
        {
            "name": "German retriever times out",
            "retriever_de": timeout_retriever,
            "retriever_en": normal_retriever,
            "query": "Was sind die Öffnungszeiten der Bibliothek?",
            "timeout": 1.0
        },
        {
            "name": "English retriever fails but German works",
            "retriever_de": normal_retriever,
            "retriever_en": error_retriever,
            "query": "Wo finde ich Informationen zu Stipendien?",
            "timeout": 2.0
        },
        {
            "name": "No timeout, should succeed",
            "retriever_de": normal_retriever,
            "retriever_en": normal_retriever,
            "query": "Welche Kurse gibt es im Sommersemester?",
            "timeout": None
        }
    ]
    
    # Run each test case
    for case in test_cases:
        logger.info(f"Running test case: {case['name']}")
        
        try:
            # Set a custom timeout for this case if specified
            if case['timeout']:
                # Store original settings
                original_timeout = settings.TASK_TIMEOUT
                # Update settings for this test
                settings.TASK_TIMEOUT = case['timeout']
                
                logger.info(f"Using custom timeout: {settings.TASK_TIMEOUT}s")
            
            # Process query with the specified retrievers
            start_time = time.time()
            result = await rag_service.process_queries_with_async_pipeline(
                query=case['query'],
                retriever_de=case['retriever_de'],
                retriever_en=case['retriever_en'],
                language="german"
            )
            
            duration = time.time() - start_time
            
            # Check result
            has_response = bool(result.get('response', ''))
            sources_count = len(result.get('sources', []))
            
            logger.info(f"Test case completed in {duration:.2f}s")
            logger.info(f"Result: has_response={has_response}, sources_count={sources_count}")
            
        except Exception as e:
            logger.error(f"Test case threw an exception: {e}")
        
        finally:
            # Restore original settings if we modified them
            if case.get('timeout'):
                settings.TASK_TIMEOUT = original_timeout
            
            # Clean up after each test case
            await coroutine_manager.cleanup()
            logger.info("-" * 50)
    
    # Check overall metrics
    errors = rag_service.metrics_manager.metrics.get('errors', {})
    logger.info("=== Error Recovery Test Summary ===")
    logger.info(f"Error counts: {errors}")


async def main():
    """Run all tests in sequence."""
    logger.info("Starting RAG with CoroutineManager Tests")
    
    # Run tests
    await test_parallel_retrieval()
    await asyncio.sleep(1)  # Pause between tests
    
    await test_heavy_load_retrieval()
    await asyncio.sleep(1)
    
    await test_error_recovery()
    
    logger.info("All tests completed")
    
    # Final cleanup
    await coroutine_manager.cleanup(force=True)


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())