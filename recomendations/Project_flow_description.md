# Project Flow Description - Basic RAG System

## Overview

This document describes the internal flow of the Basic RAG (Retrieval Augmented Generation) system, which implements advanced techniques for document retrieval and response generation. The system supports multi-language queries (German and English) and includes sophisticated caching mechanisms for performance optimization.

## System Architecture

The system is built as a containerized web application using FastAPI as the backend and Streamlit as the frontend. It uses:

- **Vector Database**: Milvus for storing document embeddings
- **Document Store**: MongoDB for storing parent documents
- **Cache Layer**: Redis for query and response caching
- **LLM Service**: Azure OpenAI GPT models
- **Reranking**: Cohere reranking models via Azure

## Main Query Processing Flow

### 1. Query Reception and Validation
- User submits a query through the frontend (Streamlit)
- The query is sent to the FastAPI backend via `/api/chat` endpoint
- System validates language parameter (German or English)
- Chat history is formatted from conversation messages

### 2. Cache Check Phase
The system implements a sophisticated two-level caching mechanism:

#### 2.1 Exact Cache Match
- System generates MD5 hash of the normalized query
- Checks Redis for exact query match
- If found and not expired (24 hours TTL), returns cached response immediately

#### 2.2 Semantic Cache Match (Enhanced Logic)
- If no exact match, system generates query embedding using appropriate language model
- Compares query embedding against stored query embeddings using cosine similarity
- If similarity score >= `QUERY_SIMILARITY_THRESHOLD` (default: 0.85), considers it a semantic match
- **New Behavior**: Instead of returning cached response directly:
  - If valid cached response and chunks exist: Uses cached chunks for new response generation
  - If no valid cached data: Continues with normal RAG processing

### 2.3 Semantic Cache Processing with Adaptive Chunk Reuse
When a semantic match is found with valid cached data:

#### 2.3.1 Cached Chunk Retrieval
- System retrieves stored document chunks from the semantically similar query
- Chunks include full content (`chunk_content`) and metadata (source, scores, etc.)
- **Initially skips document retrieval phase** - attempts to use cached chunks first

#### 2.3.2 Targeted Reranking with Validation
- Performs reranking using cached chunks against the NEW query
- Uses appropriate language-specific Cohere reranking model
- **Critical Validation**: Checks if any chunk achieves `reranking_score >= MIN_RERANKING_SCORE` (default: 0.2)

#### 2.3.3 Adaptive Processing Decision
**If chunks are relevant after reranking (≥ MIN_RERANKING_SCORE):**
- Continues with cached chunk strategy
- Filters context to include only sufficiently relevant chunks
- Generates fresh response using validated reranked chunks
- Stores new response in Redis following standard criteria

**If no chunks are relevant after reranking:**
- **Intelligent Fallback**: Abandons cached chunk strategy
- **Automatic Recovery**: Proceeds with complete RAG processing
- Executes full document retrieval using ensemble retrievers
- Performs standard reranking on fresh documents
- Generates response with newly retrieved, relevant content

#### 2.3.4 Quality Assurance and Performance Tracking
- System logs which strategy was employed and why
- Tracks cache effectiveness and fallback frequency
- Maintains metrics on chunk relevance post-reranking
- Ensures no compromise on response quality regardless of path taken

### 3. Query Generation and Enhancement
If no cache hit (exact or semantic with valid data), the system generates multiple query variations:

#### 3.1 Multi-Query Generation
Using a single LLM call, the system generates:
- Original query in source language
- Translated query to target language (German ↔ English)
- Step-back query in source language (more generic version)
- Step-back query in target language

#### 3.2 Glossary-Aware Processing
- System checks for specialized terms in University of Graz glossary
- Incorporates term definitions into query processing
- Ensures accurate translation of domain-specific terms

### 4. Document Retrieval Phase
The system uses an Ensemble Retriever that combines five different retrieval techniques:

#### 4.1 Base Vector Retriever
- Standard semantic search using document embeddings
- Weight: `RETRIEVER_WEIGHTS_BASE` (default: 0.1)

#### 4.2 Parent Document Retriever
- Retrieves larger parent documents for better context
- Uses hierarchical document structure
- Weight: `RETRIEVER_WEIGHTS_PARENT` (default: 0.3)

#### 4.3 Multi-Query Retriever
- Generates multiple query variations for broader document coverage
- Uses LLM to create alternative phrasings
- Weight: `RETRIEVER_WEIGHTS_MULTI_QUERY` (default: 0.4)

#### 4.4 HyDE Retriever (Hypothetical Document Embedder)
- Generates hypothetical answer document
- Embeds generated document to find similar real documents
- Weight: `RETRIEVER_WEIGHTS_HYDE` (default: 0.1)

#### 4.5 BM25 Retriever
- Keyword-based search for complementary results
- Traditional TF-IDF approach
- Weight: `RETRIEVER_WEIGHTS_BM25` (default: 0.1)

#### 4.6 Parallel Retrieval Execution
- All four queries (original DE/EN + step-back DE/EN) are processed in parallel
- Each query runs through all available retrievers
- Results are collected and deduplicated by content hash

### 5. Document Reranking Phase
#### 5.1 Unified Reranking
- All retrieved documents are combined into a single list
- Cohere reranking model scores each document against the original query
- Documents below `MIN_RERANKING_SCORE` (default: 0.2) are filtered out
- Documents are sorted by relevance score

#### 5.2 Final Document Selection
- Top `MAX_CHUNKS_LLM` (default: 6) documents are selected
- Source metadata is preserved for citation

### 6. Response Generation Phase
#### 6.1 Context Preparation
- Selected documents are formatted as context
- Glossary terms and definitions are included if relevant
- Language-specific prompt templates are applied

#### 6.2 LLM Response Generation
- Azure OpenAI GPT model generates response
- Context window includes retrieved documents and query
- Response is generated in requested language

#### 6.3 Response Validation
- System checks if valid response was generated
- If no relevant information found, returns appropriate message
- Validates minimum relevance threshold was met

### 7. Caching and Storage
#### 7.1 Enhanced Response Caching
- **Conditional Storage**: Only responses with relevant documents (reranking score >= `MIN_RERANKING_SCORE`) are cached
- **Enhanced Metadata**: Cached data includes:
  - Complete response text
  - Document chunks with full content (`chunk_content`)
  - Source metadata and reranking scores
  - Query embedding for semantic matching
- **TTL Management**: 24-hour expiration with automatic cleanup
- **Validation**: Empty responses or responses without sources are not cached

#### 7.2 Intelligent Cache Strategy
- **Exact Match**: Direct response return for identical queries
- **Semantic Match**: Chunk reuse with fresh response generation
- **Cache Warming**: Proactive storage of high-quality responses
- **Memory Efficiency**: LRU eviction and size limits

#### 7.3 Metrics Collection
- Processing time, cache hit rates, and document scores are logged
- **New Metrics**: Tracks cached chunk usage and semantic match effectiveness
- Query patterns and retriever effectiveness are tracked
- Error rates and API usage are monitored
- Performance comparison between cached chunk reuse vs. full retrieval

## Cache System Details

### Redis Cache Structure
The system uses multiple Redis data structures:

1. **Query-Response Cache**: Stores complete responses keyed by query hash
2. **Query Embeddings**: Stores embeddings for semantic similarity matching
3. **Enhanced Document Metadata**: Stores source information, reranking scores, and full chunk content
4. **Semantic Match Index**: Maintains mapping between similar queries for fast lookup

### Cache Optimization Features
- **Dynamic Similarity Threshold**: Configurable threshold (default: 0.85) for semantic matching
- **Intelligent TTL Management**: 24-hour expiration with automatic cleanup
- **Smart Memory Management**: LRU eviction and size limits with quality-based retention
- **Adaptive Cache Warming**: Proactive caching based on query patterns and success rates
- **Chunk Content Preservation**: Full document content stored for efficient reuse

## Performance Optimizations

### 1. Parallel Processing
- Multiple queries processed simultaneously using asyncio
- Concurrent retriever execution
- Background cache cleanup

### 2. Advanced Intelligent Caching with Adaptive Processing
- **Multi-tier caching strategy**:
  - **Exact match**: Immediate response return
  - **Semantic match with relevant chunks**: Skip retrieval, use reranked cached chunks
  - **Semantic match with irrelevant chunks**: Automatic fallback to full processing
  - **No match**: Complete RAG pipeline execution
- **Embedding caching**: Avoids recomputation of query embeddings
- **Chunk content preservation**: Stores full document content for reuse
- **Quality-based storage**: Only caches responses with relevant sources
- **Adaptive reranking with validation**: Optimizes cached chunks and validates relevance
- **Intelligent fallback mechanism**: Automatic recovery when cached chunks are insufficient

### 3. Resource Management
- Connection pooling for databases
- Coroutine management for async operations
- Memory-efficient document processing

### 4. Quality Filtering
- Reranking score thresholds
- Document relevance validation
- Response quality checks

## Error Handling and Monitoring

### Error Recovery
- Graceful degradation when retrievers fail
- Fallback mechanisms for cache misses
- Retry logic for API calls

### Monitoring and Metrics
- Comprehensive metrics collection via MetricsManager
- Performance tracking across all components
- Error rate monitoring and alerting
- Cache efficiency analytics

## Configuration Parameters

Key system parameters that control behavior:

- `QUERY_SIMILARITY_THRESHOLD`: Minimum similarity for semantic cache hits (default: 0.85)
- `MIN_RERANKING_SCORE`: Minimum score for document inclusion and cache storage (default: 0.2)
- `MAX_CHUNKS_LLM`: Maximum documents sent to LLM (default: 6)
- `RETRIEVER_WEIGHTS_*`: Ensemble retriever weight distribution
- `ADVANCED_CACHE_TTL_HOURS`: Cache entry time-to-live (default: 24 hours)
- `ADVANCED_CACHE_ENABLED`: Enable/disable advanced caching features (default: True)
- `SEMANTIC_CACHING_ENABLED`: Enable/disable semantic similarity matching (default: True)

## Security and Data Privacy

- No persistent storage of user queries beyond cache TTL
- Secure API key management for external services
- Input validation and sanitization
- Rate limiting and abuse prevention

## Summary of Enhanced Performance Features

The updated system introduces significant performance improvements while maintaining response quality:

### Key Enhancements

1. **Smart Cache Storage**: Only high-quality responses with relevant sources are cached
2. **Adaptive Semantic Processing**: Similar queries trigger intelligent chunk evaluation
3. **Conditional Retrieval Bypass**: Document retrieval is skipped only when cached chunks prove relevant
4. **Validated Reranking**: Cached chunks are reranked and validated for relevance before use
5. **Intelligent Fallback**: Automatic recovery to full processing when cached chunks are insufficient
6. **Quality Assurance**: Multiple validation layers ensure optimal response quality
7. **Performance Optimization**: Maximizes efficiency while maintaining quality standards

### Performance Benefits

- **Adaptive Efficiency**: Reduces API calls and processing time when cached chunks are relevant
- **Quality-First Approach**: Prioritizes response quality over speed, falling back when necessary
- **Intelligent Resource Usage**: Optimizes computational resources based on chunk relevance validation
- **Maintained Standards**: Ensures consistently high response quality through validation gates
- **Scalable Architecture**: System adapts to query patterns and scales efficiently
- **Robust Performance**: Fallback mechanisms ensure reliable performance under all conditions

### Operational Excellence

- **Multi-Layer Validation**: Relevance validation at multiple stages ensures quality
- **Intelligent Error Recovery**: Automatic fallback when cached chunks prove insufficient
- **Comprehensive Monitoring**: Detailed metrics track cache effectiveness, fallback rates, and chunk relevance
- **Adaptive Decision Making**: System automatically chooses optimal processing strategy
- **Configurable Thresholds**: Tunable parameters for similarity and relevance scoring
- **Memory Efficiency**: Smart cache management with quality-based retention
- **Performance Transparency**: Clear logging indicates processing path and decision rationale

## Processing Flow Decision Tree

The system follows this decision logic for query processing:

```
Query Received
    ↓
Exact Cache Match?
    ├─ YES → Return Cached Response
    ↓
    NO → Generate Query Embedding
    ↓
Semantic Match Found (≥ 0.85 similarity)?
    ├─ NO → Execute Full RAG Pipeline
    ↓
    YES → Retrieve Cached Chunks
    ↓
Rerank Cached Chunks vs New Query
    ↓
Any Chunk Score ≥ MIN_RERANKING_SCORE?
    ├─ NO → Execute Full RAG Pipeline (Fallback)
    ↓
    YES → Generate Response with Relevant Chunks
    ↓
Store New Response in Cache
```

This comprehensive flow ensures high-quality, contextually relevant responses while achieving superior performance through intelligent caching, adaptive chunk reuse, and quality-validated processing strategies. The system maintains optimal balance between efficiency and quality by making intelligent decisions at each stage based on actual content relevance rather than assumptions.