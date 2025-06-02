# Variables de entorno para procesamiento unificado de documentos

## 🔧 **VARIABLES REQUERIDAS PARA LA MIGRACIÓN**

### **1. Configuración de embeddings unificada**
```bash
# ELIMINADAS - Ya no necesarias
# GERMAN_EMBEDDING_MODEL_NAME=
# ENGLISH_EMBEDDING_MODEL_NAME=

# CONSERVAR - Modelo unificado
EMBEDDING_MODEL_NAME=azure_openai
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

### **2. Configuración de reranking unificada**
```bash
# ELIMINADAS - Ya no necesarias
# GERMAN_COHERE_RERANKING_MODEL=
# ENGLISH_COHERE_RERANKING_MODEL=

# NUEVA - Modelo unificado
COHERE_RERANKING_MODEL=rerank-multilingual-v3.0
```

### **3. Configuración de colecciones**
```bash
# CONSERVAR - Colección unificada
COLLECTION_NAME=uni_docs_unified

# ELIMINAR - Ya no necesarias para idiomas específicos
# No usar sufijos _de o _en en COLLECTION_NAME
```

### **4. Variables de Azure OpenAI (CONSERVAR)**
```bash
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_API_LLM_DEPLOYMENT_ID=gpt-4
AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID=text-embedding-ada-002
AZURE_OPENAI_LLM_MODEL=gpt-4
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

### **5. Variables de Cohere (CONSERVAR pero simplificar)**
```bash
AZURE_COHERE_ENDPOINT=https://your-cohere-endpoint
AZURE_COHERE_API_KEY=your_cohere_key
# NUEVO - Un solo modelo para todos los idiomas
COHERE_RERANKING_MODEL=rerank-multilingual-v3.0
```

### **6. Configuración de directorios simplificada**
```bash
# CONSERVAR
SOURCES_PATH=/app/data/documents
# Nota: Ya no se usan subdirectorios /de y /en
```

### **7. Variables eliminadas de idioma por defecto**
```bash
# ELIMINAR - Ya no necesaria
# DEFAULT_LANGUAGE=german
```

## 📋 **CHECKLIST DE MIGRACIÓN DE VARIABLES**

### ✅ **Variables a eliminar completamente:**
- [ ] `GERMAN_EMBEDDING_MODEL_NAME`
- [ ] `ENGLISH_EMBEDDING_MODEL_NAME` 
- [ ] `GERMAN_COHERE_RERANKING_MODEL`
- [ ] `ENGLISH_COHERE_RERANKING_MODEL`
- [ ] `DEFAULT_LANGUAGE`

### ✅ **Variables a agregar:**
- [ ] `COHERE_RERANKING_MODEL=rerank-multilingual-v3.0`

### ✅ **Variables a conservar sin cambios:**
- [ ] `EMBEDDING_MODEL_NAME`
- [ ] `AZURE_OPENAI_*` (todas las variables de Azure OpenAI)
- [ ] `AZURE_COHERE_ENDPOINT`
- [ ] `AZURE_COHERE_API_KEY`
- [ ] `COLLECTION_NAME` (sin sufijos de idioma)
- [ ] `SOURCES_PATH`
- [ ] Variables de configuración de chunks y retrieval

## 🎯 **EJEMPLO DE .env ACTUALIZADO**

```bash
# =============================================================================
# CONFIGURACIÓN BÁSICA
# =============================================================================
API_HOST=0.0.0.0
API_PORT=8000
PROJECT_NAME=RAG API
VERSION=0.1.0

# =============================================================================
# AZURE OPENAI - CONFIGURACIÓN UNIFICADA
# =============================================================================
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_API_LLM_DEPLOYMENT_ID=gpt-4
AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID=text-embedding-ada-002
AZURE_OPENAI_LLM_MODEL=gpt-4
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# =============================================================================
# EMBEDDINGS - MODELO UNIFICADO
# =============================================================================
EMBEDDING_MODEL_NAME=azure_openai

# =============================================================================
# RERANKING - MODELO UNIFICADO MULTIIDIOMA
# =============================================================================
AZURE_COHERE_ENDPOINT=https://your-cohere-endpoint
AZURE_COHERE_API_KEY=your_cohere_key
COHERE_RERANKING_MODEL=rerank-multilingual-v3.0
RERANKING_TYPE=cohere
MIN_RERANKING_SCORE=0.2

# =============================================================================
# COLECCIONES Y DOCUMENTOS - PROCESAMIENTO UNIFICADO
# =============================================================================
COLLECTION_NAME=uni_docs_unified
SOURCES_PATH=/app/data/documents
MAX_CHUNKS_CONSIDERED=10
MAX_CHUNKS_LLM=6

# =============================================================================
# MONGODB
# =============================================================================
MONGODB_CONNECTION_STRING=mongodb://mongodb:27017/
MONGODB_DATABASE_NAME=rag_db

# =============================================================================
# CONFIGURACIÓN DE CHUNKS
# =============================================================================
CHUNK_SIZE=512
CHUNK_OVERLAP=16
PARENT_CHUNK_SIZE=4096
PARENT_CHUNK_OVERLAP=32
PAGE_OVERLAP=16

# =============================================================================
# CONFIGURACIÓN DE RETRIEVERS
# =============================================================================
RETRIEVER_WEIGHTS_BASE=0.1
RETRIEVER_WEIGHTS_PARENT=0.3
RETRIEVER_WEIGHTS_MULTI_QUERY=0.4
RETRIEVER_WEIGHTS_HYDE=0.1
RETRIEVER_WEIGHTS_BM25=0.1

# =============================================================================
# CACHE Y OPTIMIZACIÓN
# =============================================================================
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600
ENABLE_CACHE=true
ADVANCED_CACHE_ENABLED=true
ADVANCED_CACHE_MAX_SIZE=1000
ADVANCED_CACHE_TTL_HOURS=24
ADVANCED_CACHE_SIMILARITY_THRESHOLD=0.85

# =============================================================================
# CONFIGURACIÓN DEL SISTEMA
# =============================================================================
LOG_LEVEL=INFO
SHOW_INTERNAL_MESSAGES=false
USER_AGENT=rag_assistant
MAX_CONCURRENT_TASKS=5
TASK_TIMEOUT=60
MAX_RETRIES=3

# =============================================================================
# CONFIGURACIÓN DE VECTOR STORE
# =============================================================================
DONT_KEEP_COLLECTIONS=false

# =============================================================================
# TIMEOUTS
# =============================================================================
CHAT_REQUEST_TIMEOUT=180
RETRIEVAL_TASK_TIMEOUT=90
LLM_GENERATION_TIMEOUT=120
```

## ⚠️ **VALIDACIÓN POST-MIGRACIÓN**

Después de actualizar las variables de entorno, verificar:

1. **Startup del backend**: El sistema inicia sin errores
2. **Upload de documentos**: Funciona sin parámetro `language`
3. **Chat**: Funciona sin selección de idioma
4. **Búsqueda**: Funciona en colección unificada
5. **Embeddings**: Se usa modelo Azure OpenAI único
6. **Reranking**: Se usa modelo Cohere multiidioma

## 🔄 **MIGRACIÓN SEGURA**

1. **Backup**: Respaldar `.env` actual antes de cambios
2. **Testing**: Probar en entorno de desarrollo primero
3. **Gradual**: Aplicar cambios uno por uno
4. **Validación**: Verificar cada componente después del cambio
5. **Rollback**: Tener plan de reverso en caso de problemas

## 📊 **BENEFICIOS ESPERADOS**

- ✅ Configuración simplificada (50% menos variables)
- ✅ Mantenimiento reducido
- ✅ Comportamiento consistente multiidioma
- ✅ Escalabilidad mejorada
- ✅ Menos puntos de falla