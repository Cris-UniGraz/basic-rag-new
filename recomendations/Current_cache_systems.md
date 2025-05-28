# Análisis de los Sistemas de Caché del Proyecto RAG

## Análisis de los Sistemas de Caché

Analizando el código del proyecto, puedo explicar por qué existen dos sistemas de caché diferentes:

### **1. ¿Por qué hay dos sistemas de caché "llm_cache" y "query_history"?**

Los dos sistemas de caché existen por **evolución del diseño del proyecto** y diferentes **propósitos funcionales**:

- **`query_history`** fue el **sistema original** diseñado para almacenar el historial completo de consultas y optimizaciones
- **`llm_cache`** es un **sistema más reciente y especializado** creado específicamente para cachear respuestas LLM completas con validación de calidad

### **2. ¿Cuál es el propósito de cada sistema de caché, y qué almacena cada uno?**

#### **A. `llm_cache` (Caché Especializado de Respuestas LLM)**
**Propósito**: Almacenar respuestas LLM completas y validadas de alta calidad

**Almacena**:
```python
{
    'response': response,                    # Respuesta completa del LLM
    'timestamp': datetime.now(),            # Cuándo se generó
    'language': language,                   # Idioma de la respuesta
    'sources': validated_sources,           # Fuentes con puntajes de reranking
    'original_query': query                 # Query original
}
```

**Criterios de almacenamiento**:
- Solo almacena respuestas **no vacías**
- Solo almacena si hay **fuentes con puntaje >= MIN_RERANKING_SCORE**
- **Validación de calidad**: Solo respuestas consideradas "buenas"

#### **B. `query_history` (Historial General de Consultas)**
**Propósito**: Mantener un historial completo de todas las consultas procesadas

**Almacena**:
```python
{
    'response': response,                    # Respuesta (puede estar vacía)
    'sources': sources,                     # Fuentes (sin validación estricta)
    'timestamp': datetime.now(),            # Cuándo se procesó
    'language': language,                   # Idioma
    'original_query': query,                # Query original
    'usage_count': 0                       # Contador de uso
}
```

**Criterios de almacenamiento**:
- Almacena **todas las consultas** procesadas
- **No valida calidad** de respuestas
- Incluye **contador de uso** para métricas

#### **C. `query_embeddings` (Caché de Embeddings)**
**Propósito**: Almacenar embeddings de consultas para búsqueda semántica

**Almacena**:
```python
{
    'query': query,                         # Query original
    'embedding': embedding,                 # Vector embedding
    'language': language,                   # Idioma
    'timestamp': datetime.now()             # Cuándo se creó
}
```

### **3. ¿Cuáles son sus diferencias?**

| Aspecto | `llm_cache` | `query_history` |
|---------|-------------|-----------------|
| **Función Principal** | Caché de respuestas de calidad | Historial completo de consultas |
| **Validación** | ✅ Estricta (puntajes, fuentes válidas) | ❌ Ninguna |
| **Respuestas Vacías** | ❌ Rechazadas | ✅ Almacenadas |
| **Criterio de Calidad** | Solo fuentes con score >= MIN_RERANKING_SCORE | Todas las fuentes |
| **Optimización** | Para **rendimiento** (respuestas rápidas) | Para **análisis** y métricas |
| **Limpieza Automática** | ✅ Sí (línea 132) | ❌ No |
| **Contador de Uso** | ❌ No | ✅ Sí |
| **TTL (Tiempo de Vida)** | Configurable por `ttl_hours` | 24 horas fijo |

### **Flujo de Almacenamiento**

1. **Al procesar una query**:
   ```python
   # Se almacena SIEMPRE en query_history
   self.query_history[query_hash] = cached_data
   
   # Se almacena en llm_cache SOLO si es de calidad
   if response and has_relevant_docs:
       self.llm_cache[query_hash] = validated_data
   ```

2. **Al buscar en caché**:
   ```python
   # Primero busca en llm_cache (respuestas de calidad)
   if query_hash in self.llm_cache:
       return cache_entry
   
   # Fallback: busca en query_history (todas las consultas)
   legacy_result = self._get_cached_result(query, language)
   ```

### **Conclusion**

La **duplicidad existe por diseño**: `llm_cache` optimiza el rendimiento con respuestas de calidad, mientras que `query_history` mantiene un registro histórico completo para análisis y métricas. Es una **arquitectura en capas** donde cada sistema tiene un propósito específico complementario.

## Ubicación en el Código

- **Archivo principal**: `/backend/app/core/query_optimizer.py`
- **Inicialización**: Líneas 46, 51 (llm_cache, query_history)
- **Almacenamiento llm_cache**: Método `_store_llm_response()` (línea 79)
- **Almacenamiento query_history**: Método `_store_query_result()` (línea 216)
- **Búsqueda**: Método `get_llm_response()` (línea 140)

## Configuración

Los sistemas de caché se controlan mediante las siguientes variables de entorno:

- `ADVANCED_CACHE_ENABLED`: Habilita/deshabilita llm_cache
- `ADVANCED_CACHE_MAX_SIZE`: Tamaño máximo del llm_cache
- `ADVANCED_CACHE_TTL_HOURS`: Tiempo de vida en horas
- `QUERY_HISTORY_SIZE`: Tamaño del query_history
- `SEMANTIC_CACHING_ENABLED`: Habilita búsqueda semántica