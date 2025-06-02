# Propuesta de migración a procesamiento unificado de documentos

## Resumen de cambios requeridos

**Objetivo**: Eliminar la clasificación por idioma y usar colecciones unificadas con embedding único para todos los documentos.

## Lista de tareas para implementar

### **FASE 1: Configuración y fundamentos** (Prioridad ALTA)

#### **Tarea 1.1: Actualizar configuración principal**
- **Archivo**: `backend/app/core/config.py`
- **Cambios**:
  - Eliminar `GERMAN_EMBEDDING_MODEL_NAME` y `ENGLISH_EMBEDDING_MODEL_NAME`
  - Usar solo `AZURE_OPENAI_EMBEDDING_MODEL` para todos los documentos
  - Eliminar `GERMAN_COHERE_RERANKING_MODEL` y `ENGLISH_COHERE_RERANKING_MODEL`
  - Simplificar `get_sources_path()` para no usar subdirectorios por idioma
  - Actualizar `DEFAULT_LANGUAGE` o eliminarlo

#### **Tarea 1.2: Refactorizar gestor de embeddings**
- **Archivo**: `backend/app/core/embedding_manager.py`
- **Cambios**:
  - Eliminar propiedades `german_model` y `english_model`
  - Usar solo una instancia del modelo Azure OpenAI
  - Simplificar método de selección de modelos
  - Actualizar inicialización para modelo único

#### **Tarea 1.3: Actualizar esquema de naming de colecciones**
- **Archivos**: `backend/app/models/vector_store.py`, `backend/app/models/document_store.py`
- **Cambios**:
  - Usar `COLLECTION_NAME` directamente para Milvus
  - Usar `{COLLECTION_NAME}_parent` para MongoDB
  - Eliminar lógica de sufijos `_de`/`_en`

### **FASE 2: API Endpoints** (Prioridad ALTA)

#### **Tarea 2.1: Simplificar endpoint de upload de documentos**
- **Archivo**: `backend/app/api/endpoints/documents.py`
- **Cambios**:
  - Eliminar parámetro `language` de endpoint `/upload`
  - Remover validación de idioma ("german"/"english")
  - Eliminar función `get_language_sufix()`
  - Usar directorio único para documentos (sin `/de/` o `/en/`)
  - Usar modelo de embedding único
  - Actualizar nomenclatura de colecciones

#### **Tarea 2.2: Simplificar endpoint de chat**
- **Archivo**: `backend/app/api/endpoints/chat.py`
- **Cambios**:
  - Eliminar parámetro `language` de endpoint `/chat`
  - Remover validación de idioma
  - Usar colección única en pipeline asíncrono
  - Actualizar selección de retriever

#### **Tarea 2.3: Actualizar endpoint de búsqueda**
- **Archivo**: `backend/app/api/endpoints/documents.py` (endpoint `/search`)
- **Cambios**:
  - Eliminar parámetro `language`
  - Usar colección unificada para búsquedas
  - Simplificar selección de embedding model

### **FASE 3: Servicios principales** (Prioridad ALTA)

#### **Tarea 3.1: Refactorizar RAG Service**
- **Archivo**: `backend/app/services/rag_service.py`
- **Cambios**:
  - Eliminar parámetro `language` de método `get_retriever()`
  - Usar colección única (`COLLECTION_NAME`)
  - Simplificar selección de embedding model en todos los métodos
  - Actualizar step-back query, multi-query, HyDE retrievers
  - Eliminar query translation entre idiomas
  - Usar reranker único

#### **Tarea 3.2: Simplificar optimizador de consultas**
- **Archivo**: `backend/app/core/query_optimizer.py`
- **Cambios**:
  - Eliminar consideraciones de idioma en caché
  - Usar claves de caché unificadas
  - Simplificar detección de términos de glossario

### **FASE 4: Frontend y utilidades** (Prioridad MEDIA)

#### **Tarea 4.1: Actualizar frontend**
- **Archivo**: `frontend/app.py`
- **Cambios**:
  - Eliminar selector de idioma de la interfaz
  - Remover parámetro `language` de llamadas API
  - Simplificar formularios de upload y búsqueda

#### **Tarea 4.2: Actualizar cargador de documentos**
- **Archivo**: `load-documents/load_documents.py`
- **Cambios**:
  - Eliminar estructura de directorios por idioma
  - Usar directorio único para todos los documentos
  - Remover parámetro de idioma del procesamiento

#### **Tarea 4.3: Simplificar glossario**
- **Archivo**: `backend/app/utils/glossary.py`
- **Cambios**:
  - Combinar glosarios alemán e inglés en uno unificado
  - Eliminar selección de glosario por idioma
  - Usar detección automática de términos multiidioma

### **FASE 5: Testing y validación** (Prioridad BAJA)

#### **Tarea 5.1: Actualizar tests**
- **Archivos**: `backend/app/tests/`, `backend/app/examples/`
- **Cambios**:
  - Eliminar parámetros de idioma en tests
  - Actualizar nombres de colecciones de prueba
  - Validar funcionamiento con colecciones unificadas

#### **Tarea 5.2: Actualizar ejemplos**
- **Archivos**: `backend/app/examples/`
- **Cambios**:
  - Simplificar ejemplos para usar procesamiento unificado
  - Actualizar scripts de benchmark

### **FASE 6: Migración de datos** (Crítica)

#### **Tarea 6.1: Script de migración de colecciones existentes**
- **Nuevo archivo**: `scripts/migrate_to_unified_collections.py`
- **Funcionalidad**:
  - Detectar colecciones existentes con sufijos `_de`/`_en`
  - Combinar documentos en colección unificada
  - Re-procesar embeddings con modelo Azure OpenAI único
  - Verificar integridad de datos migrados
  - Backup de colecciones originales

#### **Tarea 6.2: Actualizar variables de entorno**
- **Archivo**: `.env` de producción
- **Cambios**:
  - Actualizar `COLLECTION_NAME` a valor unificado
  - Verificar que `AZURE_OPENAI_EMBEDDING_MODEL` esté configurado
  - Eliminar variables de modelos por idioma

## Orden de implementación recomendado

1. **Fase 1** → **Fase 2** → **Fase 3** (núcleo del sistema)
2. **Fase 6.2** (variables de entorno)
3. **Fase 4** (frontend y utilidades)
4. **Fase 6.1** (migración de datos)
5. **Fase 5** (testing y validación)

## Consideraciones especiales

- **Backup obligatorio**: Respaldar todas las colecciones antes de la migración
- **Testing gradual**: Probar cada fase en entorno de desarrollo
- **Rollback plan**: Mantener capacidad de revertir cambios si es necesario
- **Performance**: Validar que el modelo único mantiene calidad de embeddings multiidioma

## Análisis detallado de archivos afectados

### **Archivos con mayor impacto:**

#### **1. Configuration (`backend/app/core/config.py`)**
- **Líneas 30-33**: Eliminar modelos de embedding por idioma
- **Líneas 44-45**: Eliminar modelos de reranking por idioma
- **Líneas 121-126**: Simplificar `get_sources_path()`

#### **2. API Endpoints (`backend/app/api/endpoints/documents.py`)**
- **Líneas 32, 50-51**: Eliminar validación de idioma
- **Líneas 54-57**: Remover lógica de sufijos de colección
- **Líneas 60**: Simplificar rutas de documentos
- **Líneas 416-421**: Eliminar función `get_language_sufix()`

#### **3. RAG Service (`backend/app/services/rag_service.py`)**
- **Líneas 143-151**: Simplificar método `get_retriever()`
- **Líneas 2069-2070, 2081-2137**: Actualizar nombres de colecciones
- **Líneas 1394-1397, 1596-1599**: Usar modelo único

#### **4. Embedding Manager (`backend/app/core/embedding_manager.py`)**
- **Líneas 61-74**: Refactorizar inicialización
- **Líneas 85-101**: Eliminar propiedades por idioma
- **Líneas 223-228, 261-266**: Simplificar selección de modelo

### **Archivos con impacto medio:**

#### **5. Chat Endpoint (`backend/app/api/endpoints/chat.py`)**
- **Líneas 28, 100-105**: Eliminar validación de idioma
- **Líneas 161-162**: Usar colección unificada

#### **6. Frontend (`frontend/app.py`)**
- **Líneas 147-148**: Eliminar estado de idioma
- **Líneas 844-852**: Remover selector de idioma

#### **7. Query Optimizer (`backend/app/core/query_optimizer.py`)**
- **Líneas 407, 487, 518**: Simplificar optimización
- **Líneas 952**: Actualizar detección de glosario

### **Archivos con impacto menor:**

#### **8. Document Loader (`load-documents/load_documents.py`)**
- **Líneas 36-37, 52-59**: Eliminar estructura por idioma

#### **9. Glossary (`backend/app/utils/glossary.py`)**
- **Líneas 13-78**: Combinar glosarios
- **Líneas 81-136**: Simplificar funciones

#### **10. Vector Store (`backend/app/models/vector_store.py`)**
- **Líneas 216-217, 570-593**: Actualizar eliminación de colecciones

## Beneficios esperados

1. **Simplificación arquitectural**: Eliminación de lógica compleja de idiomas
2. **Mantenimiento reducido**: Menos configuraciones y rutas de código
3. **Escalabilidad mejorada**: Soporte nativo para múltiples idiomas
4. **Consistencia**: Comportamiento uniforme independiente del idioma
5. **Performance**: Menos overhead en selección de modelos y rutas

## Riesgos y mitigaciones

### **Riesgos:**
- **Pérdida de datos**: Durante migración de colecciones
- **Degradación de calidad**: Modelo único vs especializados
- **Downtime**: Durante actualización del sistema

### **Mitigaciones:**
- **Backup completo**: Antes de cualquier cambio
- **Testing exhaustivo**: En entorno de desarrollo
- **Migración gradual**: Por fases con validación
- **Rollback automático**: En caso de errores críticos

## Métricas de éxito

1. **Funcionalidad**: Todos los endpoints funcionan sin parámetro de idioma
2. **Performance**: Tiempo de respuesta similar o mejor
3. **Calidad**: Resultados de búsqueda mantienen relevancia
4. **Estabilidad**: Sin errores en producción post-migración
5. **Cobertura**: Todos los tests pasan con nueva arquitectura