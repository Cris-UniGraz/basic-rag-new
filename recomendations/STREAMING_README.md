# Streaming Response Feature

Este documento describe la nueva funcionalidad de streaming de respuestas implementada en el sistema RAG.

## Descripción

Se ha implementado una funcionalidad de streaming que permite recibir respuestas de los proveedores LLM (OpenAI y Meta) en tiempo real durante la Phase 6 del pipeline RAG, mejorando la experiencia del usuario al mostrar los tokens de respuesta a medida que se generan.

## Configuración

### Variable de Entorno

Se ha agregado una nueva variable de entorno:

```bash
STREAMING_RESPONSE=True  # Habilita streaming (valor por defecto)
STREAMING_RESPONSE=False # Deshabilita streaming
```

### Configuración en config.py

La variable se define en `backend/app/core/config.py`:

```python
# Streaming Response Configuration
STREAMING_RESPONSE: bool = Field(default=True)  # Enable streaming response in Phase 6
```

## Implementación

### 1. Proveedores LLM

Se ha agregado un método abstracto `generate_response_stream` a la clase base `LLMProvider`:

```python
@abstractmethod
async def generate_response_stream(
    self, 
    prompt: str, 
    system_prompt: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Generate a streaming response asynchronously."""
    pass
```

#### OpenAI Provider

Implementa streaming usando `stream=True` en `chat.completions.create()`:

```python
response = self.client.chat.completions.create(
    model=settings.AZURE_OPENAI_LLM_MODEL,
    messages=[...],
    stream=True
)

for update in response:
    if update.choices and update.choices[0].delta.content:
        yield update.choices[0].delta.content
```

#### Meta Provider

Implementa streaming usando `stream=True` en `client.complete()`:

```python
response = self.client.complete(
    messages=[...],
    model=settings.AZURE_META_LLM_MODEL,
    stream=True,
    ...
)

for update in response:
    if update.choices and update.choices[0].delta.content:
        yield update.choices[0].delta.content
```

### 2. RAG Service

Se ha modificado la Phase 6 del pipeline en `rag_service.py` para soportar streaming:

- **Si `STREAMING_RESPONSE=True`**: Usa `generate_response_stream()` y recolecta los chunks
- **Si `STREAMING_RESPONSE=False`**: Usa el comportamiento original con `chain.ainvoke()`
- **Fallback**: Si el streaming falla, automáticamente recurre al método no-streaming

## Características

### ✅ Beneficios

1. **Mejor experiencia de usuario**: Los tokens aparecen en tiempo real
2. **Respuesta más rápida**: El primer token llega más rápido
3. **Compatibilidad**: Funciona con ambos proveedores (OpenAI y Meta)
4. **Fallback automático**: Si el streaming falla, usa el método tradicional
5. **Configuración flexible**: Puede habilitarse/deshabilitarse fácilmente

### 🔧 Implementación

1. **Phase 6 únicamente**: El streaming solo se aplica en la generación final de respuesta
2. **Preserva funcionalidad**: Todas las otras fases (cache, retrieval, reranking) permanecen igual
3. **Logging mejorado**: Incluye información sobre el modo de streaming usado

## Uso

### 1. Configurar Variables de Entorno

Copiar `.env.example` a `.env` y configurar:

```bash
cp .env.example .env
```

Editar `.env` con tus credenciales:

```bash
# Habilitar streaming
STREAMING_RESPONSE=True

# Configurar proveedor
AZURE_LLM_MODEL=openai  # o 'meta'

# Configurar credenciales del proveedor elegido
AZURE_OPENAI_API_KEY=tu-api-key
AZURE_OPENAI_ENDPOINT=tu-endpoint
# ... otras variables
```

### 2. Ejecutar Pruebas

```bash
# Verificar que el streaming funciona
python test_streaming.py
```

### 3. Usar en Aplicación

El streaming se activa automáticamente en la Phase 6 cuando `STREAMING_RESPONSE=True`.

## Ejemplos de Respuesta

### Con Streaming Habilitado

```
Phase 6 (LLM generation) completed in 2.34s with streaming=enabled
Generated streaming response with 45 chunks
```

### Con Streaming Deshabilitado

```
Phase 6 (LLM generation) completed in 2.56s with streaming=disabled
```

## Consideraciones Técnicas

### Performance

- **Streaming**: Primer token más rápido, mejor UX
- **No-streaming**: Puede ser ligeramente más rápido para respuestas cortas

### Reliability

- **Fallback automático**: Si el streaming falla, usa método tradicional
- **Error handling**: Logs detallados para debugging
- **Timeout**: Mismo timeout aplicado a ambos métodos

### Compatibilidad

- ✅ OpenAI Azure
- ✅ Meta Azure
- ✅ Funciona en Phase 6 del pipeline
- ✅ Compatible con semantic cache
- ✅ Compatible con glossary terms

## Troubleshooting

### Error: "AttributeError: 'OpenAIProvider' object has no attribute 'generate_response_stream'"

Asegúrate de que los cambios en `llm_providers.py` se han aplicado correctamente.

### Streaming no funciona

1. Verificar `STREAMING_RESPONSE=True` en variables de entorno
2. Revisar logs para errores de streaming
3. Verificar credenciales del proveedor LLM

### Respuestas incompletas

- El streaming recolecta todos los chunks antes de devolver la respuesta completa
- Si hay problemas, el sistema hace fallback automático al método no-streaming

## Testing

### Backend Streaming
Para probar solo el backend:

```bash
# Test básico del backend
python test_streaming.py

# Test con diferentes proveedores
export AZURE_LLM_MODEL=openai && python test_streaming.py
export AZURE_LLM_MODEL=meta && python test_streaming.py

# Test con streaming deshabilitado
export STREAMING_RESPONSE=False && python test_streaming.py
```

### Full Streaming (Backend + Frontend)
Para probar el streaming completo:

```bash
# Test completo del streaming
python test_full_streaming.py

# Iniciar frontend con streaming
export ENABLE_FRONTEND_STREAMING=True && streamlit run frontend/app.py

# Iniciar frontend sin streaming
export ENABLE_FRONTEND_STREAMING=False && streamlit run frontend/app.py
```

### Configuración de Variables de Entorno

```bash
# Backend streaming
STREAMING_RESPONSE=True    # Habilita streaming en Phase 6

# Frontend streaming  
ENABLE_FRONTEND_STREAMING=True      # Habilita UI de streaming en frontend

# Para deshabilitar completamente
STREAMING_RESPONSE=False
ENABLE_FRONTEND_STREAMING=False
```

## ✅ Streaming Completo Implementado

### Backend
- ✅ Streaming en Phase 6 del pipeline RAG
- ✅ Soporte para OpenAI y Meta providers
- ✅ Endpoint `/api/chat/stream` con Server-Sent Events
- ✅ Fallback automático a método tradicional
- ✅ Variable `STREAMING_RESPONSE` para control

### Frontend
- ✅ Interfaz de streaming en tiempo real
- ✅ Actualización visual progresiva de respuestas
- ✅ Fallback automático si streaming falla
- ✅ Variable `ENABLE_FRONTEND_STREAMING` para control
- ✅ Indicadores de estado durante el streaming

### Características del Streaming Completo

1. **Experiencia Visual**: Los tokens aparecen progresivamente en la interfaz
2. **Respuesta Rápida**: El primer token aparece inmediatamente
3. **Fallback Robusto**: Si streaming falla, usa método tradicional automáticamente
4. **Control Granular**: Streaming se puede habilitar/deshabilitar independientemente en backend y frontend
5. **Compatibilidad**: Funciona con ambos proveedores LLM (OpenAI y Meta)

### Arquitectura del Streaming

```
[Usuario] -> [Frontend] -> [/api/chat/stream] -> [RAG Service] -> [LLM Provider]
    ↑                                                                    ↓
    |<-- Server-Sent Events <-- Streaming Response <-- Streaming API <--|
```

### Configuraciones Posibles

| STREAMING_RESPONSE | ENABLE_FRONTEND_STREAMING | Resultado |
|-------------------|------------------|-----------|
| True | True | ✅ Streaming completo |
| True | False | ⚡ Backend streaming, frontend tradicional |
| False | True | 🔄 Frontend intenta streaming, backend devuelve respuesta completa |
| False | False | 📝 Método tradicional completo |

### Próximos Pasos

El streaming completo está implementado y listo para usar. Para aprovechar al máximo:

1. **Configurar variables de entorno** según tus necesidades
2. **Probar con `test_full_streaming.py`** para verificar funcionalidad
3. **Iniciar la aplicación** y experimentar con respuestas en tiempo real
4. **Monitorear logs** para optimización adicional

El sistema ahora ofrece la mejor experiencia de usuario posible con respuestas que aparecen token por token en tiempo real.