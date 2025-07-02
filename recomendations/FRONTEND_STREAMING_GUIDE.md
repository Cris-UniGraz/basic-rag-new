# Frontend Streaming Implementation Guide

## Estado Actual

El frontend actualmente **NO soporta streaming** de respuestas. Usa requests síncronos y espera la respuesta completa.

## ¿Es necesario modificar el frontend?

### Opción 1: **NO modificar** (Recomendado inicialmente)
- ✅ El streaming funciona en backend
- ✅ Primera respuesta más rápida
- ✅ No requiere cambios complejos
- ⚠️ Usuario no ve tokens en tiempo real

### Opción 2: **Modificar para streaming completo**
- ✅ Experiencia de usuario óptima
- ❌ Cambios significativos requeridos
- ❌ Mayor complejidad

## Implementación de Streaming en Frontend

Si decides implementar streaming completo, necesitarías:

### 1. Crear endpoint de streaming en backend

```python
# En backend/app/api/endpoints/chat.py
from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
async def chat_stream(
    messages: List[ChatMessage],
    collection_name: Optional[str] = None,
):
    async def generate_stream():
        # Inicializar RAG service
        rag_service = # ... inicializar
        
        # Procesar query hasta Phase 5
        # ... código existente hasta Phase 5
        
        # Phase 6 con streaming
        if settings.STREAMING_RESPONSE:
            async for chunk in llm_provider.generate_response_stream(prompt):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        
        # Enviar metadata final
        yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )
```

### 2. Modificar función send_message en frontend

```python
# En frontend/app.py
import asyncio
import aiohttp

async def send_message_stream(message: str):
    """Send message with streaming support."""
    
    # Preparar mensajes
    messages = []
    for msg in st.session_state.messages:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    messages.append({"role": "user", "content": message})
    
    # Crear contenedor para respuesta streaming
    response_container = st.empty()
    current_response = ""
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                get_api_url("/chat/stream"),
                json=messages,
                headers={"Accept": "text/event-stream"}
            ) as response:
                
                async for line in response.content:
                    if line.startswith(b"data: "):
                        data = json.loads(line[6:])
                        
                        if "chunk" in data:
                            # Agregar chunk a respuesta actual
                            current_response += data["chunk"]
                            # Actualizar UI en tiempo real
                            response_container.write(current_response)
                        
                        elif data.get("done"):
                            # Respuesta completa, procesar metadata
                            sources = data.get("sources", [])
                            return current_response, sources
                            
    except Exception as e:
        st.error(f"Error en streaming: {e}")
        return None, []
```

### 3. Modificar handle_user_input

```python
def handle_user_input():
    """Handle user input with streaming support."""
    # ... código existente para input
    
    if not user_input:
        return
    
    # Agregar mensaje del usuario
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Crear contenedor para respuesta streaming
    with st.spinner("Conectando..."):
        # Ejecutar streaming de forma asíncrona
        response, sources = asyncio.run(send_message_stream(user_input))
        
        if response:
            # Agregar respuesta completa al historial
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            })
    
    st.rerun()
```

### 4. Alternativa más simple con Server-Sent Events

```python
# Versión simplificada usando requests con streaming
def send_message_stream_simple(message: str):
    """Simple streaming with requests."""
    import requests
    
    # Preparar datos
    messages = [...]  # Como antes
    
    # Crear contenedor para respuesta
    response_container = st.empty()
    current_response = ""
    
    try:
        response = requests.post(
            get_api_url("/chat/stream"),
            json=messages,
            stream=True,
            timeout=180
        )
        
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data = json.loads(line[6:])
                
                if "chunk" in data:
                    current_response += data["chunk"]
                    response_container.write(current_response)
                    time.sleep(0.01)  # Pequeña pausa para suavizar
                
                elif data.get("done"):
                    sources = data.get("sources", [])
                    return current_response, sources
                    
    except Exception as e:
        st.error(f"Error: {e}")
        return None, []
```

## Consideraciones Técnicas

### Limitaciones de Streamlit
- **Re-renderizado**: Streamlit re-renderiza toda la página en cada actualización
- **Estado**: Mantener el estado durante streaming puede ser complejo
- **Performance**: Actualizaciones frecuentes pueden ser lentas

### Alternativas más simples
1. **Typing effect**: Simular streaming con JavaScript
2. **Chunks discretos**: Mostrar respuesta en bloques grandes
3. **Progress indicators**: Mostrar progreso sin streaming real

## Recomendación Final

**Para tu caso de uso, NO es necesario modificar el frontend inicialmente** porque:

1. ✅ El streaming en backend ya mejora el rendimiento
2. ✅ La implementación actual funciona bien
3. ✅ Puedes agregar streaming frontend más tarde si es necesario
4. ✅ La configuración `STREAMING_RESPONSE` te permite testear ambos modos

El streaming en el backend es suficiente para obtener los beneficios de rendimiento principales.