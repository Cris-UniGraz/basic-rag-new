# Integración de Glosarios en el Sistema RAG

## Descripción

La integración de glosarios añade sensibilidad a terminología específica en las consultas de los usuarios. Esto mejora significativamente la calidad de las respuestas cuando los usuarios utilizan términos específicos de la Universidad de Graz que podrían tener significados especiales o técnicos.

## Funcionalidades implementadas

### 1. Módulo de Glosario

Se ha creado un nuevo módulo `glossary.py` en `backend/app/utils/` que contiene:

- **Diccionarios de glosario**: Términos y sus definiciones tanto en alemán como en inglés
- **Función `find_glossary_terms`**: Identifica términos del glosario en una consulta del usuario
- **Función `find_glossary_terms_with_explanation`**: Devuelve términos coincidentes junto con sus explicaciones

### 2. Integración en la Traducción de Consultas

La función `translate_query` ha sido mejorada para:

- Detectar términos del glosario en la consulta original
- Instruir al LLM para que no traduzca estos términos específicos
- Preservar los términos exactos en la traducción para mantener su significado técnico

### 3. Integración en la Generación de Respuestas

El proceso de generación de respuestas ahora:

- Identifica términos del glosario en la consulta del usuario
- Incluye definiciones de estos términos en el prompt enviado al LLM
- Crea un prompt especializado que instruye al modelo a considerar estos significados específicos

## Beneficios

1. **Mayor precisión en respuestas**: Los términos específicos de la universidad se entienden correctamente
2. **Consistencia terminológica**: Se mantiene la terminología correcta entre consultas y respuestas
3. **Mejor experiencia de usuario**: Respuestas más relevantes cuando se utilizan términos técnicos

## Ejemplo de flujo

1. Usuario pregunta: "¿Cómo puedo renovar mi UNIGRAzcard?"
2. El sistema identifica "UNIGRAzcard" como término del glosario
3. Si se traduce la consulta, "UNIGRAzcard" se mantiene sin traducir
4. La explicación de "UNIGRAzcard" se incluye en el prompt del LLM
5. La respuesta generada refleja el conocimiento correcto sobre qué es una UNIGRAzcard y cómo renovarla

## Estructura del Glosario

El glosario está organizado como dos diccionarios (alemán e inglés) donde:
- Las **claves** son los términos específicos (ej. "UNIGRAzcard", "flExam")
- Los **valores** son explicaciones detalladas de cada término

## Mantenimiento y Actualización

Para añadir nuevos términos al glosario:

1. Editar `backend/app/utils/glossary.py` 
2. Añadir el nuevo término y su definición a ambos diccionarios (alemán e inglés)
3. No se requieren cambios adicionales en el código, ya que las funciones detectarán automáticamente los nuevos términos

## Futuras Mejoras

Algunas posibles mejoras para esta integración:

1. Añadir un sistema de administración de glosarios a través de una interfaz web
2. Implementar detección de variantes y errores tipográficos en términos del glosario
3. Integrar un sistema de retroalimentación para mejorar las definiciones del glosario
4. Expandir la integración a otras partes del sistema como la recuperación de documentos