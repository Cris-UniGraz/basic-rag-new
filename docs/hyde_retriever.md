# Hypothetical Document Embedder (HyDE) en el Sistema RAG

## Descripción

El Hypothetical Document Embedder (HyDE) es una técnica avanzada que mejora significativamente la recuperación de documentos relevantes al generar un documento "hipotético" que podría responder a la consulta del usuario, y luego utilizar ese documento como base para la búsqueda vectorial.

## Funcionamiento

HyDE opera siguiendo estos pasos:

1. **Generación de documento hipotético**: Cuando el usuario realiza una consulta, el sistema genera un documento sintético que hipotéticamente podría responder a esa consulta.

2. **Embedding del documento**: El documento hipotético generado se convierte en un vector mediante el modelo de embeddings.

3. **Búsqueda de similitud**: Este vector se utiliza para encontrar documentos reales en la base de conocimiento que sean semánticamente similares al documento hipotético.

4. **Recuperación**: Se recuperan los documentos más similares, que probablemente contengan información relevante para la consulta original.

## Integración con el Glosario

Nuestra implementación de HyDE incluye una integración con el sistema de glosario:

1. Se detectan términos del glosario en la consulta del usuario
2. Si se encuentran términos especializados, se incluyen sus definiciones en el prompt para el LLM
3. Esto permite que el documento hipotético generado comprenda y utilice correctamente los términos técnicos específicos
4. Resulta en una recuperación más precisa cuando se utilizan términos especializados de la Universidad de Graz

## Beneficios

HyDE ofrece varias ventajas clave:

1. **Mejor comprensión del contexto**: Genera un documento completo que captura el contexto de la consulta, no solo palabras clave.

2. **Superación de la brecha léxica**: Ayuda a encontrar documentos relevantes incluso cuando utilizan terminología diferente a la consulta.

3. **Robustez ante consultas complejas**: Funciona bien con consultas largas o complejas que podrían ser difíciles de procesar directamente.

4. **Complementariedad**: Se combina eficazmente con otras técnicas de recuperación como MultiQuery y ParentDocumentRetriever.

## Implementación Técnica

Nuestra implementación de HyDE incluye:

1. La clase `GlossaryAwareHyDEEmbedder` que adapta la generación del documento hipotético según los términos de glosario encontrados.

2. Integración en el EnsembleRetriever con un peso del 20%, equilibrado con los otros métodos de recuperación.

3. Gestión de errores robusta que permite que el sistema funcione incluso si HyDE no está disponible para una colección específica.

## Ejemplo

Para la consulta:
> "¿Qué necesito para renovar mi UNIGRAzcard?"

HyDE podría generar un documento hipotético como:

```
Para renovar tu UNIGRAzcard, que es la tarjeta que utilizan los empleados, estudiantes y profesores de la Universidad de Graz para acceder a diversos servicios, necesitas acudir a un lector de actualización o a una cerradura en línea. La tarjeta debe renovarse cada 30 días para mantener su validez. Asegúrate de tener tu identificación universitaria contigo. El proceso de renovación es rápido y simple, solo necesitas pasar la tarjeta por el lector y el sistema la actualizará automáticamente...
```

Este documento hipotético contiene información contextual relevante sobre la UNIGRAzcard (extraída del glosario) y sirve para encontrar documentos reales que contengan información similar.

## Consideraciones de Rendimiento

- La generación del documento hipotético añade una pequeña latencia adicional al proceso de recuperación.
- Este costo se compensa con la mejora significativa en la calidad de los resultados recuperados.
- El sistema incluye optimizaciones para reducir la latencia, como la ejecución paralela con otros métodos de recuperación.