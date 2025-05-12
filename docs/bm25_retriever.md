# BM25Retriever para Búsquedas por Palabras Clave

## Descripción

El BM25Retriever es una técnica de recuperación basada en el algoritmo BM25 (Best Matching 25), que proporciona búsquedas de alta calidad basadas en palabras clave. A diferencia de los retrievers basados en embeddings, que buscan similitud semántica, BM25 se centra en la coincidencia exacta de términos y su frecuencia, lo que lo hace complementario a los métodos de búsqueda vectorial.

## Funcionamiento

El BM25Retriever opera siguiendo estos principios:

1. **Coincidencia de términos**: Busca documentos que contengan exactamente las palabras clave utilizadas en la consulta.

2. **Ponderación TF-IDF mejorada**: Evalúa la importancia de un término basándose en:
   - Frecuencia del término en el documento (TF)
   - Inverso de la frecuencia en la colección completa (IDF)
   - Pero con saturación para evitar sesgos por términos muy frecuentes

3. **Normalización por longitud**: Considera la longitud del documento para evitar favorecer documentos más largos.

4. **Rankeo**: Ordena los documentos según una puntuación calculada que refleja la relevancia para la consulta.

## Ventajas en el Contexto RAG

La inclusión del BM25Retriever en nuestro ensemble ofrece varias ventajas:

1. **Complementariedad**: Funciona de manera diferente a los retrievers basados en embeddings, capturando documentos que podrían pasarse por alto con métodos puramente semánticos.

2. **Precisión para términos técnicos**: Excelente para encontrar documentos que contengan términos técnicos específicos, números o acrónimos que podrían ser difíciles de capturar mediante similaridad vectorial.

3. **Robustez ante terminología especializada**: Particularmente útil para términos del glosario de la Universidad de Graz que deben encontrarse exactamente como aparecen.

4. **Independencia lingüística**: Funciona bien tanto para documentos en alemán como en inglés sin necesidad de modelos específicos por idioma.

## Implementación

Nuestra implementación del BM25Retriever:

1. **Reutiliza documentos padres**: Aprovecha los documentos ya almacenados en MongoDB para la recuperación de documentos padres.

2. **Integración en Ensemble**: Se integra como un componente adicional en el EnsembleRetriever con un peso del 15%.

3. **Configuración dinámica**: Ajusta automáticamente su configuración y los pesos de otros retrievers según su disponibilidad.

4. **Gestión de errores robusta**: El sistema funciona correctamente incluso si el BM25Retriever no está disponible para una colección particular.

## Casos de Uso Óptimos

El BM25Retriever destaca particularmente en estos escenarios:

1. **Búsqueda de términos específicos**: Consultas que incluyen nombres propios, códigos o identificadores específicos.

2. **Documentación técnica**: Búsqueda en manuales técnicos o documentos de procedimientos donde la terminología exacta es crucial.

3. **Consultas con términos del glosario**: Preguntas que incluyen términos específicos como "UNIGRAzcard", "PSP-Code" o "flExam".

4. **Complemento a búsquedas semánticas**: Como parte de una estrategia de búsqueda que combina diferentes enfoques.

## Limitaciones

Es importante reconocer que el BM25Retriever también tiene limitaciones:

1. No captura relaciones semánticas entre términos que no comparten palabras.
2. Sensible a errores ortográficos o variaciones de palabras.
3. Puede fallar con consultas conceptuales abstractas.

Estas limitaciones se compensan en nuestro sistema al combinar BM25 con retrievers semánticos en el ensemble.