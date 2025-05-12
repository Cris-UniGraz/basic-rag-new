# Step-Back Query y Generación Unificada de Consultas

## Descripción

El sistema Step-Back Query es una técnica avanzada que mejora la calidad de las respuestas al "dar un paso atrás" y generalizar las consultas específicas, permitiendo recuperar un contexto más amplio que podría ser relevante para responder a la pregunta original. Esta técnica se combina con un sistema de generación unificada de consultas que, en una sola llamada al LLM, genera todas las variaciones necesarias (consulta original, traducida, y versiones step-back en ambos idiomas).

## Funcionamiento

### Step-Back Query

El proceso de Step-Back Query funciona de la siguiente manera:

1. **Generalización conceptual**: Cuando un usuario hace una pregunta muy específica, el sistema genera automáticamente una versión más genérica que abarca un concepto más amplio.

2. **Contextualización mejorada**: Esta versión generalizada permite recuperar información contextual relevante que podría no ser captada por la consulta original más específica.

3. **Recuperación paralela**: Tanto la consulta original como la versión step-back se utilizan para recuperar documentos, ampliando el espectro de información relevante.

### Generación Unificada de Consultas

Para optimizar el rendimiento, el sistema utiliza una única llamada al LLM para generar todas las variantes de consulta necesarias:

1. **Entrada única, múltiples salidas**: Con una sola llamada al modelo, se generan:
   - La consulta original en el idioma de entrada
   - La traducción al otro idioma (alemán o inglés)
   - La versión step-back en el idioma original
   - La versión step-back en el idioma traducido

2. **Formato JSON**: El modelo responde en formato JSON estructurado que permite extraer fácilmente cada variante de consulta.

3. **Integración del glosario**: El proceso es sensible a términos del glosario, asegurando que los términos técnicos específicos de la Universidad de Graz se manejen correctamente.

## Integración con el Glosario

El sistema detecta cuando una consulta contiene términos del glosario y adapta su comportamiento:

1. **Traducción de términos**: Los términos del glosario se preservan exactamente sin traducir cuando se genera la consulta en el otro idioma.

2. **Generación de step-back con conocimiento del dominio**: Las consultas step-back se generan teniendo en cuenta el significado específico de los términos del glosario.

3. **Contexto enriquecido**: Se incluyen las definiciones de los términos del glosario en el prompt del LLM, asegurando que entienda su significado preciso.

## Ejemplos

### Consulta Original
> "¿Cómo puedo renovar mi UNIGRAzcard después de 30 días?"

### Variaciones Generadas
- **Traducción**: "How can I renew my UNIGRAzcard after 30 days?"
- **Step-back en alemán**: "¿Cuál es el proceso de renovación de tarjetas universitarias en la Universidad de Graz?"
- **Step-back en inglés**: "What is the renewal process for university cards at the University of Graz?"

### Beneficios en este ejemplo
La consulta step-back recuperará documentos generales sobre procesos de renovación de tarjetas que podrían no ser encontrados por la consulta específica, pero contienen información contextual vital para entender el proceso completo.

## Implementación Técnica

1. **Modelo Pydantic**: Utiliza un modelo Pydantic para validar el formato JSON de la respuesta del LLM.

2. **Prompt con ejemplos**: Incluye ejemplos de consultas y sus versiones step-back para ayudar al modelo a entender la tarea.

3. **Manejo robusto de errores**: Si la generación unificada falla, el sistema puede recurrir a métodos individuales como fallback.

4. **Optimización para contextualización**: Las consultas generadas se utilizan para recuperar y reordenar documentos en paralelo.

## Ventajas

1. **Eficiencia**: Reduce el número de llamadas al LLM, mejorando la latencia.

2. **Cobertura ampliada**: Las consultas step-back aumentan significativamente el recall de documentos relevantes.

3. **Respuestas más completas**: La combinación de resultados específicos y contextuales produce respuestas más informativas.

4. **Respeto por la terminología técnica**: La integración con el glosario asegura que los términos técnicos se manejen correctamente.

5. **Multilingüe por diseño**: Funciona fluidamente entre alemán e inglés, manteniendo la precisión terminológica en ambos idiomas.