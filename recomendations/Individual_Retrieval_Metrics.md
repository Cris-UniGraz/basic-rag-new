# Implementación de Métricas por Retriever Individual

## 📋 **Lista de Tareas para Implementar Métricas por Retriever**

### **Fase 1: Extensión del MetricsManager**

#### **Tarea 1.1: Agregar método log_retriever_performance()**
- **Archivo**: `backend/app/core/metrics_manager.py`
- **Descripción**: Implementar método para registrar performance individual de cada retriever
- **Estimación**: 30 minutos

```python
def log_retriever_performance(self, retriever_type: str, query: str, 
                              execution_time: float, documents_found: int,
                              success: bool, error_details: str = None):
    """
    Registra performance individual de cada retriever.
    
    Args:
        retriever_type: 'base_vector', 'parent_doc', 'multi_query', 'hyde', 'bm25'
        query: Query procesada por el retriever
        execution_time: Tiempo de ejecución en segundos
        documents_found: Número de documentos recuperados
        success: Si la operación fue exitosa
        error_details: Detalles del error si falló
    """
    if 'retriever_performance' not in self.metrics:
        self.metrics['retriever_performance'] = defaultdict(list)
        
    self.metrics['retriever_performance'][retriever_type].append({
        'timestamp': datetime.now().isoformat(),
        'execution_time': execution_time,
        'documents_found': documents_found,
        'success': success,
        'error_details': error_details,
        'query_preview': query[:50] + '...' if len(query) > 50 else query,
        'weight': self._get_retriever_weight(retriever_type)
    })
    
    # Registrar también como operación general
    self.log_operation(
        operation_type=f"retriever_{retriever_type}",
        duration=execution_time,
        success=success,
        details={
            'documents_found': documents_found,
            'retriever_type': retriever_type
        }
    )
```

#### **Tarea 1.2: Agregar métodos de análisis estadístico**
- **Archivo**: `backend/app/core/metrics_manager.py`
- **Descripción**: Métodos para analizar performance por retriever
- **Estimación**: 45 minutos

```python
def get_retriever_statistics(self) -> Dict[str, Any]:
    """Obtiene estadísticas detalladas por retriever."""
    if 'retriever_performance' not in self.metrics:
        return {}
    
    stats = {}
    for retriever_type, performances in self.metrics['retriever_performance'].items():
        if not performances:
            continue
            
        execution_times = [p['execution_time'] for p in performances]
        document_counts = [p['documents_found'] for p in performances]
        success_count = sum(1 for p in performances if p['success'])
        
        stats[retriever_type] = {
            'total_operations': len(performances),
            'success_rate': success_count / len(performances),
            'avg_execution_time': float(np.mean(execution_times)),
            'p50_execution_time': float(np.percentile(execution_times, 50)),
            'p90_execution_time': float(np.percentile(execution_times, 90)),
            'p99_execution_time': float(np.percentile(execution_times, 99)),
            'avg_documents_found': float(np.mean(document_counts)),
            'min_execution_time': float(np.min(execution_times)),
            'max_execution_time': float(np.max(execution_times)),
            'weight': self._get_retriever_weight(retriever_type),
            'last_operation': performances[-1]['timestamp']
        }
    
    return stats

def get_retriever_comparison(self) -> Dict[str, Any]:
    """Compara performance entre retrievers."""
    stats = self.get_retriever_statistics()
    if not stats:
        return {}
    
    # Comparación de métricas clave
    comparison = {
        'fastest_retriever': min(stats.keys(), key=lambda k: stats[k]['avg_execution_time']),
        'most_productive': max(stats.keys(), key=lambda k: stats[k]['avg_documents_found']),
        'most_reliable': max(stats.keys(), key=lambda k: stats[k]['success_rate']),
        'execution_time_ranking': sorted(stats.keys(), key=lambda k: stats[k]['avg_execution_time']),
        'productivity_ranking': sorted(stats.keys(), key=lambda k: stats[k]['avg_documents_found'], reverse=True),
        'total_operations_by_type': {k: v['total_operations'] for k, v in stats.items()}
    }
    
    return comparison
```

### **Fase 2: Instrumentación del RAGService**

#### **Tarea 2.1: Instrumentar get_retriever() method**
- **Archivo**: `backend/app/services/rag_service.py`
- **Descripción**: Agregar timing individual para cada retriever en la creación del ensemble
- **Estimación**: 60 minutos

```python
async def get_retriever(self, collection_name: str, top_k: int = 3, max_concurrency: int = 5):
    """Instrumentar con métricas por retriever individual."""
    
    retriever_creation_times = {}
    retriever_errors = {}
    
    try:
        # Base Vector Retriever
        start_time = time.time()
        try:
            base_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
            creation_time = time.time() - start_time
            retriever_creation_times['base_vector'] = creation_time
            
            # Log metrics
            self.metrics_manager.log_retriever_performance(
                retriever_type='base_vector',
                query='',  # No specific query for creation
                execution_time=creation_time,
                documents_found=0,  # Creation phase
                success=True
            )
            
        except Exception as e:
            creation_time = time.time() - start_time
            retriever_errors['base_vector'] = str(e)
            self.metrics_manager.log_retriever_performance(
                retriever_type='base_vector',
                query='',
                execution_time=creation_time,
                documents_found=0,
                success=False,
                error_details=str(e)
            )
            logger.error(f"Failed to create base_vector retriever: {e}")
            
        # Parent Document Retriever
        start_time = time.time()
        try:
            parent_retriever = await self.create_parent_retriever(
                vectorstore, parent_collection_name, top_k
            )
            creation_time = time.time() - start_time
            retriever_creation_times['parent_doc'] = creation_time
            
            self.metrics_manager.log_retriever_performance(
                retriever_type='parent_doc',
                query='',
                execution_time=creation_time,
                documents_found=0,
                success=True
            )
            
        except Exception as e:
            creation_time = time.time() - start_time
            retriever_errors['parent_doc'] = str(e)
            self.metrics_manager.log_retriever_performance(
                retriever_type='parent_doc',
                query='',
                execution_time=creation_time,
                documents_found=0,
                success=False,
                error_details=str(e)
            )
            
        # Repetir para multi_query, hyde, y bm25 retrievers...
        
        # Log ensemble creation summary
        total_creation_time = sum(retriever_creation_times.values())
        async_metadata_processor.log_async(
            "INFO",
            "Ensemble retriever creation completed",
            {
                "collection": collection_name,
                "total_creation_time": total_creation_time,
                "retriever_times": retriever_creation_times,
                "retriever_errors": retriever_errors,
                "successful_retrievers": len(retriever_creation_times),
                "failed_retrievers": len(retriever_errors)
            }
        )
        
    except Exception as e:
        logger.error(f"Error in get_retriever instrumentation: {e}")
        raise
```

#### **Tarea 2.2: Instrumentar retrieve_context_without_reranking()**
- **Archivo**: `backend/app/services/rag_service.py`
- **Descripción**: Agregar métricas durante la ejecución de retrieval
- **Estimación**: 45 minutos

```python
async def retrieve_context_without_reranking(
    self, query: str, retriever: Any, chat_history: List[Tuple[str, str]] = []
):
    """Instrumentar retrieval execution con métricas detalladas."""
    
    start_time = time.time()
    retriever_type = self._identify_retriever_type(retriever)
    
    try:
        # Format chat history
        formatted_history = []
        for human_msg, ai_msg in chat_history:
            formatted_history.extend([
                HumanMessage(content=human_msg),
                AIMessage(content=ai_msg)
            ])
        
        # Execute retrieval
        retrieved_docs = await retriever.ainvoke({
            "input": query,
            "chat_history": formatted_history
        })
        
        execution_time = time.time() - start_time
        
        # Log individual retriever performance
        self.metrics_manager.log_retriever_performance(
            retriever_type=retriever_type,
            query=query,
            execution_time=execution_time,
            documents_found=len(retrieved_docs),
            success=True
        )
        
        # Log async performance data
        async_metadata_processor.record_performance_async(
            f"retrieval_{retriever_type}",
            execution_time,
            True,
            {
                "query_preview": query[:100],
                "num_docs": len(retrieved_docs),
                "retriever_type": retriever_type,
                "sources": [doc.metadata.get('source', 'unknown') for doc in retrieved_docs][:5]
            }
        )
        
        return retrieved_docs
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        # Log error metrics
        self.metrics_manager.log_retriever_performance(
            retriever_type=retriever_type,
            query=query,
            execution_time=execution_time,
            documents_found=0,
            success=False,
            error_details=str(e)
        )
        
        async_metadata_processor.log_async(
            "ERROR",
            f"Retrieval failed for {retriever_type}",
            {
                "error": str(e),
                "query_preview": query[:100],
                "retriever_type": retriever_type,
                "execution_time": execution_time
            },
            priority=3
        )
        
        return []

def _identify_retriever_type(self, retriever) -> str:
    """Identifica el tipo de retriever basado en su clase."""
    class_name = retriever.__class__.__name__
    
    if 'MultiQuery' in class_name or 'GlossaryAwareMultiQuery' in class_name:
        return 'multi_query'
    elif 'Parent' in class_name:
        return 'parent_doc'
    elif 'HyDE' in class_name or 'Hypothetical' in class_name:
        return 'hyde'
    elif 'BM25' in class_name:
        return 'bm25'
    elif 'History' in class_name:
        return 'history_aware'  # Wrapper del ensemble
    else:
        return 'base_vector'
```

### **Fase 3: Métricas Prometheus**

#### **Tarea 3.1: Agregar métricas Prometheus específicas**
- **Archivo**: `backend/app/core/metrics.py`
- **Descripción**: Definir métricas Prometheus para retrievers individuales
- **Estimación**: 30 minutos

```python
# Agregar al archivo metrics.py

# Métricas de duración por retriever
RETRIEVER_DURATION = Histogram(
    'rag_retriever_duration_seconds',
    'Duration of individual retriever operations',
    ['retriever_type', 'collection', 'operation']  # operation: creation, retrieval
)

# Documentos encontrados por retriever
RETRIEVER_DOCUMENTS_FOUND = Histogram(
    'rag_retriever_documents_found',
    'Number of documents found by each retriever',
    ['retriever_type', 'collection']
)

# Tasa de éxito por retriever
RETRIEVER_SUCCESS_RATE = Counter(
    'rag_retriever_operations_total',
    'Total retriever operations',
    ['retriever_type', 'status']  # status: success, error
)

# Peso efectivo del retriever en el ensemble
RETRIEVER_WEIGHT = Gauge(
    'rag_retriever_weight',
    'Current weight of retriever in ensemble',
    ['retriever_type', 'collection']
)

# Función helper para registrar métricas
def record_retriever_metrics(retriever_type: str, collection: str, 
                           execution_time: float, documents_found: int,
                           success: bool, operation: str = 'retrieval'):
    """Record Prometheus metrics for individual retriever performance."""
    
    # Duration
    RETRIEVER_DURATION.labels(
        retriever_type=retriever_type,
        collection=collection,
        operation=operation
    ).observe(execution_time)
    
    # Documents found (only for successful operations)
    if success:
        RETRIEVER_DOCUMENTS_FOUND.labels(
            retriever_type=retriever_type,
            collection=collection
        ).observe(documents_found)
    
    # Success rate
    status = 'success' if success else 'error'
    RETRIEVER_SUCCESS_RATE.labels(
        retriever_type=retriever_type,
        status=status
    ).inc()
```

#### **Tarea 3.2: Integrar métricas en el pipeline**
- **Archivo**: `backend/app/services/rag_service.py`
- **Descripción**: Integrar las métricas Prometheus en los métodos instrumentados
- **Estimación**: 20 minutos

```python
# En los métodos instrumentados, agregar:
from app.core.metrics import record_retriever_metrics

# Después de log_retriever_performance(), agregar:
record_retriever_metrics(
    retriever_type=retriever_type,
    collection=collection_name,
    execution_time=execution_time,
    documents_found=len(retrieved_docs),
    success=success,
    operation='retrieval'
)
```

### **Fase 4: AsyncMetadataProcessor Extension**

#### **Tarea 4.1: Agregar nuevo tipo de evento**
- **Archivo**: `backend/app/core/async_metadata_processor.py`
- **Descripción**: Extender para soportar eventos específicos de retrievers
- **Estimación**: 25 minutos

```python
# Agregar a MetadataType enum
class MetadataType(Enum):
    LOG = "log"
    METRIC = "metric"
    ERROR = "error"
    PERFORMANCE = "performance"
    API_CALL = "api_call"
    RETRIEVER_PERFORMANCE = "retriever_performance"  # NUEVO
    PIPELINE_PHASE = "pipeline_phase"                # NUEVO

# Agregar método específico
def record_retriever_performance_async(
    self,
    retriever_type: str,
    query: str,
    execution_time: float,
    documents_found: int,
    success: bool,
    error_details: Optional[str] = None,
    collection: str = "default"
) -> bool:
    """
    Registra performance de retriever de forma asíncrona.
    
    Args:
        retriever_type: Tipo de retriever
        query: Query procesada
        execution_time: Tiempo de ejecución
        documents_found: Documentos encontrados
        success: Si fue exitosa
        error_details: Detalles del error
        collection: Nombre de la colección
        
    Returns:
        True si se procesó correctamente
    """
    data = {
        "retriever_type": retriever_type,
        "query_preview": query[:100] + "..." if len(query) > 100 else query,
        "execution_time": execution_time,
        "documents_found": documents_found,
        "success": success,
        "error_details": error_details,
        "collection": collection,
        "timestamp": datetime.now().isoformat()
    }
    
    priority = 3 if not success else 1  # Alta prioridad para errores
    return self.queue_event(MetadataType.RETRIEVER_PERFORMANCE, data, priority)

# Agregar método de procesamiento
async def _process_retriever_performance_events(
    self,
    events: List[MetadataEvent],
    config: Dict[str, Any]
) -> None:
    """Procesar eventos de performance de retrievers."""
    if config.get("file_enabled", True):
        await self._write_retriever_performance_to_file(events)

async def _write_retriever_performance_to_file(self, events: List[MetadataEvent]) -> None:
    """Escribir métricas de retrievers a archivo."""
    try:
        perf_file = self.metrics_directory / f"retriever_performance_{datetime.now().strftime('%Y%m%d')}.jsonl"
        async with asyncio.Lock():
            with open(perf_file, "a", encoding="utf-8") as f:
                for event in events:
                    event_dict = self._make_serializable(event)
                    json.dump(event_dict, f, ensure_ascii=False)
                    f.write("\n")
    except Exception as e:
        loguru_logger.error(f"Error escribiendo retriever performance a archivo: {e}")
```

### **Fase 5: Dashboard y Visualización**

#### **Tarea 5.1: Configurar Grafana dashboard**
- **Archivo**: `monitoring/grafana/dashboards/retriever-performance.json`
- **Descripción**: Dashboard específico para métricas de retrievers
- **Estimación**: 45 minutos

```json
{
  "dashboard": {
    "id": null,
    "title": "RAG Retriever Performance",
    "tags": ["rag", "retrievers", "performance"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Retriever Execution Times",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(rag_retriever_duration_seconds) by (retriever_type)",
            "legendFormat": "{{retriever_type}}"
          }
        ]
      },
      {
        "title": "Documents Found by Retriever",
        "type": "bargauge",
        "targets": [
          {
            "expr": "avg(rag_retriever_documents_found) by (retriever_type)",
            "legendFormat": "{{retriever_type}}"
          }
        ]
      },
      {
        "title": "Retriever Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(rag_retriever_operations_total{status=\"success\"}[5m]) / rate(rag_retriever_operations_total[5m])",
            "legendFormat": "{{retriever_type}}"
          }
        ]
      },
      {
        "title": "Retriever Performance Timeline",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rag_retriever_duration_seconds",
            "legendFormat": "{{retriever_type}} - {{operation}}"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

#### **Tarea 5.2: Configurar alertas**
- **Archivo**: `monitoring/alert_rules.yml`
- **Descripción**: Alertas para performance de retrievers
- **Estimación**: 30 minutos

```yaml
# Agregar a alert_rules.yml
groups:
  - name: retriever_performance
    rules:
      - alert: RetrieverHighLatency
        expr: avg(rag_retriever_duration_seconds) by (retriever_type) > 2.0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected for {{ $labels.retriever_type }} retriever"
          description: "{{ $labels.retriever_type }} retriever has average latency of {{ $value }}s"
          
      - alert: RetrieverLowSuccessRate
        expr: rate(rag_retriever_operations_total{status="success"}[5m]) / rate(rag_retriever_operations_total[5m]) < 0.95
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "Low success rate for {{ $labels.retriever_type }} retriever"
          description: "{{ $labels.retriever_type }} retriever success rate is {{ $value | humanizePercentage }}"
          
      - alert: RetrieverNoDocuments
        expr: avg(rag_retriever_documents_found) by (retriever_type) < 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "{{ $labels.retriever_type }} retriever finding few documents"
          description: "{{ $labels.retriever_type }} retriever average documents found: {{ $value }}"
```

### **Fase 6: Testing y Validación**

#### **Tarea 6.1: Crear tests unitarios**
- **Archivo**: `backend/app/tests/test_retriever_metrics.py`
- **Descripción**: Tests para verificar la correcta instrumentación
- **Estimación**: 60 minutos

```python
import pytest
import time
from unittest.mock import Mock, patch
from app.core.metrics_manager import MetricsManager
from app.services.rag_service import RAGService

class TestRetrieverMetrics:
    
    def test_log_retriever_performance(self):
        """Test logging de performance de retriever."""
        metrics_manager = MetricsManager()
        
        # Log successful operation
        metrics_manager.log_retriever_performance(
            retriever_type='base_vector',
            query='test query',
            execution_time=0.5,
            documents_found=10,
            success=True
        )
        
        stats = metrics_manager.get_retriever_statistics()
        assert 'base_vector' in stats
        assert stats['base_vector']['total_operations'] == 1
        assert stats['base_vector']['success_rate'] == 1.0
        assert stats['base_vector']['avg_execution_time'] == 0.5
        
    def test_retriever_error_logging(self):
        """Test logging de errores de retriever."""
        metrics_manager = MetricsManager()
        
        # Log failed operation
        metrics_manager.log_retriever_performance(
            retriever_type='hyde',
            query='test query',
            execution_time=1.2,
            documents_found=0,
            success=False,
            error_details='Connection timeout'
        )
        
        stats = metrics_manager.get_retriever_statistics()
        assert stats['hyde']['success_rate'] == 0.0
        
    @patch('app.services.rag_service.time.time')
    def test_retriever_timing_instrumentation(self, mock_time):
        """Test instrumentación de timing en RAGService."""
        # Mock time progression
        mock_time.side_effect = [0, 0.5, 1.0, 1.5]  # Start, end times
        
        rag_service = RAGService(Mock())
        rag_service.metrics_manager = Mock()
        
        # Test timing capture
        # Implementation would depend on specific method structure
        
    def test_retriever_comparison(self):
        """Test comparación entre retrievers."""
        metrics_manager = MetricsManager()
        
        # Log different retrievers
        metrics_manager.log_retriever_performance('base_vector', 'q1', 0.3, 5, True)
        metrics_manager.log_retriever_performance('multi_query', 'q1', 0.8, 15, True)
        metrics_manager.log_retriever_performance('hyde', 'q1', 1.2, 3, True)
        
        comparison = metrics_manager.get_retriever_comparison()
        assert comparison['fastest_retriever'] == 'base_vector'
        assert comparison['most_productive'] == 'multi_query'
```

#### **Tarea 6.2: Crear script de validación**
- **Archivo**: `backend/scripts/validate_retriever_metrics.py`
- **Descripción**: Script para validar que las métricas se registran correctamente
- **Estimación**: 30 minutos

```python
#!/usr/bin/env python3
"""Script para validar métricas de retrievers."""

import asyncio
import time
from app.core.metrics_manager import MetricsManager
from app.core.async_metadata_processor import async_metadata_processor
from app.services.rag_service import create_rag_service
from app.services.llm_service import llm_service

async def test_retriever_metrics():
    """Test completo de métricas de retrievers."""
    
    print("🔍 Iniciando validación de métricas de retrievers...")
    
    # Initialize services
    await async_metadata_processor.start()
    rag_service = create_rag_service(llm_service)
    await rag_service.ensure_initialized()
    
    try:
        # Test retriever creation timing
        print("\n📊 Testing retriever creation metrics...")
        start_time = time.time()
        
        retriever = await rag_service.get_retriever("test_collection", top_k=5)
        creation_time = time.time() - start_time
        
        print(f"✅ Retriever created in {creation_time:.2f}s")
        
        # Test retrieval metrics
        print("\n🔍 Testing retrieval execution metrics...")
        test_query = "What is the University of Graz?"
        
        start_time = time.time()
        documents = await rag_service.retrieve_context_without_reranking(
            test_query, retriever, []
        )
        retrieval_time = time.time() - start_time
        
        print(f"✅ Retrieved {len(documents)} documents in {retrieval_time:.2f}s")
        
        # Check metrics were recorded
        print("\n📈 Checking recorded metrics...")
        stats = rag_service.metrics_manager.get_retriever_statistics()
        
        if stats:
            print("✅ Retriever statistics found:")
            for retriever_type, data in stats.items():
                print(f"   - {retriever_type}: {data['total_operations']} ops, "
                      f"{data['avg_execution_time']:.3f}s avg, "
                      f"{data['success_rate']:.2%} success rate")
        else:
            print("❌ No retriever statistics found")
            
        # Check comparison data
        comparison = rag_service.metrics_manager.get_retriever_comparison()
        if comparison:
            print("✅ Retriever comparison data:")
            print(f"   - Fastest: {comparison.get('fastest_retriever', 'N/A')}")
            print(f"   - Most productive: {comparison.get('most_productive', 'N/A')}")
        
        print("\n🎉 Validación completada exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante validación: {e}")
        raise
    finally:
        await async_metadata_processor.stop()

if __name__ == "__main__":
    asyncio.run(test_retriever_metrics())
```

## **📋 Resumen de Tareas por Prioridad**

### **🔴 Alta Prioridad (Core Implementation)**
1. **Tarea 1.1**: Extensión MetricsManager - 30 min
2. **Tarea 2.1**: Instrumentación RAGService - 60 min  
3. **Tarea 3.1**: Métricas Prometheus - 30 min

### **🟡 Media Prioridad (Analytics & Monitoring)**
4. **Tarea 1.2**: Métodos estadísticos - 45 min
5. **Tarea 2.2**: Instrumentación retrieval - 45 min
6. **Tarea 4.1**: AsyncMetadataProcessor - 25 min

### **🟢 Baja Prioridad (Visualization & Testing)**
7. **Tarea 5.1**: Grafana dashboard - 45 min
8. **Tarea 5.2**: Alertas - 30 min  
9. **Tarea 6.1**: Tests unitarios - 60 min
10. **Tarea 6.2**: Script validación - 30 min

## **⏱️ Estimación Total**
- **Tiempo estimado total**: ~6.5 horas
- **Implementación mínima viable**: ~2 horas (tareas 1.1, 2.1, 3.1)
- **Implementación completa**: ~6.5 horas (todas las tareas)

## **🎯 Beneficios Esperados**

### **Observabilidad Mejorada**
- Visibility granular de performance por retriever
- Identificación de bottlenecks específicos
- Optimización basada en datos reales

### **Debugging Avanzado** 
- Traces detallados de failures por retriever
- Análisis de effectiveness relativa
- Tuning de pesos del ensemble basado en métricas

### **Production Monitoring**
- Alertas proactivas por retriever individual
- Dashboards especializados
- SLA tracking por componente

### **Performance Optimization**
- Data-driven tuning de retriever weights
- Identificación de retrievers problemáticos
- Optimización selectiva de componentes lentos