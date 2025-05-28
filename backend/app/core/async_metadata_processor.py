import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from collections import deque
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from loguru import logger as loguru_logger


class MetadataType(Enum):
    """Tipos de metadatos que pueden ser procesados."""
    LOG = "log"
    METRIC = "metric"
    ERROR = "error"
    PERFORMANCE = "performance"
    API_CALL = "api_call"


@dataclass
class MetadataEvent:
    """Evento de metadatos para procesamiento asíncrono."""
    id: str
    event_type: MetadataType
    timestamp: float
    data: Dict[str, Any]
    priority: int = 1  # 1=normal, 2=alta, 3=crítica
    retry_count: int = 0
    max_retries: int = 3


class AsyncMetadataProcessor:
    """
    Procesador asíncrono de metadatos para mejorar el rendimiento del RAG.
    
    Características:
    - Cola asíncrona para procesar logging y métricas en segundo plano
    - Procesamiento por lotes para reducir I/O
    - Priorización de eventos críticos
    - Buffer en memoria para eventos de alta frecuencia
    - Escritura asíncrona sin bloquear el hilo principal
    """
    
    def __init__(
        self,
        max_queue_size: int = 10000,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        enable_file_logging: bool = True,
        log_directory: str = "logs",
        metrics_directory: str = "logs/metrics"
    ):
        """
        Inicializar el procesador asíncrono.
        
        Args:
            max_queue_size: Tamaño máximo de la cola
            batch_size: Tamaño del lote para procesamiento
            flush_interval: Intervalo en segundos para procesar la cola
            enable_file_logging: Si habilitar logging a archivos
            log_directory: Directorio para logs
            metrics_directory: Directorio para métricas
        """
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.enable_file_logging = enable_file_logging
        
        # Crear directorios si no existen
        self.log_directory = Path(log_directory)
        self.metrics_directory = Path(metrics_directory)
        self.log_directory.mkdir(exist_ok=True)
        self.metrics_directory.mkdir(exist_ok=True)
        
        # Cola asíncrona para eventos
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Buffer en memoria para métricas frecuentes
        self._metrics_buffer: Dict[str, Any] = {}
        self._buffer_lock = threading.Lock()
        
        # Estado del procesador
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Estadísticas internas
        self._stats = {
            "events_processed": 0,
            "events_dropped": 0,
            "batches_processed": 0,
            "errors": 0,
            "last_flush_time": time.time()
        }
        
        # Configuración para diferentes tipos de metadatos
        self._type_configs = {
            MetadataType.LOG: {
                "batch_enabled": True,
                "file_enabled": enable_file_logging,
                "console_enabled": True
            },
            MetadataType.METRIC: {
                "batch_enabled": True,
                "file_enabled": True,
                "console_enabled": False
            },
            MetadataType.ERROR: {
                "batch_enabled": False,  # Procesar inmediatamente
                "file_enabled": True,
                "console_enabled": True
            },
            MetadataType.PERFORMANCE: {
                "batch_enabled": True,
                "file_enabled": True,
                "console_enabled": False
            },
            MetadataType.API_CALL: {
                "batch_enabled": True,
                "file_enabled": True,
                "console_enabled": False
            }
        }
    
    async def start(self) -> None:
        """Iniciar el procesador asíncrono."""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._process_events())
        loguru_logger.info("AsyncMetadataProcessor iniciado")
    
    async def stop(self) -> None:
        """Detener el procesador asíncrono."""
        if not self._running:
            return
            
        self._running = False
        
        # Procesar eventos restantes
        await self._flush_remaining_events()
        
        # Cancelar tarea
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
        loguru_logger.info("AsyncMetadataProcessor detenido")
    
    def queue_event(
        self,
        event_type: MetadataType,
        data: Dict[str, Any],
        priority: int = 1
    ) -> bool:
        """
        Agregar evento a la cola de procesamiento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            priority: Prioridad (1=normal, 2=alta, 3=crítica)
            
        Returns:
            True si se agregó correctamente, False si la cola está llena
        """
        if not self._running:
            return False
            
        event = MetadataEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            data=data,
            priority=priority
        )
        
        try:
            # Para eventos críticos, procesar inmediatamente
            if priority >= 3 or event_type == MetadataType.ERROR:
                asyncio.create_task(self._process_event_immediately(event))
                return True
            
            # Para otros eventos, agregar a la cola
            self._queue.put_nowait(event)
            return True
            
        except asyncio.QueueFull:
            self._stats["events_dropped"] += 1
            # En caso de cola llena, al menos loguear eventos críticos
            if priority >= 3:
                asyncio.create_task(self._process_event_immediately(event))
            return False
    
    def log_async(
        self,
        level: str,
        message: str,
        extra_data: Optional[Dict[str, Any]] = None,
        priority: int = 1
    ) -> bool:
        """
        Registrar log de forma asíncrona.
        
        Args:
            level: Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Mensaje de log
            extra_data: Datos adicionales
            priority: Prioridad del log
            
        Returns:
            True si se procesó correctamente
        """
        data = {
            "level": level.upper(),
            "message": message,
            "extra": extra_data or {},
            "timestamp": datetime.now().isoformat()
        }
        
        return self.queue_event(MetadataType.LOG, data, priority)
    
    def record_metric_async(
        self,
        metric_name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
        metric_type: str = "counter"
    ) -> bool:
        """
        Registrar métrica de forma asíncrona.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor de la métrica
            labels: Etiquetas adicionales
            metric_type: Tipo de métrica (counter, gauge, histogram)
            
        Returns:
            True si se procesó correctamente
        """
        data = {
            "metric_name": metric_name,
            "value": value,
            "labels": labels or {},
            "type": metric_type,
            "timestamp": datetime.now().isoformat()
        }
        
        return self.queue_event(MetadataType.METRIC, data)
    
    def record_performance_async(
        self,
        operation: str,
        duration: float,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Registrar métricas de rendimiento de forma asíncrona.
        
        Args:
            operation: Nombre de la operación
            duration: Duración en segundos
            success: Si la operación fue exitosa
            details: Detalles adicionales
            
        Returns:
            True si se procesó correctamente
        """
        data = {
            "operation": operation,
            "duration": duration,
            "success": success,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        return self.queue_event(MetadataType.PERFORMANCE, data)
    
    def record_api_call_async(
        self,
        api_name: str,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None
    ) -> bool:
        """
        Registrar llamada a API de forma asíncrona.
        
        Args:
            api_name: Nombre de la API
            endpoint: Endpoint llamado
            method: Método HTTP
            status_code: Código de estado de respuesta
            duration: Duración de la llamada
            request_size: Tamaño de la petición
            response_size: Tamaño de la respuesta
            
        Returns:
            True si se procesó correctamente
        """
        data = {
            "api_name": api_name,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration": duration,
            "request_size": request_size,
            "response_size": response_size,
            "timestamp": datetime.now().isoformat()
        }
        
        return self.queue_event(MetadataType.API_CALL, data)
    
    async def _process_events(self) -> None:
        """Bucle principal para procesar eventos."""
        batch = []
        last_flush = time.time()
        
        while self._running:
            try:
                # Esperar por eventos con timeout
                try:
                    event = await asyncio.wait_for(
                        self._queue.get(), 
                        timeout=self.flush_interval
                    )
                    batch.append(event)
                except asyncio.TimeoutError:
                    # Timeout alcanzado, procesar lote actual si existe
                    pass
                
                current_time = time.time()
                
                # Procesar lote si:
                # 1. Alcanzamos el tamaño del lote
                # 2. Ha pasado el intervalo de flush
                # 3. Hay un evento de alta prioridad
                should_flush = (
                    len(batch) >= self.batch_size or
                    (current_time - last_flush) >= self.flush_interval or
                    any(e.priority >= 2 for e in batch)
                )
                
                if should_flush and batch:
                    await self._process_batch(batch)
                    batch = []
                    last_flush = current_time
                    self._stats["last_flush_time"] = current_time
                    
            except Exception as e:
                loguru_logger.error(f"Error en el procesador de metadatos: {e}")
                self._stats["errors"] += 1
                # Pequeña pausa para evitar bucle de errores
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, events: List[MetadataEvent]) -> None:
        """
        Procesar un lote de eventos.
        
        Args:
            events: Lista de eventos a procesar
        """
        if not events:
            return
            
        try:
            # Agrupar eventos por tipo para procesamiento eficiente
            events_by_type = {}
            for event in events:
                if event.event_type not in events_by_type:
                    events_by_type[event.event_type] = []
                events_by_type[event.event_type].append(event)
            
            # Procesar cada tipo de evento
            for event_type, type_events in events_by_type.items():
                await self._process_events_by_type(event_type, type_events)
            
            self._stats["events_processed"] += len(events)
            self._stats["batches_processed"] += 1
            
        except Exception as e:
            loguru_logger.error(f"Error procesando lote de eventos: {e}")
            self._stats["errors"] += 1
    
    async def _process_events_by_type(
        self,
        event_type: MetadataType,
        events: List[MetadataEvent]
    ) -> None:
        """
        Procesar eventos de un tipo específico.
        
        Args:
            event_type: Tipo de eventos
            events: Lista de eventos del mismo tipo
        """
        config = self._type_configs.get(event_type, {})
        
        if event_type == MetadataType.LOG:
            await self._process_log_events(events, config)
        elif event_type == MetadataType.METRIC:
            await self._process_metric_events(events, config)
        elif event_type == MetadataType.ERROR:
            await self._process_error_events(events, config)
        elif event_type == MetadataType.PERFORMANCE:
            await self._process_performance_events(events, config)
        elif event_type == MetadataType.API_CALL:
            await self._process_api_call_events(events, config)
    
    async def _process_log_events(
        self,
        events: List[MetadataEvent],
        config: Dict[str, Any]
    ) -> None:
        """Procesar eventos de logging."""
        for event in events:
            data = event.data
            level = data.get("level", "INFO")
            message = data.get("message", "")
            
            # Log a consola si está habilitado
            if config.get("console_enabled", True):
                getattr(loguru_logger, level.lower())(message)
            
            # Log a archivo si está habilitado
            if config.get("file_enabled", False):
                await self._write_to_log_file(event)
    
    async def _process_metric_events(
        self,
        events: List[MetadataEvent],
        config: Dict[str, Any]
    ) -> None:
        """Procesar eventos de métricas."""
        if config.get("file_enabled", False):
            await self._write_metrics_to_file(events)
        
        # Actualizar buffer en memoria
        with self._buffer_lock:
            for event in events:
                data = event.data
                metric_name = data.get("metric_name", "unknown")
                if metric_name not in self._metrics_buffer:
                    self._metrics_buffer[metric_name] = {
                        "count": 0,
                        "total_value": 0,
                        "last_update": event.timestamp
                    }
                
                self._metrics_buffer[metric_name]["count"] += 1
                self._metrics_buffer[metric_name]["total_value"] += data.get("value", 0)
                self._metrics_buffer[metric_name]["last_update"] = event.timestamp
    
    async def _process_error_events(
        self,
        events: List[MetadataEvent],
        config: Dict[str, Any]
    ) -> None:
        """Procesar eventos de error (alta prioridad)."""
        for event in events:
            data = event.data
            
            # Log inmediato a consola
            if config.get("console_enabled", True):
                loguru_logger.error(f"Error: {data}")
            
            # Escribir a archivo de errores
            if config.get("file_enabled", True):
                await self._write_error_to_file(event)
    
    async def _process_performance_events(
        self,
        events: List[MetadataEvent],
        config: Dict[str, Any]
    ) -> None:
        """Procesar eventos de rendimiento."""
        if config.get("file_enabled", True):
            await self._write_performance_to_file(events)
    
    async def _process_api_call_events(
        self,
        events: List[MetadataEvent],
        config: Dict[str, Any]
    ) -> None:
        """Procesar eventos de llamadas API."""
        if config.get("file_enabled", True):
            await self._write_api_calls_to_file(events)
    
    async def _process_event_immediately(self, event: MetadataEvent) -> None:
        """Procesar un evento inmediatamente (para eventos críticos)."""
        try:
            await self._process_events_by_type(event.event_type, [event])
        except Exception as e:
            loguru_logger.error(f"Error procesando evento crítico: {e}")
    
    async def _write_to_log_file(self, event: MetadataEvent) -> None:
        """Escribir evento de log a archivo."""
        try:
            log_file = self.log_directory / "async_logs.jsonl"
            async with asyncio.Lock():
                with open(log_file, "a", encoding="utf-8") as f:
                    # Convertir a diccionario serializable
                    event_dict = self._make_serializable(event)
                    json.dump(event_dict, f, ensure_ascii=False)
                    f.write("\n")
        except Exception as e:
            loguru_logger.error(f"Error escribiendo log a archivo: {e}")
    
    async def _write_metrics_to_file(self, events: List[MetadataEvent]) -> None:
        """Escribir métricas a archivo."""
        try:
            metrics_file = self.metrics_directory / f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
            async with asyncio.Lock():
                with open(metrics_file, "a", encoding="utf-8") as f:
                    for event in events:
                        # Convertir a diccionario serializable
                        event_dict = self._make_serializable(event)
                        json.dump(event_dict, f, ensure_ascii=False)
                        f.write("\n")
        except Exception as e:
            loguru_logger.error(f"Error escribiendo métricas a archivo: {e}")
    
    async def _write_error_to_file(self, event: MetadataEvent) -> None:
        """Escribir error a archivo."""
        try:
            error_file = self.log_directory / "errors.jsonl"
            async with asyncio.Lock():
                with open(error_file, "a", encoding="utf-8") as f:
                    # Convertir a diccionario serializable
                    event_dict = self._make_serializable(event)
                    json.dump(event_dict, f, ensure_ascii=False)
                    f.write("\n")
        except Exception as e:
            loguru_logger.error(f"Error escribiendo error a archivo: {e}")
    
    async def _write_performance_to_file(self, events: List[MetadataEvent]) -> None:
        """Escribir métricas de rendimiento a archivo."""
        try:
            perf_file = self.metrics_directory / f"performance_{datetime.now().strftime('%Y%m%d')}.jsonl"
            async with asyncio.Lock():
                with open(perf_file, "a", encoding="utf-8") as f:
                    for event in events:
                        # Convertir a diccionario serializable
                        event_dict = self._make_serializable(event)
                        json.dump(event_dict, f, ensure_ascii=False)
                        f.write("\n")
        except Exception as e:
            loguru_logger.error(f"Error escribiendo rendimiento a archivo: {e}")
    
    async def _write_api_calls_to_file(self, events: List[MetadataEvent]) -> None:
        """Escribir llamadas API a archivo."""
        try:
            api_file = self.metrics_directory / f"api_calls_{datetime.now().strftime('%Y%m%d')}.jsonl"
            async with asyncio.Lock():
                with open(api_file, "a", encoding="utf-8") as f:
                    for event in events:
                        # Convertir a diccionario serializable
                        event_dict = self._make_serializable(event)
                        json.dump(event_dict, f, ensure_ascii=False)
                        f.write("\n")
        except Exception as e:
            loguru_logger.error(f"Error escribiendo llamadas API a archivo: {e}")
    
    async def _flush_remaining_events(self) -> None:
        """Procesar eventos restantes al cerrar."""
        remaining_events = []
        
        # Extraer todos los eventos restantes
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                remaining_events.append(event)
            except asyncio.QueueEmpty:
                break
        
        if remaining_events:
            await self._process_batch(remaining_events)
            loguru_logger.info(f"Procesados {len(remaining_events)} eventos restantes")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del procesador."""
        with self._buffer_lock:
            return {
                **self._stats,
                "queue_size": self._queue.qsize(),
                "metrics_buffer_size": len(self._metrics_buffer),
                "running": self._running
            }
    
    def get_metrics_buffer(self) -> Dict[str, Any]:
        """Obtener buffer de métricas en memoria."""
        with self._buffer_lock:
            return self._metrics_buffer.copy()
    
    def _make_serializable(self, event: MetadataEvent) -> Dict[str, Any]:
        """
        Convierte un MetadataEvent en un diccionario serializable.
        
        Args:
            event: Evento a convertir
            
        Returns:
            Diccionario serializable
        """
        event_dict = asdict(event)
        event_dict['event_type'] = event.event_type.value  # Convertir enum a string
        return event_dict


# Instancia global del procesador asíncrono
async_metadata_processor = AsyncMetadataProcessor()