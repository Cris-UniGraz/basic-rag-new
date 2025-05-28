from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.core.async_metadata_processor import async_metadata_processor
from app.core.query_optimizer import QueryOptimizer

router = APIRouter()


@router.get("/async-metrics/stats", summary="Get async metadata processor statistics")
async def get_async_stats() -> Dict[str, Any]:
    """
    Obtener estadísticas del procesador asíncrono de metadatos.
    
    Incluye información sobre:
    - Eventos procesados
    - Eventos descartados
    - Tamaño de cola
    - Errores
    - Estado del procesador
    """
    try:
        stats = async_metadata_processor.get_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")


@router.get("/async-metrics/buffer", summary="Get metrics buffer")
async def get_metrics_buffer() -> Dict[str, Any]:
    """
    Obtener el buffer de métricas en memoria.
    
    Incluye métricas agregadas que están siendo procesadas en segundo plano.
    """
    try:
        buffer = async_metadata_processor.get_metrics_buffer()
        return {
            "status": "success",
            "data": buffer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo buffer: {str(e)}")


@router.post("/async-metrics/flush", summary="Force flush pending events")
async def force_flush() -> Dict[str, str]:
    """
    Forzar el procesamiento inmediato de eventos pendientes.
    
    Útil para depuración o antes de un cierre del sistema.
    """
    try:
        # Forzar el procesamiento de eventos pendientes
        await async_metadata_processor._flush_remaining_events()
        return {
            "status": "success",
            "message": "Eventos pendientes procesados exitosamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error forzando flush: {str(e)}")


@router.get("/async-metrics/health", summary="Check async processor health")
async def check_health() -> Dict[str, Any]:
    """
    Verificar el estado de salud del procesador asíncrono.
    
    Incluye información sobre si el procesador está ejecutándose correctamente.
    """
    try:
        stats = async_metadata_processor.get_stats()
        
        # Determinar el estado de salud
        is_healthy = (
            stats.get("running", False) and
            stats.get("errors", 0) < 100 and  # Menos de 100 errores
            stats.get("queue_size", 0) < 9000  # Cola no saturada
        )
        
        health_status = "healthy" if is_healthy else "degraded"
        
        return {
            "status": health_status,
            "running": stats.get("running", False),
            "queue_size": stats.get("queue_size", 0),
            "error_count": stats.get("errors", 0),
            "events_processed": stats.get("events_processed", 0),
            "uptime_info": {
                "last_flush_time": stats.get("last_flush_time", 0),
                "batches_processed": stats.get("batches_processed", 0)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/async-metrics/clean-error-cache", summary="Clean error responses from cache")
async def clean_error_cache() -> Dict[str, Any]:
    """
    Limpiar respuestas de error del caché de consultas.
    
    Útil cuando se han almacenado respuestas de error que necesitan ser eliminadas.
    """
    try:
        query_optimizer = QueryOptimizer()
        removed_count = query_optimizer.clean_error_responses_from_cache()
        
        return {
            "status": "success",
            "message": f"Limpieza completada: eliminadas {removed_count} respuestas de error del caché",
            "removed_count": removed_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error limpiando caché: {str(e)}")