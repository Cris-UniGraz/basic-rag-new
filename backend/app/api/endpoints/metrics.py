from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Dict, Any, Optional
import os
import time
from loguru import logger
import numpy as np

from app.core.metrics_manager import MetricsManager

router = APIRouter()
metrics_manager = MetricsManager()


@router.get("/summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Obtiene un resumen de las métricas actuales.
    
    Returns:
        Diccionario con métricas resumidas del sistema
    """
    return metrics_manager.get_summary()


@router.get("/detailed")
async def get_detailed_metrics() -> Dict[str, Any]:
    """
    Obtiene estadísticas detalladas para análisis.
    
    Returns:
        Diccionario con estadísticas detalladas y distribuciones
    """
    return metrics_manager.get_detailed_statistics()


@router.get("/period")
async def start_new_period() -> Dict[str, Any]:
    """
    Inicia un nuevo período de medición y obtiene métricas hasta el momento.
    
    Returns:
        Diccionario con métricas del período anterior
    """
    # Obtener métricas del período anterior si existe
    if hasattr(metrics_manager, 'period_start_time'):
        period_metrics = metrics_manager.get_period_metrics()
    else:
        period_metrics = {}
    
    # Iniciar un nuevo período
    metrics_manager.start_period()
    
    return {
        "previous_period": period_metrics,
        "new_period_started_at": metrics_manager.period_start_time.isoformat()
    }


@router.post("/export")
async def export_metrics(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Guarda las métricas actuales en archivos para análisis posterior.
    Esta operación se ejecuta en segundo plano.
    
    Returns:
        Mensaje de confirmación
    """
    def export_task():
        try:
            # Crear timestamp para nombres de archivo
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Guardar métricas en JSON
            json_path = metrics_manager.save_metrics(f"metrics_{timestamp}.json")
            
            # Exportar a CSV
            csv_dir = os.path.join(metrics_manager._metrics_dir, f"export_{timestamp}")
            csv_files = metrics_manager.export_to_csv(csv_dir)
            
            logger.info(f"Métricas exportadas: JSON={json_path}, CSV dir={csv_dir}")
        except Exception as e:
            logger.error(f"Error exportando métricas: {e}")
    
    # Ejecutar la exportación en segundo plano
    background_tasks.add_task(export_task)
    
    return {"message": "Exportación de métricas iniciada en segundo plano"}


@router.post("/reset")
async def reset_metrics() -> Dict[str, Any]:
    """
    Reinicia todas las métricas.
    
    Returns:
        Mensaje de confirmación
    """
    # Guardar un respaldo antes de resetear
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_before_reset_{timestamp}.json"
        backup_path = metrics_manager.save_metrics(filename)
    except Exception as e:
        logger.warning(f"No se pudo guardar respaldo antes de resetear métricas: {e}")
        backup_path = None
    
    # Resetear métricas
    metrics_manager.reset_metrics()
    
    return {
        "message": "Métricas reiniciadas correctamente",
        "backup_file": backup_path
    }


@router.get("/configure")
async def configure_metrics(
    max_history: Optional[int] = None,
    auto_save: Optional[bool] = None,
    auto_save_interval: Optional[int] = None,
    metrics_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Configura el gestor de métricas.
    
    Args:
        max_history: Número máximo de elementos en el historial
        auto_save: Si debe guardar automáticamente las métricas
        auto_save_interval: Intervalo en segundos para guardado automático
        metrics_dir: Directorio donde guardar las métricas
        
    Returns:
        Configuración actualizada
    """
    # Aplicar configuración
    metrics_manager.configure(
        max_history=max_history,
        auto_save=auto_save,
        auto_save_interval=auto_save_interval,
        metrics_dir=metrics_dir
    )
    
    # Devolver configuración actual
    return {
        "max_history_items": metrics_manager._max_history_items,
        "auto_save": metrics_manager._auto_save,
        "auto_save_interval": metrics_manager._auto_save_interval,
        "metrics_dir": metrics_manager._metrics_dir
    }