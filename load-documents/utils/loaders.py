# Este archivo mantiene compatibilidad con importaciones anteriores
# pero ahora es simplificado para trabajar con basic-rag-new

import os
from pathlib import Path

def get_files_recursively(directory):
    """Obtiene todos los archivos de un directorio y sus subdirectorios."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if not filename.startswith('~') and not filename.startswith('.'):
                file_path = os.path.join(root, filename)
                files.append(file_path)
    return files


# Funciones mantenidas por compatibilidad con código anterior
# El procesamiento real ahora lo hace basic-rag-new automáticamente

# Tipos de archivo para compatibilidad
class FileType:
    PDF = "pdf"
    WORD = "doc"
    WORDX = "docx"
    EXCEL = "xls"
    EXCELX = "xlsx"
    WEBPAGE = "html"
