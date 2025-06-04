import os
import requests
import argparse
import time
import json
from pathlib import Path
from tqdm import tqdm

# PARA USAR ESTE SCRIPT (PROCESAMIENTO UNIFICADO):
# python load_documents.py --dir "C:/Pruebas/RAG Search/demo_docu_5_min/" --collection uni_test_2_0
# python load_documents.py --url http://localhost:8000 --dir "C:/Pruebas/RAG Search/documentos_idioma_all/" --collection uni_docs_1_0
# python load_documents.py --url http://143.50.27.65:8000 --dir "C:/Pruebas/RAG Search/demo_docs/" --collection uni_test_2_0
#
# NOTA: Ya no se requieren subcarpetas por idioma. Todos los documentos se procesan desde
# el directorio especificado directamente, sin importar su idioma.
#
# DEPENDENCIAS REQUERIDAS:
# pip install requests tqdm PyMuPDF docx2txt openpyxl

def get_files_recursively(directory):
    """Obtiene todos los archivos de un directorio y sus subdirectorios."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            # Filtrar archivos temporales y ocultos
            if not filename.startswith('~') and not filename.startswith('.') and not filename.startswith('$'):
                file_path = os.path.join(root, filename)
                # Solo incluir archivos (no directorios)
                if os.path.isfile(file_path):
                    files.append(file_path)
    return files

def upload_documents(base_url, directory_path, collection_name):
    """Carga documentos al servicio RAG desde un directorio unificado.
    
    PROCESAMIENTO UNIFICADO: Ya no se diferencia por idioma.
    Todos los documentos se procesan con el mismo embedding model y van a la misma colección.
    """
    # Verificar que el directorio exista
    if not os.path.exists(directory_path):
        print(f"Error: El directorio {directory_path} no existe.")
        return False

    print(f"\n📄 Utilidad de carga masiva de documentos para RAG (PROCESAMIENTO UNIFICADO)")
    print(f"============================================================================")
    print(f"URL Base: {base_url}")
    print(f"Directorio: {directory_path}")
    print(f"Nombre de colección: {collection_name}")
    print(f"Modo: Procesamiento unificado multiidioma sin clasificación")

    # Procesar todos los documentos del directorio unificado
    success = process_directory(base_url, directory_path, collection_name)
        
    # En cualquier caso, consideramos que el proceso fue exitoso mientras estemos cargando archivos

    if success:
        print(f"\n✅ Procesamiento de todos los documentos completado con éxito!")
    else:
        print(f"\n⚠️ Procesamiento completado con advertencias. Los documentos se cargaron, pero puede haber algún problema de configuración en la colección.")
    
    return success


def process_directory(base_url, directory, collection_name):
    """Procesa documentos de un directorio unificado.
    
    PROCESAMIENTO UNIFICADO: 
    - Lee todos los archivos del directorio y subdirectorios
    - No busca subcarpetas por idioma (/de, /en)
    - Usa un solo embedding model (Azure OpenAI) para todos los documentos
    - Todos los documentos van a la misma colección unificada
    """
    # Obtener todos los archivos del directorio y sus subdirectorios
    files = get_files_recursively(directory)

    if not files:
        print(f"No se encontraron archivos en {directory}.")
        return True

    print(f"\nProcesando {len(files)} documentos desde {directory}...")

    # Mostrar barra de progreso para el procesamiento total
    file_progress = tqdm(total=len(files), desc=f"Progreso general", unit="archivo")
    
    # Variables para seguimiento de éxitos y errores
    successful_files = 0
    failed_files = []

    # Procesar cada archivo individualmente
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        
        try:
            # Ya no necesitamos calcular metadatos específicos
            # El sistema basic-rag-new los extrae automáticamente
            print(f"\nProcesando archivo: {file_name}")
            
            # Preparar archivo para subir
            files_to_upload = [
                ('files', (file_name, open(file_path, 'rb'), 'application/octet-stream'))
            ]
            
            # PROCESAMIENTO UNIFICADO: Compatible con basic-rag-new
            # El endpoint /api/documents/upload solo requiere collection_name
            # Los metadatos se extraen automáticamente durante el procesamiento
            
            upload_data = {
                'collection_name': collection_name
            }
            
            print(f"Subiendo a colección: {collection_name}")
            
            # Subir documento
            try:
                # Ajustar timeout basado en tamaño del archivo
                file_size = os.path.getsize(file_path)
                # Si el archivo es grande (>5MB), dar más tiempo para carga
                upload_timeout = 120 if file_size > 5*1024*1024 else 60
                
                response = requests.post(
                    f"{base_url}/api/documents/upload",
                    files=files_to_upload,
                    data=upload_data,
                    timeout=upload_timeout
                )

                response.raise_for_status()
                result = response.json()

                # Monitorear progreso de carga
                task_id = result.get('task_id')
                if task_id:
                    success = monitor_upload_progress(base_url, task_id)
                    if success:
                        successful_files += 1
                    else:
                        failed_files.append((file_name, "Fallo en el procesamiento"))

            except requests.exceptions.RequestException as e:
                print(f"\n⚠️ Error al subir documento {file_name}: {e}")
                failed_files.append((file_name, str(e)))

            # Cerrar handlers de archivos
            for file_entry in files_to_upload:
                _, file_obj = file_entry
                file_obj[1].close()

        except Exception as e:
            print(f"\n⚠️ Error procesando archivo {file_name}: {e}")
            failed_files.append((file_name, str(e)))
        
        # Actualizar progreso
        file_progress.update(1)

    file_progress.close()
    
    # Resumen final
    print(f"\n📊 Resumen de procesamiento:")
    print(f"  ✅ Archivos procesados correctamente: {successful_files}/{len(files)}")
    
    if failed_files:
        print(f"  ⚠️ Archivos con advertencias: {len(failed_files)}")
        for name, error in failed_files[:5]:  # Mostrar los primeros 5 errores
            print(f"     - {name}: {error}")
        
        if len(failed_files) > 5:
            print(f"     ... y {len(failed_files) - 5} más")
            
    file_progress.close()
    return successful_files == len(files)


def monitor_upload_progress(base_url, task_id):
    """
    Monitorea el progreso de un proceso de carga de documentos.
    
    Returns:
        bool: True si la carga se completó con éxito, False en caso contrario
    """
    progress_bar = tqdm(total=100, desc="Procesando", unit="%", leave=False)
    last_percentage = 0
    success = False

    while True:
        try:
            # Aumentar el timeout para archivos grandes
            response = requests.get(f"{base_url}/api/documents/progress/{task_id}", timeout=120)
            response.raise_for_status()

            data = response.json()
            status = data.get('status')
            percentage = data.get('percentage', 0)

            # Actualizar barra de progreso
            progress_diff = percentage - last_percentage
            if progress_diff > 0:
                progress_bar.update(progress_diff)
                last_percentage = percentage

            if status == "completed":
                progress_bar.update(100 - last_percentage)  # Asegurar que lleguemos al 100%
                progress_bar.close()
                print(f"✅ Carga completada: {data.get('message', 'Éxito')}")
                success = True
                break
                
            elif status == "error":
                progress_bar.close()
                # Analizar el mensaje de error para ser más descriptivo
                error_message = data.get('message', 'Error desconocido')
                
                # Lista de campos que podrían faltar según los errores comunes
                missing_fields = ["sheet_index", "width", "height", "total_pages", "total_sheets", "sheet_name", "page_number", "source", "file_type"]
                
                # Verificar si es un error de campo faltante en la colección
                if "DataNotMatchException" in error_message:
                    missing_field_found = False
                    for field in missing_fields:
                        if f"missed an field `{field}`" in error_message:
                            print(f"⚠️ Error de campo: Milvus requiere el campo '{field}'")
                            missing_field_found = True
                            break
                    
                    if not missing_field_found:
                        print(f"❌ Error en la carga: {error_message}")
                else:
                    print(f"❌ Error en la carga: {error_message}")
                break
                
            # Verificar timeout - evitar esperar indefinidamente
            elif progress_bar.n == last_percentage and progress_bar.n > 0:
                if progress_bar._time_elapsed() > 120:  # 2 minutos sin cambios
                    progress_bar.close()
                    print("⚠️ Tiempo de espera excedido sin progreso")
                    break

            time.sleep(2)  # Consultar cada 2 segundos

        except requests.exceptions.RequestException as e:
            progress_bar.close()
            print(f"⚠️ Error al monitorear el progreso: {e}")
            break
            
    return success


# Las funciones de cálculo de metadatos ya no son necesarias
# El sistema basic-rag-new extrae automáticamente todos los metadatos
        
# Funciones auxiliares de metadatos eliminadas
# El sistema basic-rag-new maneja automáticamente todos los metadatos


def verify_unified_compatibility(base_url):
    """
    Verifica que el API esté accesible para procesamiento unificado.
    """
    print("\n🔍 Verificando acceso al API...\n")
    
    try:
        # Verificar que el API esté accesible
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API accesible")
            return True
        else:
            print(f"⚠️ API no accesible. Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando API: {e}")
        return False

def check_api_schema(base_url):
    """
    Realiza una verificación directa al API para obtener información sobre el schema.
    Esta función es útil para depurar problemas con los campos de metadatos.
    """
    print("\n🔍 Verificando API y esquema de metadatos...\n")
    
    # 1. Verificar conexión básica al API
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print("✅ Conexión al API verificada (documentación disponible)")
        else:
            print(f"⚠️ La documentación del API no está disponible. Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error de conexión a {base_url}/docs: {e}")
        
    # 2. Verificar si hay colecciones existentes
    try:
        response = requests.get(f"{base_url}/api/documents/collections", timeout=10)
        if response.status_code == 200:
            collections = response.json()
            print(f"✅ API de colecciones accesible. Colecciones encontradas: {len(collections)}")
            
            # Mostrar algunas colecciones para referencia
            if collections:
                print("📚 Primeras 3 colecciones en el sistema:")
                for idx, coll in enumerate(collections[:3]):
                    print(f"   {idx+1}. {coll.get('name', 'Sin nombre')} - Documentos: {coll.get('count', 'N/A')}")
        else:
            print(f"⚠️ No se pudo acceder a la lista de colecciones. Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error accediendo a colecciones: {e}")
        
    print("\n🔍 Verificación de API y esquema finalizada\n")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Carga masiva de documentos para RAG compatible con basic-rag-new.')
    parser.add_argument('--url', default='http://localhost:8000', help='URL base de la API de RAG')
    parser.add_argument('--dir', help='Directorio que contiene todos los documentos (lee recursivamente)')
    parser.add_argument('--collection', default='documents', help='Nombre de la colección')
    parser.add_argument('--test', action='store_true', help='Modo de prueba - agrega un sufijo de timestamp a la colección')
    parser.add_argument('--check-schema', action='store_true', help='Verificar esquema API sin cargar documentos')
    parser.add_argument('--verify-only', action='store_true', help='Solo verificar acceso al API')

    args = parser.parse_args()
    
    # Si está activada la opción de verificar schema, solo hacemos eso
    if args.check_schema:
        check_api_schema(args.url)
        return
    
    # Si solo queremos verificar acceso
    if args.verify_only:
        print("\n🔍 MODO VERIFICACIÓN SOLAMENTE")
        if verify_unified_compatibility(args.url):
            print("✅ El API está accesible.")
        else:
            print("❌ No se puede acceder al API.")
        return
    
    # Verificar acceso al API antes de proceder
    print("\n🔧 Verificando acceso al API...")
    if not verify_unified_compatibility(args.url):
        print("❌ No se puede acceder al API.")
        print("Asegúrate de que el sistema basic-rag-new esté ejecutándose.")
        
        # Preguntar si quiere continuar anyway
        continue_anyway = input("\n¿Quieres continuar de todos modos? (y/N): ")
        if continue_anyway.lower() not in ['y', 'yes', 's', 'si']:
            print("❌ Operación cancelada.")
            return
    else:
        print("✅ API accesible.")
    
    # Para cargar documentos, el directorio es obligatorio (excepto en modos de verificación)
    if not args.dir and not args.verify_only:
        parser.error("El argumento --dir es obligatorio para cargar documentos")
    
    # Verificar que el directorio existe (solo si se especificó)
    if args.dir and not os.path.exists(args.dir):
        print(f"❌ Error: El directorio '{args.dir}' no existe.")
        return
    
    # Informar sobre el procesamiento
    print(f"\n🚀 PROCESAMIENTO UNIFICADO")
    print(f"Todos los archivos en '{args.dir}' serán procesados:")
    print(f"  - Directorio: {args.dir}")
    print(f"  - Colección: {args.collection}")
    print(f"  - Sin subcarpetas por idioma requeridas")
    print(f"  - Metadatos extraídos automáticamente")
        
    # Si estamos en modo de prueba, agregar el timestamp a la colección
    collection_name = args.collection
    if args.test:
        import time
        timestamp = int(time.time())
        collection_name = f"{args.collection}_test_{timestamp}"
        print(f"Modo de prueba activado. Usando colección: {collection_name}")

    upload_documents(args.url, args.dir, collection_name)

if __name__ == "__main__":
    main()