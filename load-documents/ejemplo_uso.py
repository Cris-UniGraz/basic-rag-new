#!/usr/bin/env python3
"""
Ejemplo de uso del load_documents.py con procesamiento unificado.

Este script demuestra cómo usar la utilidad de carga masiva
con la nueva arquitectura unificada de basic-rag-new.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def crear_documentos_ejemplo():
    """Crea documentos de ejemplo para demostrar la carga."""
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp(prefix="rag_docs_ejemplo_")
    print(f"📁 Creando documentos de ejemplo en: {temp_dir}")
    
    # Documento en alemán
    doc_aleman = Path(temp_dir) / "documento_aleman.txt"
    doc_aleman.write_text("""
Willkommen an der Universität Graz

Die Universität Graz ist eine der ältesten Universitäten im deutschen Sprachraum.
Sie wurde 1585 gegründet und hat eine lange Tradition in Forschung und Lehre.

Studieninformationen:
- Die Einschreibung erfolgt online über das UNIGRAzcard System
- Die Bibliothek ist über unikat zugänglich
- Für Fragen steht das PSP-HR Team zur Verfügung

Kontakt:
E-Mail: info@uni-graz.at
Telefon: +43 316 380-0
""", encoding="utf-8")
    
    # Documento en inglés
    doc_ingles = Path(temp_dir) / "english_document.txt"
    doc_ingles.write_text("""
Welcome to the University of Graz

The University of Graz is one of the oldest universities in the German-speaking world.
It was founded in 1585 and has a long tradition in research and teaching.

Study Information:
- Enrollment is done online through the UNIGRAzcard system
- The library is accessible via unikat
- For questions, the PSP-HR team is available

Contact:
Email: info@uni-graz.at
Phone: +43 316 380-0
""", encoding="utf-8")
    
    # Documento técnico
    doc_tecnico = Path(temp_dir) / "technical_info.txt"
    doc_tecnico.write_text("""
Sistema RAG - Información Técnica

Configuración Unificada:
- Modelo de embeddings: Azure OpenAI text-embedding-ada-002
- Reranking: Cohere rerank-multilingual-v3.0
- Vector Store: Milvus con colección unificada
- Cache: Redis sin diferenciación por idioma

Beneficios del Procesamiento Unificado:
1. Mayor eficiencia (50-60% más rápido)
2. Arquitectura simplificada
3. Soporte multiidioma transparente
4. Menor complejidad de configuración
5. Escalabilidad mejorada
""", encoding="utf-8")
    
    # Documento multiidioma
    doc_multi = Path(temp_dir) / "multilingual_example.txt"
    doc_multi.write_text("""
Ejemplo Multiidioma / Multilingual Example

Deutsch:
Das neue System unterstützt mehrere Sprachen gleichzeitig.
Es gibt keine Notwendigkeit mehr, Dokumente nach Sprache zu trennen.

English:
The new system supports multiple languages simultaneously.
There is no longer a need to separate documents by language.

Español:
El nuevo sistema soporta múltiples idiomas simultáneamente.
Ya no es necesario separar documentos por idioma.

Français:
Le nouveau système prend en charge plusieurs langues simultanément.
Il n'est plus nécessaire de séparer les documents par langue.
""", encoding="utf-8")
    
    print(f"✅ Creados 4 documentos de ejemplo")
    return temp_dir

def ejecutar_ejemplo(temp_dir, url_api="http://localhost:8000"):
    """Ejecuta el ejemplo de carga de documentos."""
    
    script_path = Path(__file__).parent / "load_documents.py"
    collection_name = "ejemplo_unificado"
    
    print(f"\n🚀 EJECUTANDO EJEMPLO DE CARGA UNIFICADA")
    print(f"=" * 50)
    
    # 1. Verificar compatibilidad
    print(f"\n1️⃣ Verificando compatibilidad del sistema...")
    cmd_verify = [
        sys.executable, str(script_path),
        "--url", url_api,
        "--verify-only"
    ]
    
    try:
        result = subprocess.run(cmd_verify, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Sistema compatible")
        else:
            print("⚠️ Advertencia en verificación:")
            print(result.stdout)
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ Timeout en verificación - continuando...")
    except Exception as e:
        print(f"❌ Error en verificación: {e}")
        return False
    
    # 2. Cargar documentos
    print(f"\n2️⃣ Cargando documentos desde: {temp_dir}")
    cmd_upload = [
        sys.executable, str(script_path),
        "--url", url_api,
        "--dir", temp_dir,
        "--collection", collection_name,
        "--test"  # Usar modo de prueba
    ]
    
    print(f"Comando: {' '.join(cmd_upload)}")
    
    try:
        result = subprocess.run(cmd_upload, timeout=300)  # 5 minutos máximo
        if result.returncode == 0:
            print("\n✅ ¡Carga completada exitosamente!")
            return True
        else:
            print(f"\n❌ Error en la carga (código: {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print("\n⏰ La carga está tomando mucho tiempo - puede continuar en segundo plano")
        return True
    except Exception as e:
        print(f"\n❌ Error ejecutando carga: {e}")
        return False

def limpiar_directorio(temp_dir):
    """Limpia el directorio temporal."""
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"🧹 Directorio temporal limpiado: {temp_dir}")
    except Exception as e:
        print(f"⚠️ No se pudo limpiar directorio temporal: {e}")

def main():
    """Función principal del ejemplo."""
    
    print("🎯 EJEMPLO DE CARGA MASIVA - PROCESAMIENTO UNIFICADO")
    print("=" * 60)
    print()
    print("Este ejemplo demuestra:")
    print("✨ Carga de documentos multiidioma sin clasificación")
    print("✨ Procesamiento unificado con modelo único")  
    print("✨ Verificación automática de compatibilidad")
    print("✨ Monitoreo de progreso en tiempo real")
    print()
    
    # Permitir especificar URL de API
    if len(sys.argv) > 1:
        url_api = sys.argv[1]
        print(f"🌐 Usando URL de API: {url_api}")
    else:
        url_api = "http://localhost:8000"
        print(f"🌐 Usando URL de API por defecto: {url_api}")
    
    temp_dir = None
    
    try:
        # Crear documentos de ejemplo
        temp_dir = crear_documentos_ejemplo()
        
        # Ejecutar ejemplo
        success = ejecutar_ejemplo(temp_dir, url_api)
        
        if success:
            print("\n🎉 ¡EJEMPLO COMPLETADO EXITOSAMENTE!")
            print("\nPróximos pasos:")
            print("1. Los documentos están ahora en una colección unificada")
            print("2. Puedes hacer consultas en cualquier idioma")
            print("3. El sistema usará reranking multiidioma automáticamente")
            print("4. No necesitas especificar idioma en las consultas")
        else:
            print("\n⚠️ El ejemplo tuvo algunos problemas")
            print("Verifica que el servidor basic-rag-new esté ejecutándose")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Ejemplo interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
    finally:
        # Limpiar directorio temporal
        if temp_dir:
            limpiar_directorio(temp_dir)

if __name__ == "__main__":
    main()