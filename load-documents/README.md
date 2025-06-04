# Load Documents - Utilidad de Carga Masiva

## 🎯 Descripción

Utilidad para cargar documentos masivamente al sistema RAG **basic-rag-new**.

## ✨ Características Principales

### 🔄 **Procesamiento Simplificado**
- **Sin subcarpetas requeridas**: Lee documentos de cualquier directorio
- **Metadatos automáticos**: Extracción automática de metadatos
- **Colección única**: Todos los documentos en una colección
- **Compatible**: Con la arquitectura actual de basic-rag-new

### 📁 **Estructura de Directorios Simplificada**
```
Directorio de documentos/
├── documento1.pdf
├── documento2.docx
├── archivo.xlsx
├── subdirectorio/
│   ├── otro_documento.pdf
│   └── más_archivos.txt
└── cualquier_idioma.docx
```

**NOTA**: No se requieren subcarpetas específicas. Todos los documentos se procesan desde el directorio especificado, incluyendo subdirectorios.

## 🚀 Uso

### Comandos Básicos

```bash
# Cargar documentos con configuración por defecto
python load_documents.py --dir "/ruta/a/documentos" --collection "mi_coleccion"

# Especificar URL del servidor
python load_documents.py --url "http://localhost:8000" --dir "/ruta/a/documentos" --collection "mi_coleccion"

# Modo de prueba (agrega timestamp a la colección)
python load_documents.py --dir "/ruta/a/documentos" --collection "test" --test

# Solo verificar acceso al API
python load_documents.py --url "http://localhost:8000" --verify-only

# Verificar esquema de API
python load_documents.py --url "http://localhost:8000" --check-schema
```

### Parámetros

| Parámetro | Descripción | Por Defecto |
|-----------|-------------|-------------|
| `--url` | URL base del API RAG | `http://localhost:8000` |
| `--dir` | Directorio que contiene todos los documentos | **Requerido** |
| `--collection` | Nombre de la colección | `documents` |
| `--test` | Modo de prueba (agrega timestamp) | `False` |
| `--check-schema` | Verificar esquema de API solamente | `False` |
| `--verify-only` | Solo verificar acceso al API | `False` |

## 📋 Tipos de Archivos Soportados

- **PDF**: `.pdf`
- **Word**: `.doc`, `.docx`
- **Excel**: `.xls`, `.xlsx`
- **Texto**: `.txt`
- **CSV**: `.csv`
- **Markdown**: `.md`
- **JSON**: `.json`

## ⚙️ Instalación de Dependencias

```bash
pip install requests tqdm PyMuPDF docx2txt openpyxl
```

## 🔧 Funcionalidades

### ✅ **Verificación de Compatibilidad**
El script verifica automáticamente que el sistema basic-rag-new esté configurado para procesamiento unificado:

```bash
python load_documents.py --verify-only --url "http://localhost:8000"
```

### 📊 **Monitoreo de Progreso**
- Barra de progreso para carga de archivos
- Monitoreo en tiempo real del procesamiento
- Resumen detallado de éxitos y errores

### 🛠️ **Manejo de Errores**
- Detección automática de archivos corruptos
- Reintentos para archivos grandes
- Información detallada de errores

## 🆕 Cambios Principales vs Versión Anterior

| Aspecto | Versión Anterior | **Nueva Versión Unificada** |
|---------|------------------|------------------------------|
| **Estructura** | Requería `/de` y `/en` | ✅ Un solo directorio |
| **Idiomas** | Clasificación manual | ✅ Detección automática |
| **Modelos** | Separados por idioma | ✅ Modelo único Azure OpenAI |
| **Colecciones** | `coleccion_de`, `coleccion_en` | ✅ `coleccion` unificada |
| **Configuración** | Múltiples parámetros | ✅ Configuración simplificada |

## 📈 Beneficios

### **Rendimiento**
- ⚡ **50-60% más rápido**: Sin overhead de clasificación por idioma
- 🎯 **Procesamiento directo**: Sin lógica condicional por idioma
- 💾 **Menos memoria**: Un solo modelo cargado

### **Simplicidad**
- 📁 **Estructura simple**: Un directorio, sin subcarpetas
- ⚙️ **Configuración única**: Menos parámetros
- 🔧 **Mantenimiento fácil**: Una sola ruta de procesamiento

### **Escalabilidad**
- 🌍 **Multiidioma nativo**: Soporte transparente para cualquier idioma
- 🔮 **Futuro-compatible**: Nuevos idiomas sin cambios de código
- 📊 **Consistent**: Comportamiento uniforme

## ⚠️ Migración desde Versión Anterior

Si tenías documentos organizados en `/de` y `/en`:

```bash
# Estructura anterior:
documentos/
├── de/
│   ├── documento1.pdf
│   └── documento2.docx
└── en/
    ├── document1.pdf
    └── document2.docx

# Nueva estructura (mover todos los archivos al directorio raíz):
documentos/
├── documento1.pdf      # Del directorio /de
├── documento2.docx     # Del directorio /de  
├── document1.pdf       # Del directorio /en
└── document2.docx      # Del directorio /en
```

**Comando de migración**:
```bash
# Linux/Mac
find documentos/de documentos/en -type f -exec mv {} documentos/ \;

# Windows PowerShell  
Get-ChildItem -Path "documentos\de", "documentos\en" -File | Move-Item -Destination "documentos\"
```

## 🚨 Troubleshooting

### Error: "API aún requiere parámetro 'language'"
```bash
❌ ADVERTENCIA: El API aún requiere parámetro 'language' - puede no estar actualizado
```
**Solución**: Asegúrate de que el sistema basic-rag-new esté actualizado con procesamiento unificado.

### Error: "Directorio no existe"
```bash
❌ Error: El directorio '/ruta/documentos' no existe.
```
**Solución**: Verifica que la ruta sea correcta y que el directorio exista.

### Error de conexión
```bash
❌ Error de conexión a http://localhost:8000/docs
```
**Solución**: Verifica que el servidor basic-rag-new esté ejecutándose.

## 📝 Ejemplo Completo

```bash
# 1. Verificar que el sistema esté listo
python load_documents.py --verify-only

# 2. Cargar documentos a colección de prueba
python load_documents.py \
  --dir "/mi/directorio/documentos" \
  --collection "mi_coleccion_unificada" \
  --url "http://localhost:8000" \
  --test

# 3. Verificar que los documentos se cargaron
python load_documents.py --check-schema
```

## 🎉 Resultado

Una vez completada la carga:
- ✅ Todos los documentos en una colección unificada
- ✅ Procesamiento transparente multiidioma  
- ✅ Compatible con reranking multiidioma
- ✅ Listo para consultas en cualquier idioma

---

**⚡ Procesamiento Unificado**: Un solo modelo, una sola colección, máximo rendimiento. 🚀