# RAG Assistant Frontend

Este es el frontend para el sistema RAG Assistant. Está construido con Streamlit y puede ser desplegado como un contenedor Docker.

## Configuración

El frontend utiliza las siguientes variables de entorno:

- `API_URL`: URL del backend API (por defecto: http://localhost:8000)
- `SHOW_FULL_FRONTEND`: Si es "true", muestra todas las pestañas (Chat, Documents, Settings). Si es "false", solo muestra la pestaña de Chat (por defecto: true)
- `COLLECTION_NAME`: Nombre de la colección a usar cuando SHOW_FULL_FRONTEND es false (por defecto: valor del archivo .env)

## Requisitos

- Python 3.10+
- Docker (para despliegue con contenedor)

## Ejecución local

1. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```

2. Ejecutar la aplicación:
   ```
   streamlit run app.py --server.baseUrlPath=frontend
   ```

## Despliegue con Docker

1. Construir la imagen Docker:
   ```
   docker build -t rag-frontend .
   ```

2. Ejecutar el contenedor:
   ```
   docker run -d -p 8501:8501 -e API_URL=http://backend:8000 -e SHOW_FULL_FRONTEND=true rag-frontend
   ```

## Despliegue en GitLab CI/CD

Este proyecto incluye un archivo `.gitlab-ci.yml` para configurar un pipeline de CI/CD en GitLab que:

1. Construye la imagen Docker
2. Despliega la aplicación en el servidor configurado

### Variables de entorno necesarias en GitLab

- `CI_REGISTRY_USER`: Usuario para registro Docker de GitLab
- `CI_REGISTRY_PASSWORD`: Contraseña para registro Docker de GitLab
- `DEPLOY_SERVER`: Servidor donde desplegar
- `DEPLOY_USER`: Usuario SSH para despliegue
- `SSH_PRIVATE_KEY`: Clave SSH para conexión al servidor
- `API_URL`: URL del backend API
- `SHOW_FULL_FRONTEND`: Configuración de visualización
- `COLLECTION_NAME`: Nombre de la colección por defecto

## Acceso a la aplicación

Después del despliegue, la aplicación estará disponible en:
https://it027065.uni-graz.at/frontend

## Configuración de Nginx

En la carpeta `nginx/` se incluye un archivo de configuración para Nginx que puede ser utilizado para configurar un proxy inverso en el servidor.

### Instalación de la configuración de Nginx

```bash
sudo cp nginx/frontend.conf /etc/nginx/conf.d/
sudo nginx -t  # Probar la configuración
sudo systemctl reload nginx  # Recargar Nginx
```