#!/bin/bash
# Script para desplegar el frontend en un servidor

# Variables
DOCKER_IMAGE="frontend:latest"
CONTAINER_NAME="frontend_container"
API_URL=${API_URL:-"http://backend:8000"}
SHOW_FULL_FRONTEND=${SHOW_FULL_FRONTEND:-"true"}
COLLECTION_NAME=${COLLECTION_NAME:-""}

# Construir la imagen de Docker
echo "Construyendo la imagen Docker..."
docker build -t $DOCKER_IMAGE .

# Detener y eliminar el contenedor anterior si existe
echo "Deteniendo el contenedor anterior si existe..."
docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true

# Ejecutar el nuevo contenedor
echo "Iniciando el nuevo contenedor..."
docker run -d \
  --name $CONTAINER_NAME \
  -p 8501:8501 \
  -e API_URL=$API_URL \
  -e SHOW_FULL_FRONTEND=$SHOW_FULL_FRONTEND \
  -e COLLECTION_NAME=$COLLECTION_NAME \
  --restart always \
  $DOCKER_IMAGE

echo "Despliegue completado."
echo "El frontend está disponible en: https://it027065.uni-graz.at/frontend"