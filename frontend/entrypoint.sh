#!/bin/bash

# Asegurarse de que el directorio .streamlit existe
mkdir -p /app/.streamlit

# Crear/actualizar archivo de configuración
cat > /app/.streamlit/config.toml << EOL
[server]
baseUrlPath = "frontend"
headless = true
enableCORS = false
enableXsrfProtection = true

[theme]
base = "dark"
primaryColor = "#4B8BF4"
backgroundColor = "#1E1E1E"
secondaryBackgroundColor = "#252525"
textColor = "#FAFAFA"
font = "sans serif"
EOL

# Ejecutar Streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.baseUrlPath=frontend