FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    STREAMLIT_THEME=dark

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt debugpy

# Ensure .streamlit directory exists
RUN mkdir -p /app/.streamlit

# Create Streamlit configuration for dark theme
RUN echo '[theme]\nbase = "dark"\nprimaryColor = "#4B8BF4"\nbackgroundColor = "#1E1E1E"\nsecondaryBackgroundColor = "#252525"\ntextColor = "#FAFAFA"\nfont = "sans serif"' > /app/.streamlit/config.toml
RUN echo '[server]\nbaseUrlPath = "frontend"\nheadless = true\nenableCORS = false\nenableXsrfProtection = true' >> /app/.streamlit/config.toml

# Copy application code
COPY . .

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose ports
EXPOSE 8501
EXPOSE 5679

# Command to run the application
CMD ["/app/entrypoint.sh"]