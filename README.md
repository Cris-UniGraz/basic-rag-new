# Multi-Language RAG Assistant

A containerized Retrieval-Augmented Generation (RAG) system with support for German and English documents, providing a web interface for document search and conversational AI.

## Features

- **Multi-language Support**: Process and query documents in both German and English
- **Multiple Document Types**: Support for PDF, DOCX, XLSX, TXT, CSV, Markdown, and more
- **Advanced Retrieval**: Ensemble retrieval combining vector search, parent-child document retrieval, multi-query retrieval, hypothetical document embedding (HyDE), and BM25 keyword search
- **Cross-Language Queries**: Translates queries between languages to improve retrieval
- **Glossary Integration**: Domain-specific glossary for improved understanding of technical terms
- **Document Reranking**: Uses Cohere's reranking models to improve retrieval precision
- **Asynchronous Processing**: Fast response times with optimized coroutine management
- **Memory Management**: Optimized for handling large document collections
- **Caching**: Redis-based caching for improved performance
- **Monitoring**: Built-in metrics for performance monitoring
- **User-friendly Interface**: Streamlit-based UI for easy interaction

## Architecture

The system consists of:

- **Backend**: FastAPI application providing RESTful API endpoints
- **Frontend**: Streamlit application for user interaction
- **Vector Database**: Milvus for storing document embeddings
- **Document Store**: MongoDB for storing parent documents
- **Cache**: Redis for caching responses and embeddings
- **LLM**: Azure OpenAI (GPT-4) for question answering and translation

## Prerequisites

- Docker and Docker Compose
- 8GB+ RAM recommended
- Azure OpenAI API credentials
- Cohere API key

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multi-language-rag.git
cd multi-language-rag
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory based on the `.env.example` template:

```bash
cp .env.example .env
```

Edit the `.env` file to add your API keys and adjust other settings:

```
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_API_LLM_DEPLOYMENT_ID=gpt-4
AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID=text-embedding-3-large

# Cohere API
COHERE_API_KEY=your_cohere_api_key

# Vector Store Management
DONT_KEEP_COLLECTIONS=false  # Set to true if you want to recreate collections each time

# Optional: Adjust other settings as needed
```

### 3. Start the Services

```bash
docker-compose up -d
```

This will start:
- Backend API on port 8000
- Frontend on port 8501
- MongoDB on port 27017
- Milvus on port 19530
- Redis on port 6379

### 4. Access the Application

Open a web browser and navigate to:

```
http://localhost:8501
```

## Usage

### Adding Documents

1. Navigate to the "Documents" tab
2. Click "Choose files to upload" and select your documents
3. Select the document language (German or English)
4. Optionally specify a collection name
5. Click "Upload Documents"

### Chatting with the RAG System

1. Navigate to the "Chat" tab
2. Type your question in the input field and press Enter
3. View the response and the sources used to generate it

### Changing Settings

1. Navigate to the "Settings" tab
2. Change the response language as needed
3. Clear chat history or test the API connection

## Development

### Backend API Documentation

The API documentation is available at:

```
http://localhost:8000/docs
```

### Folder Structure

```
/
├── backend/                 # FastAPI backend application
│   ├── app/
│   │   ├── api/             # API endpoints
│   │   ├── core/            # Core components
│   │   ├── models/          # Database models
│   │   ├── schemas/         # Pydantic schemas
│   │   ├── services/        # Business logic
│   │   └── utils/           # Utility functions
│   └── requirements.txt     # Backend dependencies
├── frontend/                # Streamlit frontend application
│   ├── app.py               # Main Streamlit application
│   └── requirements.txt     # Frontend dependencies
├── data/                    # Data directory for documents
├── docker-compose.yml       # Docker Compose configuration
├── .env.example             # Environment variables template
└── README.md                # Project documentation
```

## Monitoring and Metrics

Prometheus metrics are available at:

```
http://localhost:8000/metrics
```

## Troubleshooting

### Common Issues

1. **Connection Errors**: Make sure all containers are running with `docker-compose ps`
2. **Memory Errors**: Increase the memory allocated to Docker in Docker Desktop settings
3. **API Key Errors**: Verify your API keys are correctly set in the `.env` file


### Logs

Check container logs with:

```bash
docker-compose logs -f backend  # Backend logs
docker-compose logs -f frontend # Frontend logs
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.