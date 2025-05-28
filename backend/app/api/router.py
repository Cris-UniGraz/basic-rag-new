from fastapi import APIRouter
from app.api.endpoints import chat, documents, metrics, async_metrics

api_router = APIRouter()

api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
api_router.include_router(async_metrics.router, prefix="/async-metrics", tags=["async-metrics"])