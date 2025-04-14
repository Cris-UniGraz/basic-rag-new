from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


class DocumentMetadata(BaseModel):
    """
    Metadata about a document.
    """
    source: Optional[str] = Field(default="unknown", description="Source of the document (filename or URL)")
    file_type: Optional[str] = Field(default="text", description="Type of the file (pdf, docx, etc.)")
    page_number: Optional[int] = Field(None, description="Page number within document")
    total_pages: Optional[int] = Field(None, description="Total pages in document")
    sheet_name: Optional[str] = Field(None, description="Sheet name for spreadsheets")
    sheet_index: Optional[int] = Field(None, description="Sheet index for spreadsheets")
    total_sheets: Optional[int] = Field(None, description="Total sheets in spreadsheet")
    chunk_number: Optional[int] = Field(None, description="Chunk number for large texts")
    total_chunks: Optional[int] = Field(None, description="Total chunks for large texts")
    title: Optional[str] = Field(None, description="Document title")
    width: Optional[float] = Field(None, description="Width of page (for PDFs)")
    height: Optional[float] = Field(None, description="Height of page (for PDFs)")
    reranking_score: Optional[float] = Field(None, description="Reranking score")
    additional_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata fields"
    )
    
    class Config:
        extra = "allow"  # Allow additional fields that aren't explicitly defined


class DocumentSchema(BaseModel):
    """
    Schema for a document or document chunk.
    """
    content: str = Field(..., description="Text content of the document")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Document metadata")


class DocumentSearchResult(BaseModel):
    """
    Result of a document search or retrieval operation.
    """
    documents: List[DocumentSchema] = Field(..., description="Retrieved documents")
    total: int = Field(..., description="Total number of matching documents")
    reranked: bool = Field(False, description="Whether results were reranked")
    from_cache: bool = Field(False, description="Whether results came from cache")
    search_time: float = Field(..., description="Time taken for search (seconds)")


class ChatMessage(BaseModel):
    """
    Schema for a chat message.
    """
    role: str = Field(..., description="Role of the message sender (user, assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.now, description="Timestamp of the message"
    )


class ChatHistory(BaseModel):
    """
    Schema for chat history.
    """
    messages: List[ChatMessage] = Field(default=[], description="List of chat messages")


class ChatResponse(BaseModel):
    """
    Schema for a chat response.
    """
    response: str = Field(..., description="Response content")
    sources: List[Dict[str, Any]] = Field(
        default=[], description="Sources used for the response"
    )
    from_cache: bool = Field(False, description="Whether response came from cache")
    processing_time: float = Field(..., description="Time taken to generate response (seconds)")
    documents: Optional[List[DocumentSchema]] = Field(
        None, description="Retrieved documents (if return_documents=True)"
    )