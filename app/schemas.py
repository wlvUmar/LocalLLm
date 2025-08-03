"""
Pydantic models for API requests and responses
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat completion"""
    message: str
    model: Optional[str] = None
    stream: bool = True
    conversation_id: Optional[str] = None


class ModelSwitchRequest(BaseModel):
    """Request model for switching models"""
    model: str


class ChatResponse(BaseModel):
    """Response model for chat completion"""
    response: str


class ModelsResponse(BaseModel):
    """Response model for available models"""
    models: List[str]
    current_model: Optional[str]
    server_ready: bool


class StatusResponse(BaseModel):
    """Response model for application status"""
    server_ready: bool
    current_model: Optional[str]
    available_models: List[str]
    process_running: bool


class MessageResponse(BaseModel):
    """Response model for simple messages"""
    message: str


class ConversationStartResponse(BaseModel):
    """Response model for starting a conversation"""
    conversation_id: str
    message: str


class ConversationContextResponse(BaseModel):
    """Response model for conversation context"""
    step: str
    summary: str
    history: List[Dict[str, Any]]


class HistoryResponse(BaseModel):
    """Response model for conversation history"""
    messages: List[Dict[str, Any]]


class SearchHistoryRequest(BaseModel):
    """Request model for searching history"""
    query: str
    top_k: int = 5


class SearchHistoryResponse(BaseModel):
    """Response model for history search results"""
    results: List[str]


class MessageHistoryItem(BaseModel):
    """Model for individual message in history"""
    role: str
    content: str
    timestamp: datetime
    is_tool_call: bool = False
    tool_name: Optional[str] = None
    tool_result: Optional[str] = None 