from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    stream: bool = True
    conversation_id: Optional[str] = None


class ModelSwitchRequest(BaseModel):
    model: str


class ChatResponse(BaseModel):
    response: str


class ModelsResponse(BaseModel):
    models: List[str]
    current_model: Optional[str]
    server_ready: bool


class StatusResponse(BaseModel):
    server_ready: bool
    current_model: Optional[str]
    available_models: List[str]
    process_running: bool


class MessageResponse(BaseModel):
    message: str


class MessageHistoryItem(BaseModel):
    """Model for individual message in history"""
    role: str
    content: str
    timestamp: datetime
    is_tool_call: bool = False
    tool_name: Optional[str] = None
    tool_result: Optional[str] = None 