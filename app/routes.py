"""
API routes and web interface endpoints
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from .schemas import ChatRequest, ModelSwitchRequest, ModelsResponse, StatusResponse, MessageResponse
from .db import get_session
from .memory import conversation_manager, save_message, get_recent_history, search_conversation_history
logger = logging.getLogger(__name__)

router = APIRouter()

templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main chat interface"""
    llm_manager = request.app.state.llm_manager
    models = llm_manager.get_available_models()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": models,
        "current_model": llm_manager.current_model
    })

@router.get("/api/models", response_model=ModelsResponse)
async def get_models(request:Request):
    """Get list of available models"""
    llm_manager = request.app.state.llm_manager

    models = llm_manager.get_available_models()
    return ModelsResponse(
        models=models,
        current_model=llm_manager.current_model,
        server_ready=await llm_manager.is_server_ready()
    )


@router.post("/api/switch-model", response_model=MessageResponse)
async def switch_model(request:Request, model_switch_request: ModelSwitchRequest):
    """Switch to a different model"""
    llm_manager = request.app.state.llm_manager
    model = model_switch_request.model
    
    models = llm_manager.get_available_models()
    if model not in models:
        raise HTTPException(status_code=400, detail="Model not found")
    
    if model == llm_manager.current_model:
        return MessageResponse(message=f"Model {model} is already active")
    
    logger.info(f"Switching to model: {model}")
    success = await llm_manager.start_server(model)
    
    if success:
        return MessageResponse(message=f"Successfully switched to model: {model}")
    else:
        raise HTTPException(status_code=500, detail="Failed to switch model")

@router.post("/api/start-conversation")
async def start_conversation():
    """Start a new conversation"""
    conversation_id = await conversation_manager.start_conversation()
    return {"conversation_id": conversation_id, "message": "New conversation started"}

@router.get("/api/conversation-context")
async def get_conversation_context(request: Request, db: AsyncSession = Depends(get_session)):
    """Get current conversation context"""
    context = await get_conversation_context(request,db)
    return context

@router.post("/api/chat")
async def chat(request: Request, chat_request: ChatRequest, db: AsyncSession = Depends(get_session)):
    llm_manager = request.app.state.llm_manager

    if not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Ensure we have a conversation started
    if not conversation_manager.current_conversation_id:
        await conversation_manager.start_conversation()
    
    # Save user message
    await save_message(db, "user", chat_request.message)

    # Switch model if requested
    if chat_request.model and chat_request.model != llm_manager.current_model:
        models = llm_manager.get_available_models()
        if chat_request.model in models:
            await llm_manager.start_server(chat_request.model)

    if chat_request.stream:
        async def streamer():
            assistant_response = ""
            async for chunk in llm_manager.chat_completion_with_tools(chat_request.message, db, stream=True):
                assistant_response += chunk
                yield f"data: {chunk}\n\n"

            await save_message(db, "assistant", assistant_response)
        return StreamingResponse(streamer(), media_type="text/event-stream")

    else:
        response_text = ""
        async for chunk in llm_manager.chat_completion_with_tools(chat_request.message, db, stream=False):
            response_text += chunk
        await save_message(db, "assistant", response_text)
        return {"response": response_text}

@router.get("/api/history")
async def get_history(limit: int = 20, db: AsyncSession = Depends(get_session)):
    """Get conversation history"""
    history = await get_recent_history(db, limit=limit)
    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "is_tool_call": msg.is_tool_call,
                "tool_name": msg.tool_name
            }
            for msg in history
        ]
    }

@router.post("/api/search-history")
async def search_history(query: str, db: AsyncSession = Depends(get_session)):
    """Search conversation history"""
    results = await search_conversation_history(db, query)
    return {"results": results}

@router.get("/api/status", response_model=StatusResponse)
async def get_status(request:Request):
    """Get current application status"""
    llm_manager = request.app.state.llm_manager
    return StatusResponse(
        server_ready=await llm_manager.is_server_ready(),
        current_model=llm_manager.current_model,
        available_models=llm_manager.get_available_models(),
        process_running=llm_manager.current_process is not None
    ) 