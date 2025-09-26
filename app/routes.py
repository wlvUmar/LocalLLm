import asyncio
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from .schemas import ChatRequest, ModelSwitchRequest, StatusResponse, MessageResponse
from .db import get_session


logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")



@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    llm_manager = request.app.state.llm_manager
    models = llm_manager.get_available_models()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": models,
        "current_model": llm_manager.current_model
    })




@router.post("/api/switch-model", response_model=MessageResponse)
async def switch_model(request: Request, model_switch_request: ModelSwitchRequest):
    llm_manager = request.app.state.llm_manager
    model = model_switch_request.model
    models = llm_manager.get_available_models()
    
    if model not in models:
        raise HTTPException(status_code=400, detail="Model not found")

    if model == llm_manager.current_model:
        return MessageResponse(message=f"Model {model} is already active")

    logger.info(f"Switching to model: {model}")
    success = llm_manager.load_model(model)

    if success:
        return MessageResponse(message=f"Successfully switched to model: {model}")
    else:
        raise HTTPException(status_code=500, detail="Failed to switch model")


@router.post("/api/chat")
async def chat(request: Request, chat_request: ChatRequest, db: AsyncSession = Depends(get_session)):
    """Main chat endpoint with improved response handling"""

    if not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    llm_manager = request.app.state.llm_manager
    rag_system = request.app.state.rag_system
    
    if not rag_system.current_conversation_id:
        rag_system.start_conversation()

    logger.info("Saving user message")
    await rag_system.save_message_fast(db, "user", chat_request.message)

    if chat_request.model and chat_request.model != llm_manager.current_model:
        models = llm_manager.get_available_models()
        if chat_request.model in models:
            success = llm_manager.load_model(chat_request.model)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to load requested model")

    if not llm_manager.is_ready():
        raise HTTPException(status_code=503, detail="No model loaded")

    logger.info("Retrieving context")
    context = await rag_system.get_context(chat_request.message)
    
    if context:
        logger.info(f"Using context: {context[:100]}...")
    else:
        logger.info("No relevant context found")

    if chat_request.stream:
        async def streamer():
            assistant_response = ""
            chunk_count = 0
            max_chunks = 500
            
            try:
                async for chunk in llm_manager.chat_completion(
                    chat_request.message, 
                    context=context, 
                    stream=True
                ):
                    chunk_count += 1
                    if chunk_count > max_chunks:
                        logger.warning("Max chunks reached, stopping generation")
                        break
                        
                    if chunk and chunk.strip():
                        assistant_response += chunk
                        yield f"data: {chunk}\n\n"
                        await asyncio.sleep(0.01)  # Small delay to prevent overwhelming

                # Clean up the final response
                assistant_response = assistant_response.strip()
                if assistant_response:
                    logger.info(f"Saving assistant response: {assistant_response[:100]}...")
                    await rag_system.save_message_fast(db, "assistant", assistant_response)
                    yield "data: [DONE]\n\n"
                else:
                    logger.warning("Empty assistant response generated")
                    yield "data: I apologize, but I couldn't generate a proper response. Please try again.\n\n"
                    yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: Error: I encountered an issue while generating the response. Please try again.\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            streamer(), 
            media_type="text/event-stream", 
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    else:
        try:
            response_text = ""
            chunk_count = 0
            max_chunks = 100
            
            async for chunk in llm_manager.chat_completion(
                chat_request.message, 
                context=context, 
                stream=False
            ):
                chunk_count += 1
                if chunk_count > max_chunks:
                    break
                response_text += chunk
            
            response_text = response_text.strip()
            
            if not response_text:
                response_text = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            await rag_system.save_message_fast(db, "assistant", response_text)
            return {"response": response_text}
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail="An error occurred while processing your request")



@router.get("/api/messages")
async def get_history(
    request:Request,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_session)
):
    """Get conversation history"""
    rag_system = request.app.state.rag_system

    history = await rag_system.get_recent_history(db, limit=limit, offset=offset)
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




@router.get("/api/status", response_model=StatusResponse)
async def get_status(request: Request):
    llm_manager = request.app.state.llm_manager

    return StatusResponse(
        server_ready=llm_manager.is_ready(),
        current_model=llm_manager.current_model,
        available_models=llm_manager.get_available_models(),
        process_running=llm_manager.is_ready()
    )


@router.get("/api/stats")
async def get_stats(request: Request, db: AsyncSession = Depends(get_session)):
    rag_system = request.app.state.rag_system
    stats = await rag_system.get_conversation_stats(db)
    return stats