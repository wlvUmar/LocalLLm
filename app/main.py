import os
import logging
import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles

from .db import init_db
from .routes import router
from config import settings
from .rag_system import RAGSystem, LLMManager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI LLM Chat Application...")
    
    lock = asyncio.Lock()    
    llm_manager = LLMManager()
    rag_system = RAGSystem(lock)
    
    await rag_system.initialize()

    models = llm_manager.get_available_models()
    if models:
        llm_manager.load_model(models[0])


    app.state.llm_manager = llm_manager
    app.state.rag_system = rag_system

    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("chroma_data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    app.mount("/static", StaticFiles(directory="static"), name="static")

    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.error("Please ensure PostgreSQL is running and accessible")
        yield
        return

    models = llm_manager.get_available_models()
    if not models:
        logger.warning("No models found in models directory")
        logger.warning("Please add model files (.gguf, .bin, .ggml) to the models/ directory")
        yield
        return

    default_model = models[0]
    logger.info(f"Starting with default model: {default_model}")

    yield  

 
    logger.info("Application shutdown completed")




app = FastAPI(
    title=settings.app_name,
    description=settings.app_disc,
    version=settings.app_version,
    lifespan=lifespan
)


app.include_router(router)

