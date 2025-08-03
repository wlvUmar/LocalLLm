"""
Application startup and shutdown event handlers
"""

import logging
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .llm_manager import LLMManager
from .db import init_db

logger = logging.getLogger(__name__)


def setup_events(app: FastAPI, llm_manager: LLMManager):
    """Setup application startup and shutdown events"""
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize the application on startup"""
        logger.info("Starting FastAPI LLM Chat Application...")
        app.state.llm_manager = llm_manager 
        app.state.chat_history = []
        
        # Create necessary directories
        os.makedirs("templates", exist_ok=True)
        os.makedirs("static", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("chroma_data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Initialize database
        try:
            await init_db()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            logger.error("Please ensure PostgreSQL is running and accessible")
            return
        
        # Initialize LLM
        models = llm_manager.get_available_models()
        if not models:
            logger.warning("No models found in models directory")
            logger.warning("Please add model files (.gguf, .bin, .ggml) to the models/ directory")
            return
        
        default_model = models[0]
        logger.info(f"Starting with default model: {default_model}")
        
        try:
            success = await llm_manager.start_server(default_model)
            if success:
                logger.info("Application startup completed successfully")
            else:
                logger.error("Failed to start LLM server on startup")
                logger.error("Please ensure llama.cpp server is available and WSL2 is properly configured")
        except Exception as e:
            logger.error(f"Error during startup: {e}")
            logger.error("Application will start but LLM functionality may not be available")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on application shutdown"""
        logger.info("Shutting down application...")
        await llm_manager.stop_server()
        logger.info("Application shutdown completed") 