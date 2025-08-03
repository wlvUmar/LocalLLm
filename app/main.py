"""
Main FastAPI application module
"""

import logging
from fastapi import FastAPI

from .routes import router
from .events import setup_events
from .llm_manager import LLMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Local LLM Chat Interface",
    description="A web interface for chatting with local LLM models via llama.cpp",
    version="1.0.0"
)

llm_manager = LLMManager()
setup_events(app, llm_manager)

from .routes import router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 