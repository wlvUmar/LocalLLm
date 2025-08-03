"""
FastAPI LLM Chat Application Package
"""

from .main import app
from .llm_manager import LLMManager

__version__ = "1.0.0"
__all__ = ["app", "LLMManager"] 