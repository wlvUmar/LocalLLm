import os
import logging
from pydantic_settings import BaseSettings
from typing import Dict , Any
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    
    app_name: str = "Local LLM Chat Interface"
    app_disc: str = "A customized local LLM with enhanced memory features for context-aware responses and offline use."

    app_version: str = "1.0.0"
    debug: bool = False
    
    host: str = "0.0.0.0"
    port: int = 8000
    
    database_url: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./database.db")
    
    models_dir: str = "./models"
    
    
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_max_tokens: int = 512
    default_context_size: int = 4096
    
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    cors_origins: list = ["*"]
    
    log_level: str = "info"
    log_file: str = "logs/app.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
settings = Settings()


class ModelConfig:

    """Configuration for different model types"""
    
    CONFIGS = {
        # TinyLlama models
        "tinyllama": {
            "prompt_format": "chatml_sections",
            "max_tokens": 256,
            "temperature": 0.6,
            "repeat_penalty": 1.1,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["User:", "Human:", "Q:", "Question:", "\n\n\n", "###", "---"],
            "system_prompt": "You are a helpful AI assistant. Give concise, accurate answers."
        },
        
        # Llama 2/3 models
        "llama": {
            "prompt_format": "llama2",
            "max_tokens": 512,
            "temperature": 0.7,
            "repeat_penalty": 1.05,
            "top_p": 0.95,
            "top_k": 50,
            "stop_sequences": ["[INST]", "[/INST]", "Human:", "Assistant:"],
            "system_prompt": "You are a helpful, respectful and honest assistant."
        },
        
        # Mistral models
        "mistral": {
            "prompt_format": "mistral",
            "max_tokens": 512,
            "temperature": 0.7,
            "repeat_penalty": 1.1,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["[INST]", "[/INST]", "</s>"],
            "system_prompt": "You are a helpful assistant."
        },
        
        # CodeLlama models
        "codellama": {
            "prompt_format": "codellama",
            "max_tokens": 1024,
            "temperature": 0.2,
            "repeat_penalty": 1.05,
            "top_p": 0.95,
            "top_k": 50,
            "stop_sequences": ["[INST]", "[/INST]", "Human:", "```\n\n"],
            "system_prompt": "You are a helpful coding assistant. Provide clean, well-commented code."
        },
        
        # Phi models
        "phi": {
            "prompt_format": "phi",
            "max_tokens": 512,
            "temperature": 0.7,
            "repeat_penalty": 1.1,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["Instruct:", "Output:", "Human:", "\n\n\n"],
            "system_prompt": "You are a helpful AI assistant."
        },
        
        # Zephyr models
        "zephyr": {
            "prompt_format": "zephyr",
            "max_tokens": 512,
            "temperature": 0.7,
            "repeat_penalty": 1.05,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["<|user|>", "<|assistant|>", "<|system|>"],
            "system_prompt": "You are a friendly and helpful assistant."
        },
        
        # Default fallback
        "default": {
            "prompt_format": "generic",
            "max_tokens": 512,
            "temperature": 0.7,
            "repeat_penalty": 1.1,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["Human:", "User:", "Assistant:", "\n\n\n"],
            "system_prompt": "You are a helpful AI assistant."
        }
    }
    
    @classmethod
    def detect_model_type(cls, model_name: str) -> str:
        """Detect model type from model name"""
        model_name_lower = model_name.lower()
        
        if "tinyllama" in model_name_lower:
            return "tinyllama"
        elif "llama-2" in model_name_lower or "llama2" in model_name_lower:
            return "llama"
        elif "llama-3" in model_name_lower or "llama3" in model_name_lower:
            return "llama"
        elif "codellama" in model_name_lower or "code-llama" in model_name_lower:
            return "codellama"
        elif "mistral" in model_name_lower:
            return "mistral"
        elif "phi" in model_name_lower:
            return "phi"
        elif "zephyr" in model_name_lower:
            return "zephyr"
        else:
            return "default"
    
    @classmethod
    def get_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for a model"""
        model_type = cls.detect_model_type(model_name)
        return cls.CONFIGS.get(model_type, cls.CONFIGS["default"])

