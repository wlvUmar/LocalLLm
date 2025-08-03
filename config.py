from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    
    # Application settings
    app_name: str = "Local LLM Chat Interface"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # LLM settings
    models_dir: str = "models"
    llm_server_host: str = "localhost"
    llm_server_port: int = 8080
    llm_server_timeout: int = 60
    llm_startup_timeout: int = 30
    
    # WSL settings
    use_wsl: bool = True
    wsl_server_path: str = "./server"
    
    # Default LLM parameters
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_max_tokens: int = 512
    default_context_size: int = 4096
    
    # File upload settings
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    # Security settings
    cors_origins: list = ["*"]
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

MODELS_DIR = Path(settings.models_dir)
LLM_SERVER_URL = f"http://{settings.llm_server_host}:{settings.llm_server_port}"
TEMPLATES_DIR = Path("templates")
STATIC_DIR = Path("static")
LOGS_DIR = Path("logs")

for directory in [MODELS_DIR, TEMPLATES_DIR, STATIC_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

SUPPORTED_MODEL_EXTENSIONS = [".gguf", ".bin", ".ggml"]

MODEL_CONFIGS = {
    "creative": {
        "temperature": 0.9,
        "top_p": 0.95,
        "repeat_penalty": 1.1
    },
    "precise": {
        "temperature": 0.3,
        "top_p": 0.85,
        "repeat_penalty": 1.05
    },
    "balanced": {
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.1
    }
}

# System prompts for different use cases
SYSTEM_PROMPTS = {
    "assistant": "You are a helpful AI assistant. Provide clear, accurate, and concise responses.",
    "creative": "You are a creative AI assistant. Feel free to be imaginative and expressive in your responses.",
    "technical": "You are a technical AI assistant. Provide detailed, accurate technical information and explanations.",
    "casual": "You are a friendly AI assistant. Keep the conversation casual and engaging."
}

def get_model_config(config_name: str = "balanced") -> dict:
    return MODEL_CONFIGS.get(config_name, MODEL_CONFIGS["balanced"])

def get_system_prompt(prompt_name: str = "assistant") -> str:
    return SYSTEM_PROMPTS.get(prompt_name, SYSTEM_PROMPTS["assistant"])