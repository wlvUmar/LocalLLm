import re
import torch
import logging
from pathlib import Path
from llama_cpp import Llama
from threading import Thread
from typing import List, AsyncGenerator, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

logger = logging.getLogger(__name__)



class HuggingFaceLlamaWrapper:
    """
    A wrapper class that mimics llama_cpp.Llama interface but uses HuggingFace Transformers.
    This allows you to replace self.llm without changing your existing generate() method.
    """
    
    def __init__(self, model_name_or_path: str, device: str = "auto", **kwargs):
        """
        Initialize the wrapper with a HuggingFace model.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            device: Device to load model on ("auto", "cuda", "cpu")
            **kwargs: Additional arguments for model loading (torch_dtype, load_in_8bit, etc.)
        """
        self.model_name_or_path = model_name_or_path
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.kwargs = kwargs
        
    def _setup_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self):
        try:
            logger.info(f"Loading model {self.model_name_or_path} on {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True,
                **self.kwargs
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded")
    
    def __call__(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, 
                 stream: bool = False, stop: list = None, **kwargs):
        """
        Generate text using the loaded model.
        Mimics the llama_cpp.Llama.__call__ interface.
        
        Returns:
            Dict with 'choices' key containing generated text, matching llama_cpp format
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                **kwargs
            }
            
            if stop:
                stop_token_ids = []
                for stop_word in stop:
                    stop_ids = self.tokenizer.encode(stop_word, add_special_tokens=False)
                    stop_token_ids.extend(stop_ids)
                if stop_token_ids:
                    generation_kwargs["eos_token_id"] = stop_token_ids
            
            if stream:
                return self._stream_generate(inputs, generation_kwargs, stop)
            else:
                return self._generate(inputs, generation_kwargs, stop)
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {"choices": [{"text": f"Error: {str(e)}"}]}
    
    def _generate(self, inputs, generation_kwargs, stop_tokens):
        with torch.no_grad():
            outputs = self.model.generate(inputs, **generation_kwargs)
        
        new_tokens = outputs[0][inputs.shape[1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        if stop_tokens:
            for stop_word in stop_tokens:
                if stop_word in generated_text:
                    generated_text = generated_text.split(stop_word)[0]
                    break
        
        return {"choices": [{"text": generated_text}]}
    
    def _stream_generate(self, inputs, generation_kwargs, stop_tokens):
        """Streaming generation that yields chunks."""
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        generation_kwargs["streamer"] = streamer
        thread = Thread(
            target=self.model.generate, 
            args=(inputs,), 
            kwargs=generation_kwargs
        )
        thread.start()
        generated_text = ""
        for chunk in streamer:
            generated_text += chunk
            
            should_stop = False
            if stop_tokens:
                for stop_word in stop_tokens:
                    if stop_word in generated_text:
                        final_text = generated_text.split(stop_word)[0]
                        remaining = final_text[len(generated_text) - len(chunk):]
                        if remaining:
                            yield {"choices": [{"text": remaining}]}
                        should_stop = True
                        break
            
            if should_stop:
                break
                
            yield {"choices": [{"text": chunk}]}
        
        thread.join()


class ModelManager:

    def __init__(self):
        self.current_model = None
        self.loaded_models = {}
        
    def load_model(self, model_name: str, model_path: str, **kwargs) -> HuggingFaceLlamaWrapper:
        """
        Load a new model or return cached one.
        
        Args:
            model_name: Name to identify the model
            model_path: HuggingFace model name or local path
            **kwargs: Additional model loading arguments
            
        Returns:
            HuggingFaceLlamaWrapper instance
        """
        if model_name in self.loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]
        
        # Unload current model if memory is a concern
        if self.current_model and len(self.loaded_models) > 0:
            logger.info("Unloading previous model to save memory")
            self.unload_current_model()
        
        # Load new model
        wrapper = HuggingFaceLlamaWrapper(model_path, **kwargs)
        wrapper.load_model()
        
        self.loaded_models[model_name] = wrapper
        self.current_model = model_name
        
        return wrapper
    
    def switch_model(self, model_name: str):
        """Switch to a previously loaded model."""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded. Use load_model() first.")
        
        self.current_model = model_name
        return self.loaded_models[model_name]
    
    def unload_model(self, model_name: str):
        """Unload a specific model."""
        if model_name in self.loaded_models:
            self.loaded_models[model_name].unload_model()
            del self.loaded_models[model_name]
            if self.current_model == model_name:
                self.current_model = None
    
    def unload_current_model(self):
        """Unload the current model."""
        if self.current_model:
            self.unload_model(self.current_model)
    
    def unload_all_models(self):
        """Unload all models."""
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)




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


class UniversalLLMManager:
    def __init__(self):
        self.models_dir = Path("models")
        self.current_model: Optional[str] = None
        self.current_config: Dict[str, Any] = {}
        self.llm: Optional[Llama] = None
        self._available_models = self._discover_models()

    def _discover_models(self) -> List[str]:
        if not self.models_dir.exists():
            logger.warning(f"Models directory {self.models_dir} does not exist")
            return []
        
        models = []
        for file in self.models_dir.glob("*.gguf"):
            models.append(file.stem)
        
        logger.info(f"Found {len(models)} models: {models}")
        return models

    def get_available_models(self) -> List[str]:
        return self._available_models

    def load_model(self, model_name: str) -> bool:
        if model_name not in self._available_models:
            logger.error(f"Model {model_name} not found in available models")
            return False

        model_path = self.models_dir / f"{model_name}.gguf"
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        try:
            if self.llm:
                del self.llm
                self.llm = None

            # Get model-specific configuration
            self.current_config = ModelConfig.get_config(model_name)
            logger.info(f"Loading model: {model_name} with config: {ModelConfig.detect_model_type(model_name)}")

            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_threads=-1,
                n_gpu_layers=-1,
                verbose=False
            )
            
            self.current_model = model_name
            logger.info(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def is_ready(self) -> bool:
        return self.llm is not None

    def _format_chatml_sections(self, message: str, context: str = "") -> str:
        """Format for TinyLlama and ChatML-style models"""
        if context:
            return f"""### System
{self.current_config['system_prompt']}

### Context
{context.strip()}

### User
{message.strip()}

### Assistant
"""
        else:
            return f"""### System
{self.current_config['system_prompt']}

### User
{message.strip()}

### Assistant
"""

    def _get_prompt_formatter(self, format_type: str):
        """Get the appropriate prompt formatter"""
        formatters = {
            "llama2": self._format_llama2,
            "mistral": self._format_mistral,
            "codellama": self._format_codellama,
            "phi": self._format_phi,
            "zephyr": self._format_zephyr,
            "generic": self._format_generic
        }
        return formatters.get(format_type, self._format_generic)

    

    def _format_llama2(self, message: str, context: str = "") -> str:
        """Format for Llama 2/3 models"""
        system_msg = self.current_config['system_prompt']
        if context:
            system_msg += f"\n\nRelevant context: {context.strip()}"
        
        return f"[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{message.strip()} [/INST]"

    def _format_mistral(self, message: str, context: str = "") -> str:
        """Format for Mistral models"""
        prompt = message.strip()
        if context:
            prompt = f"Context: {context.strip()}\n\nQuestion: {prompt}"
        
        return f"[INST] {prompt} [/INST]"

    def _format_codellama(self, message: str, context: str = "") -> str:
        """Format for CodeLlama models"""
        system_msg = self.current_config['system_prompt']
        if context:
            system_msg += f"\n\nRelevant context: {context.strip()}"
            
        return f"[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{message.strip()} [/INST]"

    def _format_phi(self, message: str, context: str = "") -> str:
        """Format for Phi models"""
        if context:
            return f"Instruct: {self.current_config['system_prompt']}\n\nContext: {context.strip()}\n\n{message.strip()}\nOutput:"
        else:
            return f"Instruct: {self.current_config['system_prompt']}\n\n{message.strip()}\nOutput:"

    def _format_zephyr(self, message: str, context: str = "") -> str:
        """Format for Zephyr models"""
        if context:
            return f"<|system|>\n{self.current_config['system_prompt']}\n\nContext: {context.strip()}<|user|>\n{message.strip()}<|assistant|>\n"
        else:
            return f"<|system|>\n{self.current_config['system_prompt']}<|user|>\n{message.strip()}<|assistant|>\n"

    def _format_generic(self, message: str, context: str = "") -> str:
        """Generic format for unknown models"""
        if context:
            return f"System: {self.current_config['system_prompt']}\n\nContext: {context.strip()}\n\nUser: {message.strip()}\n\nAssistant:"
        else:
            return f"System: {self.current_config['system_prompt']}\n\nUser: {message.strip()}\n\nAssistant:"

    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> AsyncGenerator[str, None]:
        if not self.llm:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Use model-specific parameters
        config = self.current_config
        max_tokens = max_tokens or config.get('max_tokens', 512)
        temperature = temperature or config.get('temperature', 0.7)
        
        try:
            logger.info(f"Generating with {self.current_model}: {prompt[:100]}...")
            
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                stop=config.get('stop_sequences', []),
                repeat_penalty=config.get('repeat_penalty', 1.1),
                top_p=config.get('top_p', 0.9),
                top_k=config.get('top_k', 40)
            )

            if stream:
                for chunk in response:
                    token = chunk.get('choices', [{}])[0].get('text', '') 
                    if token and not self._should_stop_token(token):
                        yield token
            else:
                text = response.get('choices', [{}])[0].get('text', '')
                text = self._clean_response(text)
                yield text

        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield f"Error generating response: {str(e)}"

    def _should_stop_token(self, token: str) -> bool:
        """Check if token should stop generation"""
        stop_indicators = ["<|", "|>"] + self.current_config.get('stop_sequences', [])
        return any(stop in token for stop in stop_indicators)

    def _clean_response(self, text: str) -> str:
        """Clean up model response based on model type"""
        # Remove special tokens
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'</?s>', '', text)  # Remove sentence markers
        
        # Model-specific cleaning
        model_type = ModelConfig.detect_model_type(self.current_model or "")
        
        if model_type == "mistral":
            text = re.sub(r'\[/?INST\]', '', text)
        elif model_type == "llama":
            text = re.sub(r'\[/?INST\]', '', text)
            text = re.sub(r'<<SYS>>\n.*?\n<</SYS>>', '', text, flags=re.DOTALL)
        elif model_type == "zephyr":
            text = re.sub(r'<\|(?:system|user|assistant)\|>', '', text)
        elif model_type == "phi":
            if text.startswith("Output:"):
                text = text[7:].strip()

        # Remove common unwanted prefixes
        prefixes_to_remove = ["Assistant:", "AI:", "Response:", "Answer:", "Output:"]
        for prefix in prefixes_to_remove:
            if text.strip().startswith(prefix):
                text = text.replace(prefix, "", 1).strip()
        
        # Stop at unwanted continuations
        stop_patterns = [
            r'\n(?:User|Human|Instruct):.*',
            r'\n\[INST\].*',
            r'\n<\|user\|>.*',
            r'\n###.*'
        ]
        
        for pattern in stop_patterns:
            text = re.split(pattern, text, maxsplit=1)[0]
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text

    def format_chat_prompt(self, message: str, context: str = "") -> str:
        """Format prompt using model-specific formatter"""
        formatter = self._get_prompt_formatter(self.current_config.get('prompt_format', 'generic'))
        return formatter(message, context)

    async def chat_completion(
        self, 
        message: str, 
        context: str = "",
        stream: bool = False
    ) -> AsyncGenerator[str, None]:
        prompt = self.format_chat_prompt(message, context)
        
        async for chunk in self.generate(
            prompt=prompt,
            stream=stream
        ):
            yield chunk

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.current_model:
            return {}
        
        model_type = ModelConfig.detect_model_type(self.current_model)
        return {
            "name": self.current_model,
            "type": model_type,
            "config": self.current_config,
            "prompt_format": self.current_config.get('prompt_format'),
            "recommended_settings": {
                "max_tokens": self.current_config.get('max_tokens'),
                "temperature": self.current_config.get('temperature')
            }
        }

# Example usage in your existing code:
"""
# Instead of loading llama_cpp model:
# self.llm = Llama(...)

# Use this:
model_manager = ModelManager()

# Load a model - this replaces your llama_cpp Llama instance
self.llm = model_manager.load_model(
    model_name="llama2-7b", 
    model_path="meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Switch models (your "switch models" feature)
self.llm = model_manager.switch_model("llama2-7b")

# Or load a different model
self.llm = model_manager.load_model(
    model_name="mistral-7b",
    model_path="mistralai/Mistral-7B-Instruct-v0.1"
)

# Your existing generate method will work unchanged!
"""