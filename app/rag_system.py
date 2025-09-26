from collections import deque
import re
import uuid
import torch
import logging
import asyncio
import aiosqlite
from pathlib import Path
from llama_cpp import Llama
from typing import List, AsyncGenerator, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .db import Memory, client
from config import ModelConfig, settings
logger = logging.getLogger(__name__)


class FTS5Utils:
    def __init__(self):
        self.db_path = self._extract_db_path(settings.database_url)

    def _extract_db_path(self, url: str) -> str:
        url = url.split("///")[1]
        logger.info(url)
        return "./database.db"
    
    async def init_table(self):
        logger.info(f"initializing the db in {self.db_path}")
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts 
                USING fts5(id UNINDEXED, content);
            """)
            await db.commit()

    async def insert(self, db_session:AsyncSession, doc_id: str, content: str):
        try:
            await db_session.execute(
                text("INSERT OR REPLACE INTO memory_fts (id, content) VALUES (:id, :content)"),
                {"id": doc_id, "content": content}
            )
        except Exception as e:
            logger.warning(f"FTS insert failed: {e}")

    async def search(self,db_session:AsyncSession, query: str ,limit: int = 5):
        try:
            result = await db_session.execute(
                text("SELECT id, content FROM memory_fts WHERE content MATCH :query LIMIT :limit"),
                {"query": query, "limit": limit}
            )
            return result.fetchall()
        except Exception as e:
            logger.warning(f"FTS search failed: {e}")
            return []

class FastHistoryManager:
    """Fast in-memory history management with circular buffer"""
    
    def __init__(self, max_history_size: int = 50):
        self.max_history_size = max_history_size
        self.history_buffer: deque = deque(maxlen=max_history_size)
        self.conversation_id: Optional[str] = None
        
    def add_message(self, role: str, content: str, message_id: Optional[int] = None):
        """Add message to fast history buffer"""
        message = {
            "role": role,
            "content": content,
            "id": message_id,
            "token_count": self._estimate_tokens(content)
        }
        self.history_buffer.append(message)
        
    def get_recent_history(self, n: int = 10, max_tokens: int = 1000) -> List[Dict]:
        """Get last n messages or messages within token limit"""
        if not self.history_buffer:
            return []
            
        # Get last n messages
        recent_messages = list(self.history_buffer)[-n:]
        
        # Filter by token count if needed
        if max_tokens:
            filtered_messages = []
            total_tokens = 0
            
            # Work backwards to prioritize recent messages
            for message in reversed(recent_messages):
                if total_tokens + message["token_count"] <= max_tokens:
                    filtered_messages.insert(0, message)
                    total_tokens += message["token_count"]
                else:
                    break
                    
            return filtered_messages
            
        return recent_messages
    
    def _estimate_tokens(self, text: str) -> int:
        """Fast token estimation (roughly 4 chars per token)"""
        return len(text) // 4 + 1
    
    def clear(self):
        """Clear history buffer"""
        self.history_buffer.clear()
    
    def set_conversation_id(self, conversation_id: str):
        """Set current conversation ID"""
        if conversation_id != self.conversation_id:
            self.clear()  # Clear buffer when switching conversations
            self.conversation_id = conversation_id


class RAGSystem:
    def __init__(self, lock):
        self.lock = lock
        self.fts = FTS5Utils()
        self.current_conversation_id: Optional[str] = None
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = client.get_or_create_collection(name="chat_memory")

        self.history_manager = FastHistoryManager(max_history_size=100)
        
        self.max_history_messages = 8  # Max messages to include in context
        self.max_history_tokens = 800   # Max tokens for history context
        self.enable_history_injection = True
        self.enable_rag_search = True

    async def initialize(self):
        if torch.cuda.is_available():
            logger.info("loading the all-MiniLM-L6-v2 to gpu")
            self.encoder = self.encoder.to('cuda') 
        else:
            logger.info("cuda unavailable - loading the all-MiniLM-L6-v2 to cpu")

        await self.fts.init_table()
        logger.info("RAG system initialized")

    def start_conversation(self) -> str:
        self.current_conversation_id = str(uuid.uuid4())
        self.history_manager.set_conversation_id(self.current_conversation_id)
        logger.info(f"Started new conversation: {self.current_conversation_id}")
        return self.current_conversation_id

    def configure_history(self, max_messages: int = 8, max_tokens: int = 800, enable: bool = True):
        """Configure history injection settings"""
        self.max_history_messages = max_messages
        self.max_history_tokens = max_tokens
        self.enable_history_injection = enable
        logger.info(f"History config: max_messages={max_messages}, max_tokens={max_tokens}, enabled={enable}")

    def _clean_and_label(self, doc, role="User"):
        doc = re.sub(r"<\|.*?\|>", "", str(doc)).strip()
        return f"{role}: {doc}"

    async def save_message_fast(self, db: AsyncSession, role: str, content: str) -> Memory:
        if not self.current_conversation_id:
            self.start_conversation()

        original_content = content
        labeled_content = self._clean_and_label(content, role)
        
        memory = Memory(
            role=role,
            content=original_content,
            conversation_id=self.current_conversation_id
        )
        
        db.add(memory)
        await db.flush()
        await db.commit()
        self.history_manager.add_message(role, original_content, memory.id)

        if role != "assistant":
            asyncio.create_task(self._save_to_vectors(str(memory.id), labeled_content))
        
        return memory

    async def _save_to_vectors(self, memory_id: str, content: str):
        try:
            embedding = await asyncio.to_thread(self.encoder.encode, [content])
            self.collection.add(embeddings=embedding.tolist(), documents=[content], ids=[memory_id])
        except Exception as e:
            logger.error(f"Background vector save failed: {e}")

    async def get_recent_history(self, db: AsyncSession, limit: int = 20, offset: int = 0) -> List[Memory]:
        query = (
            select(Memory)
            .where(Memory.conversation_id == self.current_conversation_id)
            .order_by(Memory.timestamp.desc())
            .limit(limit)
            .offset(offset)
        )
        
        result = await db.execute(query)
        messages = result.scalars().all()    
        return list(reversed(messages))
      
    def get_fast_history(self, n: int = 0, max_tokens: int = 0) -> List[Dict]:
            """Get recent history from fast buffer"""
            n = n or self.max_history_messages
            max_tokens = max_tokens or self.max_history_tokens
            return self.history_manager.get_recent_history(n, max_tokens)

    def format_history_context(self, history: List[Dict]) -> str:
        """Format history for context injection"""
        if not history:
            return ""
        
        formatted_messages = []
        for msg in history:
            role = msg["role"].title()
            content = msg["content"].strip()
            # Truncate very long messages
            if len(content) > 200:
                content = content[:200] + "..."
            formatted_messages.append(f"{role}: {content}")
        
        return "\n".join(formatted_messages)

    async def search_history(self, db: AsyncSession, query: str, limit: int = 10) -> List[dict]:
        try:
            fts_results = await self.fts.search(db, query, limit)
            
            if not fts_results:
                return []

            memory_ids = [int(row[0]) for row in fts_results]
            
            db_query = (
                select(Memory)
                .where(Memory.id.in_(memory_ids))
                .order_by(Memory.timestamp.desc())
            )
            
            result = await db.execute(db_query)
            memories = result.scalars().all()
            
            return [
                {
                    "id": memory.id,
                    "role": memory.role,
                    "content": memory.content,
                    "timestamp": memory.timestamp.isoformat(),
                    "conversation_id": memory.conversation_id
                }
                for memory in memories
            ]
            
        except Exception as e:
            logger.error(f"Error searching history: {e}")
            return []

    async def get_conversation_stats(self, db: AsyncSession) -> dict:
        if not self.current_conversation_id:
            return {"total_messages": 0, "user_messages": 0, "assistant_messages": 0}

        query = (
            select(
                func.count(Memory.id).label('total'),
                func.count(Memory.id).filter(Memory.role == 'user').label('user_count'),
                func.count(Memory.id).filter(Memory.role == 'assistant').label('assistant_count')
            )
            .where(Memory.conversation_id == self.current_conversation_id)
        )
        
        result = await db.execute(query)
        stats = result.first()
        if stats:
            return {
                "total_messages": stats.total or 0,
                "user_messages": stats.user_count or 0,
                "assistant_messages": stats.assistant_count or 0
            }
        else:
            return {"total_messages": 0, "user_messages": 0, "assistant_messages": 0}

    def _calculate_relevance_score(self, query_embedding, doc_embedding, text_similarity=0.0):
        semantic_sim = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        # Weight semantic similarity more heavily
        return 0.8 * semantic_sim + 0.2 * text_similarity

    def _filter_and_rank_contexts(self, query: str, documents: List[str], distances: List[float]) -> List[str]:
        if not documents:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_docs = []
        for doc, distance in zip(documents, distances):
            doc_lower = doc.lower()
            doc_words = set(doc_lower.split())
            
            word_overlap = len(query_words.intersection(doc_words)) / len(query_words) if query_words else 0
            
            # Combine semantic similarity (1 - distance) with word overlap
            relevance_score = (1 - distance) * 0.7 + word_overlap * 0.3
            
            # Filter out very short or very long contexts
            if 15 <= len(doc) <= 500:
                scored_docs.append((doc, relevance_score))
        
        # Sort by relevance score and return top contexts
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:3] if score > 0.3]  # Only keep relevant contexts

    async def get_rag_context(self, message: str, top_k: int = 8) -> str:
        if not message.strip() or not self.enable_rag_search:
            return ""
        
        try:
            chroma_results = self.collection.query(query_texts=[message], n_results=top_k)
            documents = chroma_results.get("documents")
            distances = chroma_results.get("distances")
            
            if not documents or not documents[0]:
                return ""
            relevant_docs = None
            if distances:
                relevant_docs = self._filter_and_rank_contexts(message, documents[0], distances[0])
            
            if not relevant_docs:
                return ""
            
            cleaned_contexts = []
            for doc in relevant_docs:
                doc = doc.strip()
                doc = re.sub(r'^(User|Assistant):\s*', '', doc, flags=re.IGNORECASE)
                if len(doc) >= 15:
                    cleaned_contexts.append(doc)

            if not cleaned_contexts:
                return ""
            
            context = "\n---\n".join(cleaned_contexts[:2])  # Limit to top 2 for efficiency
            
            if len(context) > 400:  # Reduced from 600 to leave room for history
                context = context[:400] + "..."
            
            return context
            
        except Exception as e:
            logger.error(f"RAG context retrieval failed: {e}")
            return ""

    async def get_combined_context(self, message: str) -> Tuple[str, str]:
        """Get both history and RAG context efficiently"""
        
        # Get fast history context
        history_context = ""
        if self.enable_history_injection:
            recent_history = self.get_fast_history()
            if recent_history:
                # Exclude the current user message if it's the last one
                if recent_history and recent_history[-1]["role"] == "user":
                    recent_history = recent_history[:-1]
                
                if recent_history:
                    history_context = self.format_history_context(recent_history)
        
        # Get RAG context
        rag_context = ""
        if self.enable_rag_search:
            rag_context = await self.get_rag_context(message)
        
        return history_context, rag_context

    def format_full_context(self, history_context: str, rag_context: str) -> str:
        """Combine history and RAG contexts"""
        contexts = []
        
        if history_context:
            contexts.append(f"=== Recent Conversation ===\n{history_context}")
        
        if rag_context:
            contexts.append(f"=== Relevant Context ===\n{rag_context}")
        
        if contexts:
            return "\n\n".join(contexts)
        
        return ""

    # Backwards compatibility method
    async def get_context(self, message: str, top_k: int = 8) -> str:
        """Get combined context (backwards compatible)"""
        history_context, rag_context = await self.get_combined_context(message)
        return self.format_full_context(history_context, rag_context)



class LLMManager:
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

    def _get_prompt_formatter(self, format_type: str):
        """Get the appropriate prompt formatter"""
        formatters = {
            "chatml_sections": self._format_chatml_sections,
            "llama2": self._format_llama2,
            "mistral": self._format_mistral,
            "codellama": self._format_codellama,
            "phi": self._format_phi,
            "zephyr": self._format_zephyr,
            "generic": self._format_generic
        }
        return formatters.get(format_type, self._format_generic)

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

        config = self.current_config
        max_tokens = max_tokens or config.get('max_tokens', 512)
        temperature = temperature or config.get('temperature', 0.7)
        
        try:
            logger.info(f"Generating with {self.current_model}: {prompt[:100]}...")
            
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature if temperature else 0.5,
                stream=stream,
                stop=config.get('stop_sequences', []),
                repeat_penalty=config.get('repeat_penalty', 1.1),
                top_p=config.get('top_p', 0.9),
                top_k=config.get('top_k', 40)
            )

            if stream:
                for chunk in response:
                    token = chunk.get('choices', [{}])[0].get('text', '') if type(chunk) is dict else " "
                    if token and not self._should_stop_token(token):
                        yield token
            else:
                text = response.get('choices', [{}])[0].get('text', '') if type(response) is dict else " "
                text = self._clean_response(text)
                yield text

        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield f"Error generating response: {str(e)}"

    def _should_stop_token(self, token: str) -> bool:
        stop_indicators = ["<|", "|>"] + self.current_config.get('stop_sequences', [])
        return any(stop in token for stop in stop_indicators)

    def _clean_response(self, text: str) -> str:
        """Clean up model response based on model type"""
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'</?s>', '', text)  # Remove sentence markers
        
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

        prefixes_to_remove = ["Assistant:", "AI:", "Response:", "Answer:", "Output:"]
        for prefix in prefixes_to_remove:
            if text.strip().startswith(prefix):
                text = text.replace(prefix, "", 1).strip()
        
        stop_patterns = [
            r'\n(?:User|Human|Instruct):.*',
            r'\n\[INST\].*',
            r'\n<\|user\|>.*',
            r'\n###.*'
        ]
        
        for pattern in stop_patterns:
            text = re.split(pattern, text, maxsplit=1)[0]
        
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text

    def format_chat_prompt(self, message: str, context: str = "") -> str:
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