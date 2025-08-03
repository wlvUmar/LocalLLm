import re
import httpx
import json
import logging
import asyncio
import subprocess
from pathlib import Path
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any, Tuple
from .memory import conversation_manager, tool_manager, get_conversation_context


logger = logging.getLogger(__name__)


class LLMManager:
    """Manages llama.cpp server subprocess and model switching with tool support"""
    
    def __init__(self, models_dir: str = "./models", server_path: str = ""):
        self.models_dir = Path(models_dir).resolve()
        self.server_path = "/home/hapko/llama.cpp/build/bin/llama-server"

        self.current_process: Optional[subprocess.Popen] = None
        self.current_model: Optional[str] = None
        self.server_url = "http://localhost:8080"
        self.wsl_command_prefix = ["wsl", "-d", "ubuntu", "--"]

        
        try:
            subprocess.run(["wsl", "--version"], capture_output=True, check=True)
            self.wsl_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.wsl_available = False
            logger.warning("WSL not available. LLM functionality will be limited.")
    
    def _to_wsl_path(self, path: Path) -> str:
        return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]

    def get_available_models(self) -> List[str]:
        """Get list of available model files from models directory"""
        try:
            if not self.models_dir.exists():
                logger.warning(f"Models directory {self.models_dir} does not exist")
                return []
            
            models = []
            for file in self.models_dir.iterdir():
                if file.is_file() and file.suffix.lower() in ['.gguf', '.bin', '.ggml']:
                    models.append(file.name)
            
            logger.info(f"Found {len(models)} models: {models}")
            return sorted(models)
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def start_server(self, model: str, timeout: int = 30) -> bool:
        """Start llama.cpp server with specified model"""
        try:
            await self.stop_server()
            
            if not self.wsl_available:
                logger.error("Cannot start server: WSL is not available")
                return False
            
            model_path = self.models_dir / model
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Build command for WSL2
            cmd = self.wsl_command_prefix + [
                self.server_path,
                "-m", self._to_wsl_path(model_path),
                "--port", "8080",
                "--host", "0.0.0.0",
                "-c", "4096",  # Context size
                "--log-disable"  # Disable verbose logging
            ]
            
            logger.info(f"Starting server with command: {' '.join(cmd)}")
            
            self.current_process = subprocess.Popen(    
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            for i in range(timeout):
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{self.server_url}/health", timeout=1.0)
                        if response.status_code == 200:
                            self.current_model = model
                            logger.info(f"Server started successfully with model: {model}")
                            return True
                except:
                    pass
                
                await asyncio.sleep(1)
                
                if self.current_process.poll() is not None:
                    stdout, stderr = self.current_process.communicate()
                    logger.error(f"Server process died. stdout: {stdout}, stderr: {stderr}")
                    return False
            
            logger.error("Server startup timeout")
            await self.stop_server()
            return False
            
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            await self.stop_server()
            return False
    
    async def stop_server(self):
        if self.current_process:
            try:
                self.current_process.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(self._wait_for_process()), 
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Process didn't terminate gracefully, killing...")
                    self.current_process.kill()
                    await asyncio.create_task(self._wait_for_process())
                
                logger.info("Server stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
            finally:
                self.current_process = None
                self.current_model = None
    
    async def _wait_for_process(self):
        while self.current_process and self.current_process.poll() is None:
            await asyncio.sleep(0.1)
    
    async def is_server_ready(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.server_url}/health", timeout=2.0)
                return response.status_code == 200
        except:
            return False

    def _build_system_prompt(self, session: AsyncSession) -> str:
        """Build system prompt with tools and context"""
        tools_description = """
Available tools:
1. search_history(query) - Search conversation history for relevant information
2. get_context() - Get current conversation context and step
3. update_step(step, summary) - Update conversation step and summary

To use a tool, respond with: [TOOL:tool_name:parameters]
Example: [TOOL:search_history:previous discussion about database]
Example: [TOOL:get_context:]
Example: [TOOL:update_step:planning:User wants to implement database features]

Always maintain conversation context and use tools when needed to provide better responses.
"""
        
        return f"""You are a helpful AI assistant with access to conversation history and tools. 

{tools_description}

Current conversation step: {conversation_manager.current_step}
Context summary: {conversation_manager.context_summary}

Remember to:
1. Use tools when you need to search history or get context
2. Update conversation steps when appropriate
3. Maintain context throughout the conversation
4. Provide helpful and accurate responses

Respond naturally to the user, but use tools when needed."""

    def _extract_tool_calls(self, response: str) -> List[Tuple[str, str]]:
        """Extract tool calls from LLM response"""
        tool_pattern = r'\[TOOL:([^:]+):([^\]]*)\]'
        matches = re.findall(tool_pattern, response)
        return [(tool_name.strip(), params.strip()) for tool_name, params in matches]

    async def _execute_tools(self, tool_calls: List[Tuple[str, str]], session: AsyncSession) -> List[str]:
        """Execute tool calls and return results"""
        results = []
        for tool_name, params in tool_calls:
            try:
                if tool_name == "search_history":
                    result, _ = await tool_manager.execute_tool("search_history", query=params, session=session)
                elif tool_name == "get_context":
                    result, _ = await tool_manager.execute_tool("get_context", session=session)
                elif tool_name == "update_step":
                    parts = params.split(':', 1)
                    step = parts[0].strip()
                    summary = parts[1].strip() if len(parts) > 1 else None
                    result, _ = await tool_manager.execute_tool("update_step", step=step, summary=summary, session=session)
                else:
                    result = f"Unknown tool: {tool_name}"
                
                results.append(f"Tool {tool_name} result: {result}")
            except Exception as e:
                results.append(f"Error executing tool {tool_name}: {str(e)}")
        
        return results

    async def chat_completion_with_tools(self, user_message: str, session: AsyncSession, stream: bool = True):
        """Send chat completion request with tool support"""
        if not await self.is_server_ready():
            raise HTTPException(status_code=503, detail="LLM server is not ready")
        
        system_prompt = self._build_system_prompt(session)
        
        # Get conversation context
        context = await get_conversation_context(session)
        recent_messages = context.get("history", [])
        
        # Build conversation history
        conversation_history = ""
        for msg in recent_messages[-10:]:  # Last 10 messages
            conversation_history += f"{msg.role.capitalize()}: {msg.content}\n"
        
        # Build full prompt
        full_prompt = f"{system_prompt}\n\nConversation History:\n{conversation_history}\nUser: {user_message}\nAssistant:"
        
        payload = {
            "prompt": full_prompt,
            "n_predict": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": stream,
            "stop": ["</s>", "\n\n", "User:", "Human:"]
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                if stream:
                    async with client.stream(
                        "POST", 
                        f"{self.server_url}/completion",
                        json=payload
                    ) as response:
                        logger.info(f"response {response}")

                        if response.status_code != 200:
                            logger.error(f"Model response {response}")
                            raise HTTPException(status_code=response.status_code, detail="LLM server error")
                        
                        full_response = ""
                        async for chunk in response.aiter_text():
                            logger.info(f"Yielding chunk: {chunk}")

                            if chunk.strip():
                                try:
                                    # Parse SSE format
                                    if chunk.startswith("data: "):
                                        data = json.loads(chunk[6:])
                                        if "content" in data:
                                            content = data["content"]
                                            full_response += content
                                            yield content
                                except json.JSONDecodeError:
                                    continue
                        
                        # Process tool calls after full response
                        tool_calls = self._extract_tool_calls(full_response)
                        if tool_calls:
                            tool_results = await self._execute_tools(tool_calls, session)
                            for result in tool_results:
                                yield f"\n[Tool Result: {result}]"
                else:
                    response = await client.post(f"{self.server_url}/completion", json=payload)
                    if response.status_code != 200:
                        raise HTTPException(status_code=response.status_code, detail="LLM server error")
                    
                    result = response.json()
                    content = result.get("content", "")
                    
                    # Process tool calls
                    tool_calls = self._extract_tool_calls(content)
                    if tool_calls:
                        tool_results = await self._execute_tools(tool_calls, session)
                        content += "\n" + "\n".join([f"[Tool Result: {result}]" for result in tool_results])
                    
                    yield content
                    
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Request timeout")
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def chat_completion(self, prompt: str, stream: bool = True):
        """Legacy chat completion method for backward compatibility"""
        if not await self.is_server_ready():
            raise HTTPException(status_code=503, detail="LLM server is not ready")
        
        payload = {
            "prompt": prompt,
            "n_predict": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": stream,
            "stop": ["</s>", "\n\n"]
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                if stream:
                    async with client.stream(
                        "POST", 
                        f"{self.server_url}/completion",
                        json=payload
                    ) as response:
                        if response.status_code != 200:
                            raise HTTPException(status_code=response.status_code, detail="LLM server error")
                        
                        async for chunk in response.aiter_text():
                            if chunk.strip():
                                try:
                                    # Parse SSE format
                                    if chunk.startswith("data: "):
                                        data = json.loads(chunk[6:])
                                        if "content" in data:
                                            yield data["content"]
                                except json.JSONDecodeError:
                                    continue
                else:
                    response = await client.post(f"{self.server_url}/completion", json=payload)
                    if response.status_code != 200:
                        raise HTTPException(status_code=response.status_code, detail="LLM server error")
                    
                    result = response.json()
                    yield result.get("content", "")
                    
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Request timeout")
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise HTTPException(status_code=500, detail=str(e)) 