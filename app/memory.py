from sqlalchemy import select, update
from sentence_transformers import SentenceTransformer
from .db import Memory, ConversationState, collection, async_session
import uuid
from typing import List, Dict, Tuple
import datetime


embedder = SentenceTransformer("all-MiniLM-L6-v2")

class ConversationManager:
    def __init__(self):
        self.current_conversation_id = None
        self.current_step = "initial"
        self.context_summary = ""
    
    async def start_conversation(self, conversation_id = None) -> str:
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        self.current_conversation_id = conversation_id
        self.current_step = "initial"
        
        async with async_session() as session:
            state = ConversationState(
                conversation_id=conversation_id,
                current_step="initial",
                context_summary=""
            )
            session.add(state)
            await session.commit()
        
        return conversation_id
    
    async def update_conversation_state(self, step: str, context_summary = None):
        if not self.current_conversation_id:
            return
        
        async with async_session() as session:
            await session.execute(
                update(ConversationState)
                .where(ConversationState.conversation_id == self.current_conversation_id)
                .values(
                    current_step=step,
                    context_summary=context_summary or self.context_summary,
                    last_updated=datetime.datetime.utcnow()
                )
            )
            await session.commit()
        
        self.current_step = step
        if context_summary:
            self.context_summary = context_summary

conversation_manager = ConversationManager()

async def save_message(session, role: str, content: str, conversation_id = None, 
                      is_tool_call: bool = False, tool_name  = None, tool_result = None):
    if not conversation_id:
        conversation_id = conversation_manager.current_conversation_id
    
    embedding_id = add_to_vector_db(content) if not is_tool_call else None
    
    msg = Memory(
        role=role, 
        content=content, 
        embedding_id=embedding_id,
        conversation_id=conversation_id,
        is_tool_call=is_tool_call,
        tool_name=tool_name,
        tool_result=tool_result
    )
    session.add(msg)
    await session.commit()

async def get_recent_history(session, conversation_id = None, limit: int = 10) -> List[Memory]:
    if not conversation_id:
        conversation_id = conversation_manager.current_conversation_id
    
    if not conversation_id:
        return []
    
    result = await session.execute(
        select(Memory)
        .where(Memory.conversation_id == conversation_id)
        .order_by(Memory.timestamp.desc())
        .limit(limit)
    )
    messages = result.scalars().all()
    return list(reversed(messages))

async def get_conversation_context(session, conversation_id = None) -> Dict:
    if not conversation_id:
        conversation_id = conversation_manager.current_conversation_id
    
    if not conversation_id:
        return {"step": "initial", "summary": "", "history": []}
    
    result = await session.execute(
        select(ConversationState)
        .where(ConversationState.conversation_id == conversation_id)
    )
    state = result.scalar_one_or_none()
    
    history = await get_recent_history(session, conversation_id, limit=20)
    
    return {
        "step": state.current_step if state else "initial",
        "summary": state.context_summary if state else "",
        "history": history
    }

def add_to_vector_db(text: str) -> str:
    embedding = embedder.encode([text])[0]
    vector_id = str(uuid.uuid4())
    collection.add(documents=[text], embeddings=[embedding], ids=[vector_id])
    return vector_id

def search_similar(query: str, top_k: int = 5) -> List[str]:
    embedding = embedder.encode([query])[0]
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results['documents'][0] if results['documents'] else []

async def search_conversation_history(session, query: str, conversation_id = None, top_k: int = 3) -> List[str]:
    if not conversation_id:
        conversation_id = conversation_manager.current_conversation_id
    
    if not conversation_id:
        return []
    
    result = await session.execute(
        select(Memory)
        .where(Memory.conversation_id == conversation_id)
        .order_by(Memory.timestamp.desc())
        .limit(50)
    )
    messages = result.scalars().all()
    
    if not messages:
        return []
    
    texts = [msg.content for msg in messages]
    embeddings = embedder.encode(texts)
    
    query_embedding = embedder.encode([query])[0]
    
    similarities = []
    for i, embedding in enumerate(embeddings):
        similarity = sum(a * b for a, b in zip(query_embedding, embedding))
        similarities.append((similarity, texts[i]))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in similarities[:top_k]]

class ToolManager:
    def __init__(self):
        self.available_tools = {
            "search_history": self.search_history_tool,
            "get_context": self.get_context_tool,
            "update_step": self.update_step_tool
        }
    
    async def search_history_tool(self, query: str, session) -> str:
        results = await search_conversation_history(session, query)
        return f"Found {len(results)} relevant messages: " + " | ".join(results)
    
    async def get_context_tool(self, session) -> str:
        context = await get_conversation_context(session)
        return f"Current step: {context['step']}, Summary: {context['summary']}, Recent messages: {len(context['history'])}"
    
    async def update_step_tool(self, step: str, summary= None, session = None) -> str:
        await conversation_manager.update_conversation_state(step, summary)
        return f"Updated conversation step to: {step}"
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Tuple[str, str]:
        if tool_name not in self.available_tools:
            return f"Tool {tool_name} not found", ""
        
        try:
            result = await self.available_tools[tool_name](**kwargs)
            return result, tool_name
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}", tool_name

tool_manager = ToolManager()
