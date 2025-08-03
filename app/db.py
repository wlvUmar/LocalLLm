import chromadb
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:getout04@localhost:5433/postgres")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def get_session():
    async with async_session() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

class Memory(Base):
    __tablename__ = "memory"

    id = Column(Integer, primary_key=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    embedding_id = Column(String, nullable=True)
    conversation_id = Column(String, nullable=True)
    is_tool_call = Column(Boolean, default=False)
    tool_name = Column(String, nullable=True)
    tool_result = Column(Text, nullable=True)

class ConversationState(Base):
    __tablename__ = "conversation_state"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, unique=True, nullable=False)
    current_step = Column(String, nullable=True)
    context_summary = Column(Text, nullable=True)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow)

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection("chat_memory")

