import datetime
import logging
import chromadb
from config import settings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import String, Text, DateTime, Boolean
logger = logging.getLogger(__name__)
"""|=======DB=======|"""


engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


client = chromadb.PersistentClient(path="chroma_data")

"""
|=======================================================|
|=======================TABLES==========================|
|=======================================================|
"""


class Memory(Base):
    __tablename__ = "memory"

    id: Mapped[int] = mapped_column(primary_key=True)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    embedding_id: Mapped[str | None] = mapped_column(String, nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    is_tool_call: Mapped[bool] = mapped_column(Boolean, default=False)
    tool_name: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_result: Mapped[str | None] = mapped_column(Text, nullable=True)



async def get_session():
    async with async_session() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

