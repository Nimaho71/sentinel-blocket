import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from database.models import Base

# By default, keep the sqlite DB in the root directory
DB_PATH = os.getenv("DB_PATH", "sqlite+aiosqlite:///sentinel_blocket.db")

# Create the AsyncEngine
engine = create_async_engine(
    DB_PATH,
    echo=False, # Set to True to see raw SQL queries in console (good for debugging)
)

# Create a configured "Session" class
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def init_db():
    """
    Creates all tables based on the models defined in models.py.
    Does nothing if tables already exist.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("Database initialized successfully.")

async def get_session() -> AsyncSession:
    """Dependency for obtaining an async database session."""
    async with AsyncSessionLocal() as session:
        yield session
