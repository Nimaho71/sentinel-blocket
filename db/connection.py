"""Engine and session management.

### Why schema creation goes through Alembic rather than `create_all`

`Base.metadata.create_all` builds whatever the models currently say, which is
correct exactly once — on an empty database. Add a column later and it does
nothing at all to an existing file: no error, no warning, just a schema that
silently disagrees with the code until something reads the missing column.

So real databases are built and evolved by migrations. `create_all` survives in
one place only — the in-memory database the tests build from scratch each run,
where there is no history to preserve and speed matters. `tests/test_migrations.py`
asserts the two paths agree, so that shortcut cannot drift into a lie.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The local database holds unredacted listing data and is never committed
# (`*.db` is gitignored). The published artifact is the redacted
# web/public/deals.json produced by the publish step.
DB_PATH = os.getenv("DB_PATH", f"sqlite+aiosqlite:///{PROJECT_ROOT / 'sentinel_blocket.db'}")

engine = create_async_engine(DB_PATH, echo=False)

AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


class UnmanagedDatabaseError(RuntimeError):
    """The database predates Alembic and cannot be migrated automatically."""


def check_is_migratable(db_url: str | None = None) -> None:
    """Refuse to run against a pre-Alembic database, with an explanation.

    A database created by the old `create_all` has the tables but no
    `alembic_version` row, so Alembic believes it is empty and tries to
    `CREATE TABLE ads` — which fails several hundred lines of SQL later with
    "table already exists". That error describes the symptom and hides the
    cause, so it is caught here instead.

    Nothing is deleted or altered automatically. The old database is the user's,
    and the two possible intents — throw it away, or keep it — are not ours to
    guess between.
    """
    db_url = db_url or DB_PATH
    if not db_url.startswith("sqlite"):
        return

    _, _, path = db_url.partition(":///")
    if not path or path == ":memory:" or not Path(path).exists():
        return

    import sqlite3

    with sqlite3.connect(path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    if tables and "alembic_version" not in tables:
        raise UnmanagedDatabaseError(
            f"{path} was created before this project used migrations, so its schema "
            "predates the current models and Alembic cannot upgrade it.\n\n"
            "Its listing data is re-scrapeable in minutes, so the simplest fix is to "
            "move it aside and let a fresh one be built:\n"
            f"    mv {path} {path}.pre-alembic.bak\n\n"
            "If you would rather keep it, stamp it as the baseline and write a "
            "migration for the difference by hand:\n"
            "    python -m alembic stamp base"
        )


async def init_db() -> None:
    """Bring the database up to the latest migration.

    Runs synchronously inside a worker thread: Alembic's runner drives its own
    event loop, and calling it from within a running loop would deadlock.
    """
    import asyncio

    from alembic import command
    from alembic.config import Config

    check_is_migratable()

    def upgrade() -> None:
        config = Config(str(PROJECT_ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(PROJECT_ROOT / "db" / "migrations"))
        # Leave the caller's logging alone — see the note in migrations/env.py.
        config.attributes["configure_logger"] = False
        command.upgrade(config, "head")

    await asyncio.to_thread(upgrade)
    logger.info("Database schema is up to date.")


async def get_session():
    """FastAPI dependency yielding a session."""
    async with AsyncSessionLocal() as session:
        yield session
