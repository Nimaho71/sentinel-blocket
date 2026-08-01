"""Shared fixtures.

The database fixture is a real SQLite database, in memory. Not a mock — the
bugs being guarded against here (a query that failed to notice changed content,
a cascade that did not fire, a unique constraint that did not hold) live in the
interaction with the database, and a mocked session asserts only that the code
calls the methods the test expected it to call.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from db.crud import content_hash
from db.models import Ad, Base, CategoryEnum, StatusEnum


@pytest_asyncio.fixture
async def session_factory():
    """A fresh in-memory database per test.

    `StaticPool` with a shared connection is required: each new connection to
    `:memory:` gets its *own* empty database, so a pooled engine would hand the
    second session a database with no tables in it.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield async_sessionmaker(bind=engine, expire_on_commit=False)

    await engine.dispose()


def make_ad(
    external_id: str = "1",
    *,
    title: str = "RTX 3080 Gaming PC",
    description: str = "Fungerar bra, kvitto finns.",
    price_sek: int = 6000,
    item_type=None,
    status: StatusEnum = StatusEnum.ACTIVE,
    category: CategoryEnum = CategoryEnum.STATIONARY,
) -> Ad:
    from db.models import ItemTypeEnum

    now = datetime.now(timezone.utc)
    return Ad(
        source="blocket",
        external_id=external_id,
        url=f"https://blocket.se/annons/{external_id}",
        title=title,
        raw_description=description,
        price_sek=price_sek,
        content_hash=content_hash(title, description, price_sek),
        category=category,
        item_type=item_type or ItemTypeEnum.UNKNOWN,
        status=status,
        scraped_at=now,
        last_checked_at=now,
    )


@pytest.fixture
def ad_factory():
    return make_ad
