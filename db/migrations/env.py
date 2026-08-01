"""Alembic environment.

Two deviations from the generated template, both required here:

**The database URL comes from `db.connection`, not `alembic.ini`.** One source
of truth for where the database lives. The generated `sqlalchemy.url` is left
blank in the ini file so a stale value there cannot silently migrate the wrong
file — and so nothing resembling a connection string is ever committed.

**`render_as_batch=True`.** SQLite cannot `ALTER TABLE ... DROP COLUMN` or alter
a column type. Batch mode makes Alembic emit the copy-table-and-rename dance
instead. Without it, migrations author fine and then fail at `upgrade` — and
they fail on the *user's* database, not on the machine that wrote them.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from db.connection import DB_PATH
from db.models import Base

config = context.config
config.set_main_option("sqlalchemy.url", DB_PATH)

# Logging is configured from alembic.ini only when Alembic owns the process —
# i.e. the `alembic` CLI. `init_db()` runs migrations in-process on every
# pipeline start and sets this attribute to False.
#
# The default behaviour is a genuinely nasty bug rather than a style point:
# `fileConfig` disables every pre-existing logger and replaces the root
# handler, so the entire pipeline run after `init_db()` went completely silent.
# It still worked — it just produced no output, which is indistinguishable from
# having hung.
if config.attributes.get("configure_logger", True) and config.config_file_name:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        # Without this, autogenerate ignores type changes entirely and produces
        # an empty migration for a schema that really did change.
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
