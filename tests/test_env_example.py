"""`.env.example` must describe the variables the code actually reads.

Documentation that names a setting nothing consumes is worse than none: it
fails silently and convincingly. `.env.example` documented `DATABASE_URL` for
months while `db/connection.py` read `DB_PATH`, so anyone who set the
documented variable pointed the database nowhere and the default kept working —
no error, no warning, just a setting that did nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTED = re.compile(r"^#?\s*([A-Z][A-Z0-9_]{2,})=", re.MULTILINE)
# Anything the code pulls out of the environment, however it is spelled.
CONSUMED = re.compile(r"""(?:getenv|environ(?:\.get)?)\(\s*["']([A-Z0-9_]+)["']""")


def _documented() -> set[str]:
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    return set(DOCUMENTED.findall(text))


def _consumed() -> set[str]:
    names: set[str] = set()
    for package in ("ai", "api", "db", "pipeline", "sources"):
        for path in (PROJECT_ROOT / package).rglob("*.py"):
            names |= set(CONSUMED.findall(path.read_text(encoding="utf-8")))
    return names


def test_every_documented_variable_is_actually_read():
    unused = _documented() - _consumed()
    assert not unused, (
        f".env.example documents {sorted(unused)}, which nothing reads. "
        "Setting them would do nothing, silently."
    )


def test_the_database_url_is_named_the_way_the_code_reads_it():
    """The specific case that motivated this file."""
    documented = _documented()
    assert "DB_PATH" in documented
    assert "DATABASE_URL" not in documented


def test_the_api_key_is_documented():
    """A guard on the guard: if the parsing above breaks, these sets go empty
    and every assertion here passes vacuously."""
    assert "GEMINI_API_KEY" in _documented()
    assert "GEMINI_API_KEY" in _consumed()
