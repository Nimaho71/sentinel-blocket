"""Local development API.

Deliberately *not* deployed. The public site is a static React bundle reading a
pre-generated, redacted `deals.json` — no server, no hosted database, no API key
in a cloud environment, nothing to attack.

This app exists so the dashboard can be developed against live local data
without regenerating the snapshot on every change.

`PUBLIC_MODE` mirrors the published surface (top-N per category, redacted) so
that what you see while developing matches what ships. It defaults to off
because the whole point of running locally is unrestricted access to your own
data.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

PUBLIC_MODE = os.getenv("PUBLIC_MODE", "false").lower() in {"1", "true", "yes"}

app = FastAPI(
    title="Sentinel-Blocket (local)",
    description="Local development API. Not deployed — see docs/ARCHITECTURE.md.",
)

# Vite dev server only. Not a wildcard: this process can read the unredacted
# local database, so it should not be reachable from arbitrary origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "public_mode": PUBLIC_MODE}


if __name__ == "__main__":
    import uvicorn

    # Bound to localhost, not 0.0.0.0 — this serves unredacted data and has no
    # business being reachable from the local network.
    uvicorn.run(app, host="127.0.0.1", port=8000)
