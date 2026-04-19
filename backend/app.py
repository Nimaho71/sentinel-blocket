from fastapi import FastAPI
from fastapi.responses import FileResponse
from backend.routes import router
import os

app = FastAPI(title="Sentinel-Blocket Deal Dashboard")

# Include our API routes
app.include_router(router)

# This route serves the actual HTML page
@app.get("/")
async def serve_index():
    # We point this to the new folder we created
    return FileResponse("frontend/templates/index.html")

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
