"""
EDISON Web UI Service
Serves the web interface for EDISON
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get repo root
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
WEB_DIR = REPO_ROOT / "web"

app = FastAPI(
    title="EDISON Web UI",
    description="Modern web interface for EDISON AI Assistant",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/")
async def root():
    """Serve the main web UI"""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        logger.error(f"index.html not found at {index_file}")
        return {"error": "Web UI not found"}
    return FileResponse(index_file)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "edison-web"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting EDISON Web UI from {WEB_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
