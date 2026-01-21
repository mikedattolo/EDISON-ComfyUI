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


# Get repo root and log it
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
WEB_DIR = REPO_ROOT / "web"
logger.info(f"[DEBUG] __file__ = {__file__}")
logger.info(f"[DEBUG] REPO_ROOT = {REPO_ROOT}")
logger.info(f"[DEBUG] WEB_DIR = {WEB_DIR}")
logger.info(f"[DEBUG] WEB_DIR exists: {WEB_DIR.exists()}")
if not WEB_DIR.exists():
    logger.error(f"[DEBUG] WEB_DIR does not exist! Contents of REPO_ROOT: {list(REPO_ROOT.iterdir())}")

app = FastAPI(
    title="EDISON Web UI",
    description="Modern web interface for EDISON AI Assistant",
    version="1.0.0"
)

# Startup event for logging
@app.on_event("startup")
async def startup_event():
    logger.info(f"EDISON Web UI starting from {WEB_DIR}")
    logger.info(f"WEB_DIR exists: {WEB_DIR.exists()}")
    logger.info(f"Files in WEB_DIR: {list(WEB_DIR.glob('*.js'))}")
    logger.info(f"app_features.js exists: {(WEB_DIR / 'app_features.js').exists()}")

# Mount static files


@app.get("/")
async def root():
    """Serve the main web UI"""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        logger.error(f"index.html not found at {index_file}")
        return {"error": "Web UI not found"}
    return FileResponse(index_file)

@app.get("/styles.css")
async def styles():
    """Serve CSS file"""
    return FileResponse(WEB_DIR / "styles.css", media_type="text/css")

@app.get("/app_enhanced.js")
async def app_enhanced():
    """Serve main JS file"""
    return FileResponse(WEB_DIR / "app_enhanced.js", media_type="application/javascript")


@app.get("/app_features.js")
async def app_features():
    """Serve features JS file"""
    file_path = WEB_DIR / "app_features.js"
    logger.info(f"[DEBUG] Attempting to serve app_features.js from {file_path}")
    logger.info(f"[DEBUG] file_path.exists(): {file_path.exists()}")
    logger.info(f"[DEBUG] WEB_DIR contents: {[str(p) for p in WEB_DIR.glob('*')]}")
    if not file_path.exists():
        logger.error(f"[DEBUG] app_features.js not found at {file_path}")
        logger.error(f"[DEBUG] WEB_DIR contents: {[str(p) for p in WEB_DIR.glob('*')]}")
        logger.error(f"[DEBUG] REPO_ROOT contents: {[str(p) for p in REPO_ROOT.glob('*')]}")
        return {"error": "app_features.js not found", "path": str(file_path)}
    return FileResponse(file_path, media_type="application/javascript")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "edison-web"
    }


# Mount static files at the end to avoid conflicts with explicit routes
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting EDISON Web UI from {WEB_DIR}")
    logger.info(f"Registered routes: {[route.path for route in app.routes]}")
    logger.info(f"Files in WEB_DIR: {list(WEB_DIR.glob('*'))}")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
