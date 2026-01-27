"""
EDISON Web UI Service
Serves the web interface for EDISON with SSL/HTTPS support
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import logging
import os

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
    logger.info(f"Serving app_features.js from {file_path}, exists: {file_path.exists()}")
    if not file_path.exists():
        logger.error(f"app_features.js not found at {file_path}")
        logger.error(f"WEB_DIR contents: {list(WEB_DIR.glob('*.js'))}")
    return FileResponse(file_path, media_type="application/javascript")

@app.get("/theme-device.js")
async def theme_device():
    """Serve theme and device detection JS file"""
    file_path = WEB_DIR / "theme-device.js"
    return FileResponse(file_path, media_type="application/javascript")

@app.get("/app_voice.js")
async def app_voice():
    """Serve voice mode JS file"""
    file_path = WEB_DIR / "app_voice.js"
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
    
    # Check for SSL certificates
    ssl_cert = Path("/opt/edison/ssl/cert.pem")
    ssl_key = Path("/opt/edison/ssl/key.pem")
    
    if ssl_cert.exists() and ssl_key.exists():
        logger.info("🔐 SSL certificates found - starting with HTTPS")
        logger.info(f"   Certificate: {ssl_cert}")
        logger.info(f"   Key: {ssl_key}")
        logger.info("🎙️  Voice mode will work with microphone access!")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8080,
            ssl_certfile=str(ssl_cert),
            ssl_keyfile=str(ssl_key),
            log_level="info"
        )
    else:
        logger.warning("⚠️  No SSL certificates found - running HTTP only")
        logger.warning(f"   Generate certificates with: scripts/generate_ssl_cert.sh")
        logger.warning("   Voice mode will NOT work without HTTPS!")
        uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
