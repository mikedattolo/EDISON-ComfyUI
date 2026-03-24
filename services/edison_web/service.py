"""
EDISON Web UI Service
Serves the web interface for EDISON with SSL/HTTPS support.
Reverse-proxies /api/* requests to the Edison Core API on localhost:8811
so the browser only needs to trust a single HTTPS origin.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.requests import Request
from pathlib import Path
import logging
import os

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get repo root
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
WEB_DIR = REPO_ROOT / "web"

# Core API base URL (internal, HTTP, localhost only)
CORE_API_BASE = os.environ.get("EDISON_CORE_URL", "http://127.0.0.1:8811")

# ── Persistent httpx client for reverse proxy ─────────────────────────
_http_client: "httpx.AsyncClient | None" = None

def _get_client() -> "httpx.AsyncClient":
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            base_url=CORE_API_BASE,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            follow_redirects=True,
        )
    return _http_client

@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info(f"EDISON Web UI starting from {WEB_DIR}")
    if HAS_HTTPX:
        logger.info(f"✓ Reverse proxy enabled → {CORE_API_BASE}")
    else:
        logger.warning("⚠ httpx not installed — /api/* proxy disabled. pip install httpx")
    yield
    # Shutdown: close httpx client
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None
    logger.info("EDISON Web UI shut down")

app = FastAPI(
    title="EDISON Web UI",
    description="Modern web interface for EDISON AI Assistant",
    version="1.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════
# REVERSE PROXY  /api/*  →  Core API (localhost:8811)
# ═══════════════════════════════════════════════════════════════════════

# Headers to NOT forward (hop-by-hop or set by proxy)
_HOP_HEADERS = frozenset([
    "host", "transfer-encoding", "connection", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade",
])

@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def reverse_proxy(request: Request, path: str):
    """Proxy all /api/* requests to Edison Core on localhost."""
    if not HAS_HTTPX:
        return Response(content='{"error":"httpx not installed"}', status_code=503,
                        media_type="application/json")

    client = _get_client()
    url = f"/{path}"
    params = dict(request.query_params)
    body = await request.body()

    # Forward headers, strip hop-by-hop
    fwd = {}
    for k, v in request.headers.items():
        if k.lower() not in _HOP_HEADERS:
            fwd[k] = v

    is_stream = ("stream" in path.lower() or
                 request.headers.get("accept") == "text/event-stream")

    try:
        if is_stream:
            # Build + send with streaming enabled
            req = client.build_request(
                method=request.method, url=url, params=params,
                headers=fwd, content=body if body else None,
            )
            resp = await client.send(req, stream=True)

            async def stream_body():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()

            resp_headers = dict(resp.headers)
            resp_headers.pop("transfer-encoding", None)
            resp_headers.pop("content-length", None)
            return StreamingResponse(
                stream_body(),
                status_code=resp.status_code,
                headers=resp_headers,
            )
        else:
            resp = await client.request(
                method=request.method, url=url, params=params,
                headers=fwd, content=body if body else None,
            )
            resp_headers = dict(resp.headers)
            for h in ("transfer-encoding", "content-encoding", "content-length"):
                resp_headers.pop(h, None)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
            )
    except httpx.ConnectError:
        return Response(
            content='{"error":"Edison Core API is not running (localhost:8811)"}',
            status_code=502, media_type="application/json",
        )
    except httpx.ReadTimeout:
        return Response(
            content='{"error":"Edison Core API timed out"}',
            status_code=504, media_type="application/json",
        )


# ═══════════════════════════════════════════════════════════════════════
# STATIC FILE ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Serve the main web UI"""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        logger.error(f"index.html not found at {index_file}")
        return {"error": "Web UI not found"}
    return FileResponse(index_file)


@app.get("/video-editor")
async def video_editor_page():
    """Serve dedicated video editor studio page."""
    page = WEB_DIR / "video_editor.html"
    if not page.exists():
        logger.error(f"video_editor.html not found at {page}")
        return {"error": "Video editor UI not found"}
    return FileResponse(page)


@app.get("/branding")
async def branding_page():
    """Serve dedicated branding/file explorer page."""
    page = WEB_DIR / "branding.html"
    if not page.exists():
        logger.error(f"branding.html not found at {page}")
        return {"error": "Branding UI not found"}
    return FileResponse(page)


@app.get("/projects")
async def projects_page():
    """Serve dedicated client and project workspace page."""
    page = WEB_DIR / "projects.html"
    if not page.exists():
        logger.error(f"projects.html not found at {page}")
        return {"error": "Projects UI not found"}
    return FileResponse(page)


@app.get("/connectors")
async def connectors_page():
    """Serve dedicated connectors management page."""
    page = WEB_DIR / "connectors.html"
    if not page.exists():
        logger.error(f"connectors.html not found at {page}")
        return {"error": "Connectors UI not found"}
    return FileResponse(page)


@app.get("/printing")
async def printing_page():
    """Serve dedicated 3D printing page."""
    page = WEB_DIR / "printing.html"
    if not page.exists():
        logger.error(f"printing.html not found at {page}")
        return {"error": "Printing UI not found"}
    return FileResponse(page)

@app.get("/styles.css")
async def styles():
    """Serve CSS file"""
    return FileResponse(
        WEB_DIR / "styles.css",
        media_type="text/css",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@app.get("/app_enhanced.js")
async def app_enhanced():
    """Serve main JS file"""
    return FileResponse(
        WEB_DIR / "app_enhanced.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@app.get("/app_features.js")
async def app_features():
    """Serve features JS file"""
    return FileResponse(
        WEB_DIR / "app_features.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@app.get("/theme-device.js")
async def theme_device():
    """Serve theme and device detection JS file"""
    return FileResponse(WEB_DIR / "theme-device.js", media_type="application/javascript")

@app.get("/app_new_features.js")
async def app_new_features():
    """Serve new features JS file"""
    return FileResponse(
        WEB_DIR / "app_new_features.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@app.get("/voice_agent_live.js")
async def voice_agent_live():
    """Serve voice assistant & agent live view JS"""
    return FileResponse(
        WEB_DIR / "voice_agent_live.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@app.get("/gallery.js")
async def gallery():
    """Serve gallery JS file"""
    return FileResponse(
        WEB_DIR / "gallery.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "edison-web",
        "proxy": HAS_HTTPX,
        "core_url": CORE_API_BASE,
    }


# Mount static files at the end to avoid conflicts with explicit routes
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting EDISON Web UI from {WEB_DIR}")
    logger.info(f"Registered routes: {[route.path for route in app.routes]}")

    # HTTPS: auto-detect self-signed certs
    cert_dir = REPO_ROOT / "certs"
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    ssl_kwargs = {}
    if cert_file.exists() and key_file.exists():
        ssl_kwargs["ssl_certfile"] = str(cert_file)
        ssl_kwargs["ssl_keyfile"] = str(key_file)
        logger.info(f"🔒 HTTPS enabled with certs from {cert_dir}")
    else:
        logger.warning(
            "⚠ No certs found — running HTTP only. "
            "Voice input requires HTTPS. Run: bash scripts/generate_certs.sh"
        )

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info", **ssl_kwargs)
