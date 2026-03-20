"""Persistent Playwright browser sessions for sandbox/agent mode.

All page/context operations must run on the dedicated Playwright thread via
`pw_run` (the queue-backed helper from app.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Set
from urllib.parse import urlparse
import base64
import re
import time
import uuid

from bs4 import BeautifulSoup


class BrowserSessionError(Exception):
    """Domain-specific browser session errors with HTTP-like status codes."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class BrowserSession:
    context: Any
    page: Any
    created_ts: float
    last_used_ts: float
    viewport: Dict[str, int]
    allowed_hosts: Set[str]
    readable_text: str = ""


class BrowserSessionManager:
    """Manage persistent browser sessions for click/type/navigate workflows."""

    def __init__(
        self,
        pw_run: Callable[..., Any],
        get_browser: Callable[[], Any],
        emit_browser_view: Callable[..., None],
        default_allowed_hosts: Optional[list[str]] = None,
        allow_any_host: bool = False,
        user_agent: Optional[str] = None,
    ):
        self._pw_run = pw_run
        self._get_browser = get_browser
        self._emit_browser_view = emit_browser_view
        self._allow_any_host = bool(allow_any_host)
        self._default_allowed_hosts = self._normalize_allowed_hosts(default_allowed_hosts)
        self._user_agent = user_agent or (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122 Safari/537.36"
        )
        self._sessions: Dict[str, BrowserSession] = {}

    @staticmethod
    def _normalize_allowed_hosts(hosts: Optional[list[str]]) -> Set[str]:
        out: Set[str] = set()
        if not hosts:
            return out
        for host in hosts:
            if not isinstance(host, str):
                continue
            value = host.strip().lower()
            if not value:
                continue
            out.add(value.lstrip("."))
        return out

    @staticmethod
    def _host_matches(hostname: str, allowed: str) -> bool:
        if hostname == allowed:
            return True
        return hostname.endswith(f".{allowed}")

    def _resolve_allowed_hosts(self, session_allowed_hosts: Optional[list[str]]) -> Set[str]:
        # Optional restriction: if no allowlist configured, allow any host.
        merged = self._normalize_allowed_hosts(session_allowed_hosts)
        if merged:
            return merged
        return set(self._default_allowed_hosts)

    def _validate_url(self, raw_url: str, allowed_hosts: Set[str]) -> str:
        url = (raw_url or "").strip()
        if not url:
            raise BrowserSessionError("url is required", status_code=400)

        if "://" not in url:
            url = f"https://{url}"

        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        hostname = (parsed.hostname or "").lower()

        if scheme not in {"http", "https"}:
            raise BrowserSessionError("Only http(s) URLs are allowed", status_code=400)
        if not hostname:
            raise BrowserSessionError("Invalid URL host", status_code=400)

        if not self._allow_any_host and allowed_hosts:
            if not any(self._host_matches(hostname, allowed) for allowed in allowed_hosts):
                raise BrowserSessionError(f"Host '{hostname}' is not allowed", status_code=403)

        return url

    def _ensure_redirect_allowed(self, final_url: str, allowed_hosts: Set[str]):
        parsed = urlparse(final_url or "")
        scheme = (parsed.scheme or "").lower()
        hostname = (parsed.hostname or "").lower()
        if scheme not in {"http", "https"}:
            raise BrowserSessionError("Redirected to a non-http(s) URL", status_code=403)
        if not hostname:
            raise BrowserSessionError("Redirected to invalid host", status_code=403)
        if not self._allow_any_host and allowed_hosts:
            if not any(self._host_matches(hostname, allowed) for allowed in allowed_hosts):
                raise BrowserSessionError(f"Redirected to disallowed host '{hostname}'", status_code=403)

    @staticmethod
    def _extract_readable_text(html: str, max_chars: int = 5000) -> str:
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
                tag.decompose()
            # Prefer main-content containers when present.
            candidate = soup.select_one("main, article, [role='main']")
            text = (candidate.get_text("\n", strip=True) if candidate else soup.get_text("\n", strip=True)) or ""
        except Exception:
            text = re.sub(r"<[^>]+>", " ", html)

        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        compact = "\n".join(line for line in lines if line)
        return compact[:max_chars]

    def _snapshot(self, page: Any, width: int, height: int) -> Dict[str, Any]:
        title = page.title() or page.url
        img_bytes = page.screenshot(type="jpeg", quality=82, full_page=False, clip=None)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        readable_text = ""
        try:
            readable_text = self._extract_readable_text(page.content() or "")
        except Exception:
            readable_text = ""
        return {
            "url": page.url,
            "title": title,
            "screenshot_b64": b64,
            "width": int(width),
            "height": int(height),
            "readable_text": readable_text,
            "status": "done",
        }

    def _get_session(self, session_id: str) -> BrowserSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise BrowserSessionError("Session not found", status_code=404)
        return session

    def _emit_result(self, session_id: str, payload: Dict[str, Any], status: str = "done", error: Optional[str] = None):
        self._emit_browser_view(
            payload.get("url", ""),
            title=payload.get("title", payload.get("url", "")),
            screenshot_b64=payload.get("screenshot_b64"),
            status=status,
            error=error,
            session_id=session_id,
            width=payload.get("width"),
            height=payload.get("height"),
        )

    def create_session(
        self,
        start_url: str,
        width: int = 1280,
        height: int = 800,
        allowed_hosts: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        width = max(320, min(int(width), 3840))
        height = max(240, min(int(height), 2160))
        session_allowed_hosts = self._resolve_allowed_hosts(allowed_hosts)

        def _create(url: str, w: int, h: int, allowed: Set[str]):
            browser = self._get_browser()
            if browser is None:
                raise BrowserSessionError("Playwright browser is not available", status_code=503)

            validated = self._validate_url(url, allowed)
            context = browser.new_context(
                viewport={"width": w, "height": h},
                user_agent=self._user_agent,
            )
            page = context.new_page()

            try:
                page.goto(validated, timeout=20000, wait_until="domcontentloaded")
                page.wait_for_timeout(500)
                self._ensure_redirect_allowed(page.url, allowed)
            except Exception:
                context.close()
                raise

            now = time.time()
            session_id = f"sbx_{uuid.uuid4().hex[:12]}"
            self._sessions[session_id] = BrowserSession(
                context=context,
                page=page,
                created_ts=now,
                last_used_ts=now,
                viewport={"width": w, "height": h},
                allowed_hosts=set(allowed),
            )
            snap = self._snapshot(page, w, h)
            snap["session_id"] = session_id
            self._sessions[session_id].readable_text = snap.get("readable_text", "")
            return snap

        payload = self._pw_run(_create, start_url, width, height, session_allowed_hosts, timeout=30)
        self._emit_result(payload["session_id"], payload)
        return payload

    def navigate(self, session_id: str, url: str) -> Dict[str, Any]:
        def _navigate(sid: str, raw_url: str):
            session = self._get_session(sid)
            validated = self._validate_url(raw_url, session.allowed_hosts)
            session.page.goto(validated, timeout=20000, wait_until="domcontentloaded")
            session.page.wait_for_timeout(500)
            self._ensure_redirect_allowed(session.page.url, session.allowed_hosts)
            session.last_used_ts = time.time()
            snap = self._snapshot(session.page, session.viewport["width"], session.viewport["height"])
            session.readable_text = snap.get("readable_text", "")
            return snap

        self._emit_browser_view(url, title="Loading...", screenshot_b64=None, status="loading", session_id=session_id)
        payload = self._pw_run(_navigate, session_id, url, timeout=30)
        payload["session_id"] = session_id
        self._emit_result(session_id, payload)
        return payload

    def click(self, session_id: str, x: int, y: int, button: str = "left", click_count: int = 1) -> Dict[str, Any]:
        def _click(sid: str, px: int, py: int, btn: str, cc: int):
            session = self._get_session(sid)
            session.page.mouse.click(int(px), int(py), button=btn, click_count=max(1, int(cc)))
            session.page.wait_for_timeout(250)
            session.last_used_ts = time.time()
            snap = self._snapshot(session.page, session.viewport["width"], session.viewport["height"])
            session.readable_text = snap.get("readable_text", "")
            return snap

        payload = self._pw_run(_click, session_id, x, y, button, click_count, timeout=25)
        payload["session_id"] = session_id
        self._emit_result(session_id, payload)
        return payload

    def type(self, session_id: str, text: str, delay_ms: int = 10) -> Dict[str, Any]:
        def _type(sid: str, value: str, delay: int):
            session = self._get_session(sid)
            session.page.keyboard.type(value or "", delay=max(0, int(delay)))
            session.page.wait_for_timeout(200)
            session.last_used_ts = time.time()
            snap = self._snapshot(session.page, session.viewport["width"], session.viewport["height"])
            session.readable_text = snap.get("readable_text", "")
            return snap

        payload = self._pw_run(_type, session_id, text, delay_ms, timeout=25)
        payload["session_id"] = session_id
        self._emit_result(session_id, payload)
        return payload

    def keypress(self, session_id: str, key: str) -> Dict[str, Any]:
        def _key(sid: str, k: str):
            session = self._get_session(sid)
            session.page.keyboard.press(k)
            session.page.wait_for_timeout(250)
            session.last_used_ts = time.time()
            snap = self._snapshot(session.page, session.viewport["width"], session.viewport["height"])
            session.readable_text = snap.get("readable_text", "")
            return snap

        payload = self._pw_run(_key, session_id, key, timeout=25)
        payload["session_id"] = session_id
        self._emit_result(session_id, payload)
        return payload

    def scroll(self, session_id: str, delta_x: int, delta_y: int) -> Dict[str, Any]:
        def _scroll(sid: str, dx: int, dy: int):
            session = self._get_session(sid)
            session.page.mouse.wheel(int(dx), int(dy))
            session.page.wait_for_timeout(200)
            session.last_used_ts = time.time()
            snap = self._snapshot(session.page, session.viewport["width"], session.viewport["height"])
            session.readable_text = snap.get("readable_text", "")
            return snap

        payload = self._pw_run(_scroll, session_id, delta_x, delta_y, timeout=25)
        payload["session_id"] = session_id
        self._emit_result(session_id, payload)
        return payload

    def move_mouse(self, session_id: str, x: int, y: int) -> Dict[str, Any]:
        def _move(sid: str, px: int, py: int):
            session = self._get_session(sid)
            session.page.mouse.move(int(px), int(py))
            session.last_used_ts = time.time()
            snap = self._snapshot(session.page, session.viewport["width"], session.viewport["height"])
            session.readable_text = snap.get("readable_text", "")
            return snap

        payload = self._pw_run(_move, session_id, x, y, timeout=25)
        payload["session_id"] = session_id
        self._emit_result(session_id, payload)
        return payload

    def set_viewport(self, session_id: str, width: int, height: int) -> Dict[str, Any]:
        width = max(320, min(int(width), 3840))
        height = max(240, min(int(height), 2160))

        def _set_viewport(sid: str, w: int, h: int):
            session = self._get_session(sid)
            session.page.set_viewport_size({"width": w, "height": h})
            session.viewport = {"width": w, "height": h}
            session.last_used_ts = time.time()
            snap = self._snapshot(session.page, w, h)
            session.readable_text = snap.get("readable_text", "")
            return snap

        payload = self._pw_run(_set_viewport, session_id, width, height, timeout=25)
        payload["session_id"] = session_id
        self._emit_result(session_id, payload)
        return payload

    def screenshot(self, session_id: str) -> Dict[str, Any]:
        def _screenshot(sid: str):
            session = self._get_session(sid)
            session.last_used_ts = time.time()
            snap = self._snapshot(session.page, session.viewport["width"], session.viewport["height"])
            session.readable_text = snap.get("readable_text", "")
            return snap

        payload = self._pw_run(_screenshot, session_id, timeout=25)
        payload["session_id"] = session_id
        self._emit_result(session_id, payload)
        return payload

    def observe(self, session_id: str) -> Dict[str, Any]:
        return self.screenshot(session_id)

    def get_text(self, session_id: str) -> Dict[str, Any]:
        def _get_text(sid: str):
            session = self._get_session(sid)
            html = session.page.content() or ""
            readable_text = self._extract_readable_text(html)
            session.readable_text = readable_text
            session.last_used_ts = time.time()
            snap = self._snapshot(session.page, session.viewport["width"], session.viewport["height"])
            return {
                **snap,
                "session_id": sid,
                "readable_text": readable_text,
                "text_length": len(readable_text),
            }

        payload = self._pw_run(_get_text, session_id, timeout=25)
        self._emit_result(session_id, payload)
        return payload

    def find_element(self, session_id: str, selector: str) -> Dict[str, Any]:
        def _find(sid: str, css_selector: str):
            session = self._get_session(sid)
            loc = session.page.locator(css_selector).first
            count = session.page.locator(css_selector).count()
            found = count > 0
            visible = bool(loc.is_visible()) if found else False
            text = (loc.inner_text(timeout=1500).strip() if found else "")
            if len(text) > 200:
                text = text[:200]
            session.last_used_ts = time.time()
            snap = self._snapshot(session.page, session.viewport["width"], session.viewport["height"])
            session.readable_text = snap.get("readable_text", "")
            return {
                **snap,
                "session_id": sid,
                "selector": css_selector,
                "found": found,
                "count": int(count),
                "visible": visible,
                "text": text,
            }

        if not (selector or "").strip():
            raise BrowserSessionError("selector is required", status_code=400)
        payload = self._pw_run(_find, session_id, selector, timeout=25)
        self._emit_result(session_id, payload)
        return payload

    def click_by_text(self, session_id: str, text: str) -> Dict[str, Any]:
        def _click_by_text(sid: str, target_text: str):
            session = self._get_session(sid)
            loc = session.page.locator(f"text={target_text}").first
            if loc.count() <= 0:
                raise BrowserSessionError(f"No element found with text '{target_text}'", status_code=404)
            loc.click(timeout=5000)
            session.page.wait_for_timeout(250)
            session.last_used_ts = time.time()
            snap = self._snapshot(session.page, session.viewport["width"], session.viewport["height"])
            session.readable_text = snap.get("readable_text", "")
            return {
                **snap,
                "session_id": sid,
                "clicked_text": target_text,
            }

        value = (text or "").strip()
        if not value:
            raise BrowserSessionError("text is required", status_code=400)
        payload = self._pw_run(_click_by_text, session_id, value, timeout=25)
        self._emit_result(session_id, payload)
        return payload

    def fill_form(self, session_id: str, fields: Dict[str, str]) -> Dict[str, Any]:
        def _fill_form(sid: str, form_fields: Dict[str, str]):
            session = self._get_session(sid)
            filled = []
            for selector, value in (form_fields or {}).items():
                if not isinstance(selector, str) or not selector.strip():
                    continue
                loc = session.page.locator(selector).first
                if loc.count() <= 0:
                    continue
                loc.fill(str(value or ""), timeout=5000)
                filled.append(selector)
            session.page.wait_for_timeout(250)
            session.last_used_ts = time.time()
            snap = self._snapshot(session.page, session.viewport["width"], session.viewport["height"])
            session.readable_text = snap.get("readable_text", "")
            return {
                **snap,
                "session_id": sid,
                "filled_fields": filled,
                "requested_fields": list((form_fields or {}).keys()),
            }

        if not isinstance(fields, dict) or not fields:
            raise BrowserSessionError("fields must be a non-empty object", status_code=400)
        payload = self._pw_run(_fill_form, session_id, fields, timeout=30)
        self._emit_result(session_id, payload)
        return payload

    def close_session(self, session_id: str) -> Dict[str, Any]:
        def _close(sid: str):
            session = self._sessions.pop(sid, None)
            if session is None:
                raise BrowserSessionError("Session not found", status_code=404)
            try:
                session.context.close()
            except Exception:
                pass
            return {"session_id": sid, "status": "closed"}

        payload = self._pw_run(_close, session_id, timeout=15)
        self._emit_browser_view("", title="Browser session closed", screenshot_b64=None, status="closed", session_id=session_id)
        return payload

    def cleanup_expired_sessions(self, ttl_seconds: int = 900) -> Dict[str, Any]:
        ttl = max(30, int(ttl_seconds))

        def _cleanup(ttl_val: int):
            now = time.time()
            expired_ids = [
                sid for sid, s in self._sessions.items()
                if (now - float(s.last_used_ts)) > ttl_val
            ]
            for sid in expired_ids:
                session = self._sessions.pop(sid, None)
                if session is None:
                    continue
                try:
                    session.context.close()
                except Exception:
                    pass
            return {"expired": len(expired_ids), "session_ids": expired_ids}

        return self._pw_run(_cleanup, ttl, timeout=20)
