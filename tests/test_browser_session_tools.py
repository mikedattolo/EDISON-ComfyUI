from services.edison_core.browser_session import BrowserSessionManager


class _FakeLocator:
    def __init__(self, text="", count=1):
        self._text = text
        self._count = count

    @property
    def first(self):
        return self

    def count(self):
        return self._count

    def is_visible(self):
        return self._count > 0

    def inner_text(self, timeout=0):
        return self._text

    def click(self, timeout=0):
        return None

    def fill(self, value, timeout=0):
        self._text = value


class _FakePage:
    def __init__(self):
        self.url = "https://example.com"
        self._html = "<html><body><main><h1>Hello</h1><p>Example page</p></main></body></html>"
        self.keyboard = type("KB", (), {"type": lambda *a, **k: None, "press": lambda *a, **k: None})()
        self.mouse = type("MS", (), {"click": lambda *a, **k: None, "wheel": lambda *a, **k: None, "move": lambda *a, **k: None})()

    def title(self):
        return "Example"

    def screenshot(self, **kwargs):
        return b"img"

    def content(self):
        return self._html

    def locator(self, selector):
        if selector == "#missing":
            return _FakeLocator(count=0)
        if selector.startswith("text="):
            return _FakeLocator(text=selector.replace("text=", ""), count=1)
        return _FakeLocator(text="found", count=1)

    def goto(self, *args, **kwargs):
        return None

    def wait_for_timeout(self, ms):
        return None

    def set_viewport_size(self, viewport):
        return None


class _FakeContext:
    def __init__(self):
        self.page = _FakePage()

    def new_page(self):
        return self.page

    def close(self):
        return None


class _FakeBrowser:
    def new_context(self, **kwargs):
        return _FakeContext()


def _pw_run(func, *args, timeout=0):
    return func(*args)


def _emit(*args, **kwargs):
    return None


def _mgr():
    return BrowserSessionManager(
        pw_run=_pw_run,
        get_browser=lambda: _FakeBrowser(),
        emit_browser_view=_emit,
        default_allowed_hosts=["example.com"],
        allow_any_host=False,
    )


def test_browser_get_text_extracts_readable_text():
    mgr = _mgr()
    created = mgr.create_session("https://example.com")
    out = mgr.get_text(created["session_id"])
    assert out["text_length"] > 0
    assert "Hello" in out["readable_text"]


def test_browser_find_element_and_click_by_text():
    mgr = _mgr()
    created = mgr.create_session("https://example.com")
    sid = created["session_id"]

    found = mgr.find_element(sid, "button.submit")
    assert found["found"] is True
    assert found["count"] >= 1

    clicked = mgr.click_by_text(sid, "Submit")
    assert clicked["clicked_text"] == "Submit"


def test_browser_fill_form_returns_structured_data():
    mgr = _mgr()
    created = mgr.create_session("https://example.com")
    sid = created["session_id"]

    out = mgr.fill_form(sid, {"#email": "user@example.com", "#name": "User"})
    assert set(out["filled_fields"]) == {"#email", "#name"}
    assert out["session_id"] == sid
