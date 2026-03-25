from services.edison_core.artifacts import detect_artifact_in_response


def test_detect_html_code_block_with_surrounding_prose():
    response = """Here is your landing page preview.

```html
<!DOCTYPE html>
<html>
<head>
  <style>body { background: #111; color: white; }</style>
</head>
<body>
  <main>Hello world</main>
</body>
</html>
```

You can customize the CTA later.
"""

    artifact = detect_artifact_in_response(response)

    assert artifact is not None
    assert artifact["type"] == "html"
    assert "<!DOCTYPE html>" in artifact["code"]
    assert artifact["renderable"] is True
    assert "<style>body { background: #111; color: white; }</style>" in artifact["html"]


def test_detect_javascript_code_block_and_wrap_for_preview():
    response = """```javascript
document.getElementById('app').innerHTML = '<h1>Pizza Funnel</h1><p>Live data</p>';
console.log('ready');
```
"""

    artifact = detect_artifact_in_response(response)

    assert artifact is not None
    assert artifact["type"] == "javascript"
    assert artifact["renderable"] is True
    assert "document.getElementById('app')" in artifact["code"]
    assert "<div id=\"app\"></div>" in artifact["html"]
    assert "<script>" in artifact["html"]


def test_detect_inline_svg_without_code_block():
    response = "Logo preview: <svg viewBox='0 0 10 10'><circle cx='5' cy='5' r='4'/></svg>"

    artifact = detect_artifact_in_response(response)

    assert artifact is not None
    assert artifact["type"] == "svg"
    assert artifact["renderable"] is True
    assert "<svg" in artifact["html"]