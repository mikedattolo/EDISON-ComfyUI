"""
Artifact Detection System
Detects artifact-worthy content (HTML, React, SVG, etc.) in LLM responses
for live preview rendering in the frontend
"""

import re
import uuid
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ArtifactDetector:
    """Detect and extract artifacts from LLM responses"""
    
    # Patterns for different artifact types
    HTML_PATTERN = re.compile(r'<!DOCTYPE\s+html|<html[\s>]|<\!--.*?HTML.*?-->', re.IGNORECASE | re.DOTALL)
    SVG_PATTERN = re.compile(r'<svg[\s>].*?</svg>', re.IGNORECASE | re.DOTALL)
    REACT_PATTERN = re.compile(r'import\s+React|from\s+[\'"]react[\'"]|export\s+default\s+function', re.IGNORECASE)
    MERMAID_PATTERN = re.compile(r'```mermaid\s+(.*?)```', re.DOTALL)
    
    # Code block extraction
    CODE_BLOCK_PATTERN = re.compile(r'```(\w+)?\s*\n(.*?)```', re.DOTALL)
    
    @staticmethod
    def detect(response: str) -> Optional[Dict[str, Any]]:
        """
        Detect if response contains artifact-worthy content
        
        Args:
            response: LLM response text
            
        Returns:
            {
                "type": str,  # html, react, svg, mermaid
                "code": str,  # extracted code
                "id": str,    # unique artifact ID
                "title": str  # optional title
            } or None if no artifact detected
        """
        # Extract code blocks first
        code_blocks = ArtifactDetector.CODE_BLOCK_PATTERN.findall(response)
        
        for lang, code in code_blocks:
            lang = lang.lower() if lang else ""
            code = code.strip()
            
            # Check for HTML
            if lang in ["html", "htm"] or ArtifactDetector.HTML_PATTERN.search(code):
                return {
                    "type": "html",
                    "code": code,
                    "id": str(uuid.uuid4())[:8],
                    "title": "HTML Preview",
                    "language": "html"
                }
            
            # Check for React/JSX
            if lang in ["jsx", "react", "tsx"] or ArtifactDetector.REACT_PATTERN.search(code):
                return {
                    "type": "react",
                    "code": code,
                    "id": str(uuid.uuid4())[:8],
                    "title": "React Component",
                    "language": "jsx"
                }
            
            # Check for SVG
            if lang == "svg" or ArtifactDetector.SVG_PATTERN.search(code):
                return {
                    "type": "svg",
                    "code": code,
                    "id": str(uuid.uuid4())[:8],
                    "title": "SVG Graphic",
                    "language": "svg"
                }
            
            # Check for Mermaid
            if lang == "mermaid":
                return {
                    "type": "mermaid",
                    "code": code,
                    "id": str(uuid.uuid4())[:8],
                    "title": "Diagram",
                    "language": "mermaid"
                }
        
        # Check for inline SVG (not in code blocks)
        svg_match = ArtifactDetector.SVG_PATTERN.search(response)
        if svg_match:
            return {
                "type": "svg",
                "code": svg_match.group(0),
                "id": str(uuid.uuid4())[:8],
                "title": "SVG Graphic",
                "language": "svg"
            }
        
        return None
    
    @staticmethod
    def extract_title(response: str) -> str:
        """Extract title from response if present"""
        # Look for markdown headers
        title_match = re.search(r'^#+\s+(.+)$', response, re.MULTILINE)
        if title_match:
            return title_match.group(1).strip()
        
        # Look for HTML title
        title_match = re.search(r'<title>(.+?)</title>', response, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        
        return "Artifact"
    
    @staticmethod
    def should_render_artifact(artifact_type: str) -> bool:
        """Check if artifact type should be rendered in frontend"""
        renderable = ["html", "svg", "mermaid"]
        return artifact_type in renderable
    
    @staticmethod
    def get_sandbox_html(code: str, artifact_type: str) -> str:
        """
        Wrap code in sandboxed HTML for iframe rendering
        
        Args:
            code: The code to wrap
            artifact_type: Type of artifact (html, svg, etc.)
            
        Returns:
            Complete HTML document with CSP headers
        """
        if artifact_type == "svg":
            return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data: https:; style-src 'unsafe-inline';">
    <style>
        body {{ margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; }}
        svg {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    {code}
</body>
</html>"""
        
        elif artifact_type == "html":
            # Add CSP but allow more for HTML artifacts
            if not code.strip().startswith("<!DOCTYPE") and not code.strip().startswith("<html"):
                # Wrap fragment in full HTML
                return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; padding: 20px; line-height: 1.6; }}
    </style>
</head>
<body>
    {code}
</body>
</html>"""
            return code
        
        elif artifact_type == "mermaid":
            return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; }}
    </style>
</head>
<body>
    <div class="mermaid">
{code}
    </div>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>"""
        
        return code


def detect_artifact_in_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to detect artifacts
    
    Args:
        response: LLM response text
        
    Returns:
        Artifact dict or None
    """
    detector = ArtifactDetector()
    artifact = detector.detect(response)
    
    if artifact:
        # Extract title
        title = detector.extract_title(response)
        artifact["title"] = title
        
        # Check if renderable
        artifact["renderable"] = detector.should_render_artifact(artifact["type"])
        
        # Get sandbox HTML
        if artifact["renderable"]:
            artifact["html"] = detector.get_sandbox_html(artifact["code"], artifact["type"])
        
        logger.info(f"Detected {artifact['type']} artifact: {artifact['title']}")
    
    return artifact
