"""
Knowledge ingestion connectors for EDISON.

These helpers ingest external knowledge sources into KnowledgeBase:
- Web/document URLs
- GitHub repositories
- arXiv papers
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_CODE_GLOBS = [
    "*.py",
    "*.js",
    "*.ts",
    "*.tsx",
    "*.jsx",
    "*.java",
    "*.go",
    "*.rs",
    "*.md",
    "*.txt",
    "*.yaml",
    "*.yml",
    "*.json",
    "*.toml",
    "*.ini",
    "*.cfg",
    "*.sh",
    "Dockerfile",
]


def _extract_text_from_html(html: str, max_chars: int = 60000) -> str:
    """Convert HTML into readable plain text without extra dependencies."""
    if not html:
        return ""

    # Remove script/style blocks first.
    cleaned = re.sub(r"<script[\\s\\S]*?</script>", " ", html, flags=re.IGNORECASE)
    cleaned = re.sub(r"<style[\\s\\S]*?</style>", " ", cleaned, flags=re.IGNORECASE)

    # Strip tags and normalize whitespace.
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"&amp;", "&", cleaned)
    cleaned = re.sub(r"&lt;", "<", cleaned)
    cleaned = re.sub(r"&gt;", ">", cleaned)
    cleaned = re.sub(r"\\s+", " ", cleaned).strip()

    if len(cleaned) > max_chars:
        return cleaned[:max_chars]
    return cleaned


def ingest_url(kb, url: str, title: str = "", category: str = "web_doc", timeout_s: int = 20) -> Dict:
    """Fetch a URL and ingest readable text into KnowledgeBase."""
    if not kb or not kb.is_ready():
        return {"ok": False, "error": "Knowledge base not ready"}

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"ok": False, "error": "URL must use http/https"}

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "EDISON-KnowledgeBot/1.0 (+https://github.com/mikedattolo/EDISON-ComfyUI)"
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            content_type = (resp.headers.get("Content-Type") or "").lower()

        # Decode best-effort.
        text_raw = raw.decode("utf-8", errors="ignore")

        if "html" in content_type or "<html" in text_raw.lower():
            text = _extract_text_from_html(text_raw)
        else:
            text = re.sub(r"\\s+", " ", text_raw).strip()

        if not text or len(text) < 100:
            return {"ok": False, "error": "Could not extract enough text from URL"}

        inferred_title = title or parsed.netloc
        chunks = kb.add_knowledge(
            text=text,
            source_type="web_doc",
            title=inferred_title,
            url=url,
            category=category,
            chunk_size=800,
        )

        return {
            "ok": True,
            "url": url,
            "title": inferred_title,
            "chunks_stored": chunks,
            "chars_ingested": len(text),
        }
    except Exception as e:
        logger.warning("URL ingestion failed for %s: %s", url, e)
        return {"ok": False, "error": str(e)}


def _matches_globs(name: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def ingest_github_repo(
    kb,
    repo_url: str,
    branch: str = "main",
    max_files: int = 160,
    max_file_bytes: int = 250_000,
    include_globs: Optional[List[str]] = None,
) -> Dict:
    """Shallow-clone a GitHub repo and ingest selected text/code files into KnowledgeBase."""
    if not kb or not kb.is_ready():
        return {"ok": False, "error": "Knowledge base not ready"}

    patterns = include_globs or DEFAULT_CODE_GLOBS

    with tempfile.TemporaryDirectory(prefix="edison_kb_repo_") as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        try:
            cmd = [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                repo_url,
                str(repo_dir),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if proc.returncode != 0:
                # Retry default branch when requested branch doesn't exist.
                retry = ["git", "clone", "--depth", "1", repo_url, str(repo_dir)]
                proc2 = subprocess.run(retry, capture_output=True, text=True, timeout=180)
                if proc2.returncode != 0:
                    return {
                        "ok": False,
                        "error": (proc.stderr or proc2.stderr or "git clone failed").strip(),
                    }

            stored_files = 0
            stored_chunks = 0
            skipped_files = 0
            repo_name = repo_dir.name

            for root, dirs, files in os.walk(repo_dir):
                # Skip noisy folders.
                dirs[:] = [
                    d
                    for d in dirs
                    if d not in {".git", "node_modules", "dist", "build", "venv", ".venv", "__pycache__"}
                ]

                for fname in files:
                    if stored_files >= max_files:
                        break

                    if not _matches_globs(fname, patterns):
                        continue

                    fpath = Path(root) / fname
                    try:
                        if fpath.stat().st_size > max_file_bytes:
                            skipped_files += 1
                            continue

                        content = fpath.read_text(encoding="utf-8", errors="ignore")
                        content = content.strip()
                        if len(content) < 80:
                            skipped_files += 1
                            continue

                        rel = fpath.relative_to(repo_dir).as_posix()
                        text = f"Repository: {repo_url}\\nPath: {rel}\\n\\n{content}"

                        chunks = kb.add_knowledge(
                            text=text,
                            source_type="github_repo",
                            title=f"{repo_name}:{rel}",
                            url=repo_url,
                            category="code",
                            chunk_size=900,
                        )
                        if chunks > 0:
                            stored_files += 1
                            stored_chunks += chunks
                        else:
                            skipped_files += 1
                    except Exception:
                        skipped_files += 1
                        continue

            return {
                "ok": True,
                "repo_url": repo_url,
                "branch": branch,
                "files_indexed": stored_files,
                "chunks_stored": stored_chunks,
                "skipped_files": skipped_files,
            }
        except Exception as e:
            logger.warning("GitHub ingestion failed for %s: %s", repo_url, e)
            return {"ok": False, "error": str(e)}


def ingest_arxiv(kb, query: str, max_results: int = 6) -> Dict:
    """Fetch arXiv Atom feed and ingest paper abstracts into KnowledgeBase."""
    if not kb or not kb.is_ready():
        return {"ok": False, "error": "Knowledge base not ready"}

    if not query or len(query.strip()) < 2:
        return {"ok": False, "error": "Query is required"}

    encoded = urllib.parse.quote_plus(query.strip())
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results={max(1, min(max_results, 20))}"

    try:
        with urllib.request.urlopen(url, timeout=25) as resp:
            xml_text = resp.read().decode("utf-8", errors="ignore")

        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        papers = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            link = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()

            authors = []
            for author in entry.findall("atom:author", ns):
                name = (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
                if name:
                    authors.append(name)

            if not title or not summary:
                continue

            papers.append(
                {
                    "title": title,
                    "summary": summary,
                    "url": link,
                    "published": published,
                    "authors": authors,
                }
            )

        stored = 0
        chunks_stored = 0
        for p in papers:
            text = (
                f"Title: {p['title']}\\n"
                f"Published: {p['published']}\\n"
                f"Authors: {', '.join(p['authors'])}\\n"
                f"Source: {p['url']}\\n\\n"
                f"Abstract: {p['summary']}"
            )
            chunks = kb.add_knowledge(
                text=text,
                source_type="arxiv",
                title=p["title"],
                url=p["url"],
                category="research",
                chunk_size=900,
            )
            if chunks > 0:
                stored += 1
                chunks_stored += chunks

        return {
            "ok": True,
            "query": query,
            "papers_fetched": len(papers),
            "papers_indexed": stored,
            "chunks_stored": chunks_stored,
        }
    except Exception as e:
        logger.warning("arXiv ingestion failed for %s: %s", query, e)
        return {"ok": False, "error": str(e)}
