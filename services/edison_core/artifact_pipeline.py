"""
Artifact-first output pipeline scaffolding.
"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json
import uuid

from .contracts import ArtifactGenerateRequest, ArtifactGenerateResponse


class ArtifactPipeline:
    def __init__(self, repo_root: Path, config: Dict[str, Any]):
        self.repo_root = repo_root
        self.config = config
        self.outputs_root = Path(config.get("artifacts", {}).get("root", "outputs")).resolve()
        if not self.outputs_root.is_absolute():
            self.outputs_root = (self.repo_root / self.outputs_root).resolve()
        self.outputs_root.mkdir(parents=True, exist_ok=True)

    def generate(self, req: ArtifactGenerateRequest) -> ArtifactGenerateResponse:
        artifact_id = f"art_{uuid.uuid4().hex[:8]}"
        project_dir = self.outputs_root / req.project_id / "artifacts" / artifact_id
        project_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "artifact_id": artifact_id,
            "project_id": req.project_id,
            "kind": req.kind,
            "prompt": req.prompt,
            "format": req.format,
            "metadata": req.metadata,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        meta_path = project_dir / "artifact.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        # Generate a minimal artifact file based on kind
        output_file = None
        content = None

        if req.kind == "document":
            output_file = project_dir / "document.md"
            content = f"# Document\n\nPrompt: {req.prompt}\n\n## Notes\n- Replace with generated content."
        elif req.kind == "code":
            repo_dir = project_dir / "repo"
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / "README.md").write_text(f"# Generated Repo\n\nPrompt: {req.prompt}\n")
            (repo_dir / "main.py").write_text("def main():\n    print('Hello from EDISON')\n\nif __name__ == '__main__':\n    main()\n")
            tests_dir = repo_dir / "tests"
            tests_dir.mkdir(exist_ok=True)
            (tests_dir / "test_main.py").write_text("def test_smoke():\n    assert True\n")
            output_file = repo_dir / "README.md"
        elif req.kind == "schema":
            output_file = project_dir / "schema.json"
            content = json.dumps({"title": "Schema", "description": req.prompt, "type": "object"}, indent=2)
        elif req.kind == "ui":
            output_file = project_dir / "Component.jsx"
            content = """export default function Component(){\n  return (\n    <div style={{padding:20}}><h1>EDISON UI</h1></div>\n  );\n}\n"""
        elif req.kind == "presentation":
            output_file = project_dir / "slides.md"
            content = f"# Slide 1\n\n{req.prompt}\n\n---\n\n# Slide 2\n\nKey points here."
        elif req.kind == "spreadsheet":
            output_file = project_dir / "data.csv"
            content = "col1,col2\nvalue1,value2\n"
        elif req.kind == "website":
            output_file = project_dir / "index.html"
            content = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>EDISON Site</title></head><body><h1>{req.prompt}</h1></body></html>"""

        if output_file and content is not None:
            output_file.write_text(content)
            meta["status"] = "generated"
            meta["output_file"] = str(output_file)
            meta_path.write_text(json.dumps(meta, indent=2))

        return ArtifactGenerateResponse(
            artifact_id=artifact_id,
            kind=req.kind,
            output_path=str(project_dir),
            status=meta["status"]
        )
