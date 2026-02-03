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

        return ArtifactGenerateResponse(
            artifact_id=artifact_id,
            kind=req.kind,
            output_path=str(project_dir),
            status="queued"
        )
