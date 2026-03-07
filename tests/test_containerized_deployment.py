from pathlib import Path


def test_docker_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "Dockerfile").exists()
    assert (root / "docker-compose.yml").exists()
    assert (root / "scripts" / "edison_docker.py").exists()


def test_compose_exposes_core_service():
    root = Path(__file__).resolve().parents[1]
    compose = (root / "docker-compose.yml").read_text()
    assert "edison-core" in compose
    assert "8811:8811" in compose
