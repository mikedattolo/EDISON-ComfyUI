#!/usr/bin/env python3
"""EDISON Docker helper CLI.

Usage examples:
  python scripts/edison_docker.py start
  python scripts/edison_docker.py stop
  python scripts/edison_docker.py init
  python scripts/edison_docker.py profiles --name python rust
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
CONFIG_FILE = REPO_ROOT / "config" / "edison.yaml"


def _load_container_env() -> Dict[str, str]:
    env: Dict[str, str] = os.environ.copy()
    if not CONFIG_FILE.exists():
        return env
    try:
        cfg = yaml.safe_load(CONFIG_FILE.read_text()) or {}
        containers = (cfg.get("edison", {}) or {}).get("containers", {}) or {}
        env.setdefault("FIREWALL_MODE", str(containers.get("firewall_mode", "allowlist")))
        cidrs = containers.get("allowed_egress_cidrs", ["0.0.0.0/0"])
        if isinstance(cidrs, list):
            env.setdefault("ALLOWED_EGRESS_CIDRS", ",".join(str(c) for c in cidrs))
    except Exception as e:
        print(f"Warning: failed to read container config: {e}")
    return env


def _run(cmd: List[str], env: Dict[str, str] | None = None):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def docker_compose_cmd(*args: str, env: Dict[str, str] | None = None):
    _run(["docker", "compose", "-f", str(COMPOSE_FILE), *args], env=env)


def cmd_init(_args: argparse.Namespace):
    for d in ["models", "data", "outputs", "config/integrations"]:
        (REPO_ROOT / d).mkdir(parents=True, exist_ok=True)
    print("Initialized Docker mount directories.")


def cmd_start(_args: argparse.Namespace):
    env = _load_container_env()
    docker_compose_cmd("up", "-d", "--build", env=env)


def cmd_stop(_args: argparse.Namespace):
    env = _load_container_env()
    docker_compose_cmd("down", env=env)


def cmd_logs(args: argparse.Namespace):
    env = _load_container_env()
    docker_compose_cmd("logs", "-f", "--tail", str(args.tail), env=env)


def cmd_profiles(args: argparse.Namespace):
    env = _load_container_env()
    profiles = args.name or []
    if not profiles:
        print("No profiles requested. Available: python, rust, node")
        return

    apt_map = {
        "python": ["python3-dev", "python3-venv"],
        "rust": ["rustc", "cargo"],
        "node": ["nodejs", "npm"],
    }
    pip_map = {
        "python": ["ipython", "pytest"],
        "rust": [],
        "node": [],
    }

    packages = []
    pip_packages = []
    for p in profiles:
        packages.extend(apt_map.get(p, []))
        pip_packages.extend(pip_map.get(p, []))

    if packages:
        docker_compose_cmd(
            "exec",
            "-T",
            "edison-core",
            "bash",
            "-lc",
            "apt-get update && apt-get install -y " + " ".join(sorted(set(packages))),
            env=env,
        )

    if pip_packages:
        docker_compose_cmd(
            "exec",
            "-T",
            "edison-core",
            "bash",
            "-lc",
            "pip install " + " ".join(sorted(set(pip_packages))),
            env=env,
        )

    print("Profiles installed:", ", ".join(profiles))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EDISON Docker environment manager")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Create host directories for Docker volumes").set_defaults(func=cmd_init)
    sub.add_parser("start", help="Build and start the stack").set_defaults(func=cmd_start)
    sub.add_parser("stop", help="Stop and remove the stack").set_defaults(func=cmd_stop)

    logs = sub.add_parser("logs", help="Tail compose logs")
    logs.add_argument("--tail", type=int, default=100)
    logs.set_defaults(func=cmd_logs)

    profiles = sub.add_parser("profiles", help="Install optional toolchains in container")
    profiles.add_argument("--name", nargs="+", help="Profiles to install (python rust node)")
    profiles.set_defaults(func=cmd_profiles)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return e.returncode


if __name__ == "__main__":
    sys.exit(main())
