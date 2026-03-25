#!/usr/bin/env python3
"""Bootstrap EDISON media models and ComfyUI nodes from the current repo.

This script automates the parts of setup that are stable and repo-known:
  - Vision models matching the current config
  - FLUX image generation stack for ComfyUI
  - Video helper nodes and baseline video assets
  - Optional 3D scaffolding / custom repo hooks

It is intentionally repo-relative so it works from a checkout like:
  /workspaces/EDISON-ComfyUI
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_DIR = REPO_ROOT / "ComfyUI"
COMFYUI_MODELS = COMFYUI_DIR / "models"
COMFYUI_NODES = COMFYUI_DIR / "custom_nodes"
LLM_MODELS_DIR = REPO_ROOT / "models" / "llm"


class SetupError(RuntimeError):
    pass


def installer_python() -> str:
    candidates = [
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / "venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable


def installer_pip_command() -> list[str]:
    python_bin = installer_python()
    if python_bin == sys.executable and sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        raise SetupError(
            "No repo virtualenv detected for Python package installs. Run setup_edison.sh first or create /opt/edison/.venv."
        )
    return [python_bin, "-m", "pip"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install EDISON media models and ComfyUI add-ons from this repo checkout."
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
    parser.add_argument("--yes", action="store_true", help="Run non-interactively and accept default bundles.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading or cloning anything.")
    parser.add_argument("--skip-vision", action="store_true", help="Do not install vision models.")
    parser.add_argument("--skip-image", action="store_true", help="Do not install FLUX image models.")
    parser.add_argument("--skip-video", action="store_true", help="Do not install video nodes or baseline video assets.")
    parser.add_argument("--include-3d", action="store_true", help="Prepare 3D directories and optional custom 3D repo/model hooks.")
    parser.add_argument(
        "--flux-variant",
        choices=["dev", "schnell"],
        default="dev",
        help="Choose the FLUX checkpoint to install.",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def ask_yes_no(prompt: str, default: bool = True, assume_yes: bool = False) -> bool:
    if assume_yes:
        return default
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix} ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def ensure_dirs(paths: Iterable[Path], dry_run: bool) -> None:
    for path in paths:
        if dry_run:
            print(f"DRY RUN mkdir -p {path}")
        else:
            path.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], dry_run: bool, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    rendered = " ".join(cmd)
    if cwd:
        rendered = f"(cd {cwd} && {rendered})"
    if dry_run:
        print(f"DRY RUN {rendered}")
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def pip_install(args: list[str], dry_run: bool) -> None:
    cmd = installer_pip_command() + args
    run(cmd, dry_run=dry_run)


def wget_download(url: str, destination: Path, dry_run: bool, auth_token: str | None = None) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"SKIP existing {destination}")
        return
    cmd = ["wget", "-c", "--progress=bar:force:noscroll", "-O", str(destination)]
    if auth_token:
        cmd.insert(1, f"--header=Authorization: Bearer {auth_token}")
    cmd.append(url)
    run(cmd, dry_run=dry_run)


def hf_resolve_url(repo_id: str, filename: str) -> str:
    encoded_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    encoded_file = "/".join(urllib.parse.quote(part, safe="") for part in filename.split("/"))
    return f"https://huggingface.co/{encoded_repo}/resolve/main/{encoded_file}?download=true"


def hf_download(
    repo_id: str,
    filename: str,
    target_dir: Path,
    dry_run: bool,
    token: str | None,
    rename_to: str | None = None,
    require_token: bool = False,
) -> None:
    output_name = rename_to or filename
    final_path = target_dir / output_name
    if final_path.exists() and final_path.stat().st_size > 0:
        print(f"SKIP existing {final_path}")
        return
    if require_token and not token:
        raise SetupError(
            f"Hugging Face token required for {repo_id}/{filename}. Set HF_TOKEN or pass --hf-token."
        )
    ensure_dirs([target_dir], dry_run=dry_run)
    wget_download(hf_resolve_url(repo_id, filename), final_path, dry_run=dry_run, auth_token=token)


def _github_archive_urls(repo_url: str) -> list[str]:
    parsed = urllib.parse.urlparse(repo_url)
    if parsed.netloc.lower() != "github.com":
        return []
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        return []
    owner, repo = parts[0], parts[1].removesuffix(".git")
    return [
        f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/main",
        f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/master",
    ]


def _download_and_extract_zip(url: str, destination: Path, dry_run: bool) -> bool:
    if dry_run:
        print(f"DRY RUN download and extract {url} -> {destination}")
        return True

    ensure_dirs([destination.parent], dry_run=False)
    with tempfile.TemporaryDirectory(prefix="edison_repo_zip_") as temp_dir:
        zip_path = Path(temp_dir) / "repo.zip"
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                top_level = None
                for member in archive.namelist():
                    if member.strip():
                        top_level = member.split("/", 1)[0]
                        break
                archive.extractall(temp_dir)
            if not top_level:
                return False
            extracted_root = Path(temp_dir) / top_level
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(extracted_root), str(destination))
            return True
        except Exception:
            return False


def clone_or_update(repo_url: str, destination: Path, dry_run: bool) -> None:
    git_env = os.environ.copy()
    git_env["GIT_TERMINAL_PROMPT"] = "0"
    git_env["GIT_ASKPASS"] = "echo"

    if destination.exists():
        if (destination / ".git").exists():
            run(["git", "-c", "credential.helper=", "pull", "--ff-only"], cwd=destination, dry_run=dry_run, env=git_env)
        else:
            print(f"SKIP existing non-git directory {destination}")
        return

    ensure_dirs([destination.parent], dry_run=dry_run)
    try:
        run(["git", "-c", "credential.helper=", "clone", repo_url, str(destination)], dry_run=dry_run, env=git_env)
        return
    except subprocess.CalledProcessError:
        archive_urls = _github_archive_urls(repo_url)
        for url in archive_urls:
            if _download_and_extract_zip(url, destination, dry_run=dry_run):
                print(f"Downloaded archive fallback for {repo_url}")
                return
        raise


def install_vision_bundle(args: argparse.Namespace, summary: list[str]) -> None:
    print_header("Installing Vision Models")
    ensure_dirs([LLM_MODELS_DIR], dry_run=args.dry_run)

    # Default vision model from config: LLaVA 1.6 Mistral 7B GGUF + mmproj.
    wget_download(
        "https://huggingface.co/mradermacher/llava-v1.6-mistral-7b-GGUF/resolve/main/llava-v1.6-mistral-7b.Q4_K_M.gguf",
        LLM_MODELS_DIR / "llava-v1.6-mistral-7b-q4_k_m.gguf",
        dry_run=args.dry_run,
    )
    wget_download(
        "https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf",
        LLM_MODELS_DIR / "llava-v1.6-mistral-7b-mmproj-q4_0.gguf",
        dry_run=args.dry_run,
    )

    # Optional better vision/code model matching current config.
    try:
        hf_download(
            "bartowski/Qwen2-VL-7B-Instruct-GGUF",
            "Qwen2-VL-7B-Instruct-Q4_K_M.gguf",
            LLM_MODELS_DIR,
            dry_run=args.dry_run,
            token=args.hf_token,
            require_token=True,
        )
        hf_download(
            "bartowski/Qwen2-VL-7B-Instruct-GGUF",
            "mmproj-model-f16.gguf",
            LLM_MODELS_DIR,
            dry_run=args.dry_run,
            token=args.hf_token,
            rename_to="mmproj-Qwen2-VL-7B-Instruct-f16.gguf",
            require_token=True,
        )
        summary.append("vision: installed llava + qwen2-vl vision assets")
    except SetupError as exc:
        print(f"NOTE {exc}")
        summary.append("vision: installed llava; skipped qwen2-vl because no HF token was provided")


def install_image_bundle(args: argparse.Namespace, summary: list[str]) -> None:
    print_header("Installing FLUX Image Stack")
    ensure_dirs(
        [
            COMFYUI_MODELS / "checkpoints",
            COMFYUI_MODELS / "vae",
            COMFYUI_MODELS / "clip",
            COMFYUI_NODES,
        ],
        dry_run=args.dry_run,
    )

    flux_repo = "black-forest-labs/FLUX.1-dev" if args.flux_variant == "dev" else "black-forest-labs/FLUX.1-schnell"
    flux_file = "flux1-dev.safetensors" if args.flux_variant == "dev" else "flux1-schnell.safetensors"

    hf_download(
        flux_repo,
        flux_file,
        COMFYUI_MODELS / "checkpoints",
        dry_run=args.dry_run,
        token=args.hf_token,
        require_token=True,
    )
    hf_download(
        flux_repo,
        "ae.safetensors",
        COMFYUI_MODELS / "vae",
        dry_run=args.dry_run,
        token=args.hf_token,
        require_token=True,
    )
    hf_download(
        "comfyanonymous/flux_text_encoders",
        "clip_l.safetensors",
        COMFYUI_MODELS / "clip",
        dry_run=args.dry_run,
        token=args.hf_token,
    )
    encoder_name = "t5xxl_fp8_e4m3fn.safetensors" if args.flux_variant == "dev" else "t5xxl_fp16.safetensors"
    hf_download(
        "comfyanonymous/flux_text_encoders",
        encoder_name,
        COMFYUI_MODELS / "clip",
        dry_run=args.dry_run,
        token=args.hf_token,
    )

    # Install FLUX fill for editing/inpainting when token is available.
    if args.hf_token:
        try:
            hf_download(
                "black-forest-labs/FLUX.1-Fill-dev",
                "flux1-fill-dev.safetensors",
                COMFYUI_MODELS / "checkpoints",
                dry_run=args.dry_run,
                token=args.hf_token,
                require_token=True,
            )
        except subprocess.CalledProcessError:
            print(
                "NOTE skipping FLUX Fill because Hugging Face returned access denied. "
                "Accept the model license on Hugging Face or use a token with access if you want inpainting/editing support."
            )
    else:
        print("NOTE skipping FLUX Fill because no HF token was provided")

    summary.append(f"image: installed FLUX {args.flux_variant} stack using built-in ComfyUI loaders")


def install_video_bundle(args: argparse.Namespace, summary: list[str]) -> None:
    print_header("Installing Video Stack")
    ensure_dirs(
        [
            COMFYUI_NODES,
            COMFYUI_MODELS / "animatediff_models",
            COMFYUI_MODELS / "checkpoints",
        ],
        dry_run=args.dry_run,
    )

    clone_or_update(
        "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git",
        COMFYUI_NODES / "ComfyUI-AnimateDiff-Evolved",
        dry_run=args.dry_run,
    )
    clone_or_update(
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        COMFYUI_NODES / "ComfyUI-VideoHelperSuite",
        dry_run=args.dry_run,
    )

    run(
        installer_pip_command() + ["install", "imageio", "imageio-ffmpeg", "opencv-python-headless"],
        dry_run=args.dry_run,
    )

    wget_download(
        "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt",
        COMFYUI_MODELS / "animatediff_models" / "v3_sd15_mm.ckpt",
        dry_run=args.dry_run,
    )
    wget_download(
        "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
        COMFYUI_MODELS / "checkpoints" / "v1-5-pruned-emaonly.safetensors",
        dry_run=args.dry_run,
    )

    summary.append("video: installed AnimateDiff + VideoHelperSuite + baseline motion/checkpoint assets")


def install_3d_bundle(args: argparse.Namespace, summary: list[str]) -> None:
    print_header("Preparing 3D Mesh Stack")
    ensure_dirs(
        [
            COMFYUI_MODELS / "checkpoints",
            COMFYUI_MODELS / "clip_vision",
            COMFYUI_DIR / "input",
            COMFYUI_NODES,
        ],
        dry_run=args.dry_run,
    )

    repo_url = os.environ.get("EDISON_3D_NODE_REPO", "").strip()
    model_url = os.environ.get("EDISON_3D_MODEL_URL", "").strip()
    model_name = os.environ.get("EDISON_3D_MODEL_NAME", "experimental-3d-model.safetensors").strip()

    if repo_url:
        clone_or_update(repo_url, COMFYUI_NODES / Path(repo_url).stem.replace(".git", ""), dry_run=args.dry_run)
        summary.append(f"3d: cloned custom node repo {repo_url}")
    else:
        summary.append("3d: created directories only; set EDISON_3D_NODE_REPO to auto-clone a ComfyUI 3D node pack")

    if model_url:
        wget_download(model_url, COMFYUI_MODELS / "checkpoints" / model_name, dry_run=args.dry_run, auth_token=args.hf_token)
        summary.append(f"3d: downloaded custom model {model_name}")
    else:
        summary.append("3d: no model URL provided; set EDISON_3D_MODEL_URL and EDISON_3D_MODEL_NAME to fetch a 3D weight")


def check_prereqs() -> None:
    missing = [tool for tool in ("git", "wget") if shutil.which(tool) is None]
    if missing:
        raise SetupError(f"Missing required tools: {', '.join(missing)}")
    if not COMFYUI_DIR.exists():
        raise SetupError(f"ComfyUI directory not found at {COMFYUI_DIR}")


def main() -> int:
    args = parse_args()
    check_prereqs()

    print_header("EDISON Media Stack Setup")
    print(f"Repo root: {REPO_ROOT}")
    print(f"LLM dir:   {LLM_MODELS_DIR}")
    print(f"ComfyUI:   {COMFYUI_DIR}")
    print(f"Python:    {installer_python()}")

    summary: list[str] = []

    try:
        if not args.skip_vision and ask_yes_no("Install vision models?", default=True, assume_yes=args.yes):
            install_vision_bundle(args, summary)
        if not args.skip_image and ask_yes_no("Install FLUX image generation stack?", default=True, assume_yes=args.yes):
            install_image_bundle(args, summary)
        if not args.skip_video and ask_yes_no("Install video nodes and baseline video assets?", default=True, assume_yes=args.yes):
            install_video_bundle(args, summary)
        if args.include_3d and ask_yes_no("Prepare 3D mesh stack?", default=True, assume_yes=args.yes):
            install_3d_bundle(args, summary)
    except SetupError as exc:
        print(f"ERROR {exc}")
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"ERROR command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}")
        return exc.returncode or 1

    print_header("Setup Summary")
    if not summary:
        print("No bundles were selected.")
    else:
        for line in summary:
            print(f"- {line}")

    print("\nNext step: restart EDISON services so new models are detected.")
    print(f"Suggested command: cd {REPO_ROOT} && ./restart_edison.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())