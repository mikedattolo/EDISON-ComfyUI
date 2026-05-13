#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${EDISON_REPO_ROOT:-/opt/edison}"
CONFIG_PATH="${EDISON_CONFIG_PATH:-$REPO_ROOT/config/edison.yaml}"
PYTHON_BIN="${EDISON_PYTHON:-$REPO_ROOT/.venv/bin/python}"
UNIT_TEMPLATE="$REPO_ROOT/services/systemd/edison-comfyui-worker@.service"
TARGET_UNIT="$REPO_ROOT/services/systemd/edison-comfyui-workers.target"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this installer with sudo."
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing Edison config: $CONFIG_PATH"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Edison Python: $PYTHON_BIN"
  exit 1
fi

if [[ ! -f "$UNIT_TEMPLATE" || ! -f "$TARGET_UNIT" ]]; then
  echo "Missing systemd worker unit files under $REPO_ROOT/services/systemd"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required before installing GPU workers."
  exit 1
fi

install -d -m 0755 /etc/edison/comfyui-workers
install -d -m 0755 /etc/systemd/system/edison-comfyui.service.d
install -m 0644 "$UNIT_TEMPLATE" /etc/systemd/system/edison-comfyui-worker@.service
install -m 0644 "$TARGET_UNIT" /etc/systemd/system/edison-comfyui-workers.target

TMP_IDS="$(mktemp)"
trap 'rm -f "$TMP_IDS"' EXIT

"$PYTHON_BIN" - "$CONFIG_PATH" "$TMP_IDS" <<'PY'
import pathlib
import subprocess
import sys

import yaml

config_path = pathlib.Path(sys.argv[1])
ids_path = pathlib.Path(sys.argv[2])
cfg = yaml.safe_load(config_path.read_text()) or {}
comfy = ((cfg.get("edison") or {}).get("comfyui") or {})
workers = [dict(w) for w in (comfy.get("workers") or []) if w.get("enabled", True)]
if not comfy.get("workers_enabled") or not workers:
    raise SystemExit("edison.comfyui.workers_enabled is false or no workers are configured.")

visible = subprocess.run(
    ["nvidia-smi", "--query-gpu=uuid,name,index", "--format=csv,noheader"],
    capture_output=True,
    text=True,
    check=False,
)
if visible.returncode != 0:
    raise SystemExit("nvidia-smi failed while validating worker GPU UUIDs:\n" + visible.stderr)
visible_text = visible.stdout
missing = [w for w in workers if str(w.get("gpu_uuid") or "") not in visible_text]
if missing:
    rows = ", ".join(f"{w.get('id')}={w.get('gpu_uuid')}" for w in missing)
    raise SystemExit("Configured ComfyUI worker GPU UUIDs are not visible to nvidia-smi: " + rows)

default_port = int(comfy.get("port") or 8188)
primary = next((w for w in workers if str(w.get("role", "")).lower() == "primary"), None)
if primary is None:
    primary = next((w for w in workers if int(w.get("port") or 0) == default_port), workers[0])

primary_uuid = str(primary.get("cuda_visible_devices") or primary.get("gpu_uuid") or "").strip()
if not primary_uuid:
    raise SystemExit("Primary ComfyUI worker is missing cuda_visible_devices/gpu_uuid.")

dropin = pathlib.Path("/etc/systemd/system/edison-comfyui.service.d/10-cuda-visible-devices.conf")
dropin.write_text(
    "[Service]\n"
    'Environment="CUDA_DEVICE_ORDER=PCI_BUS_ID"\n'
    f'Environment="CUDA_VISIBLE_DEVICES={primary_uuid}"\n',
    encoding="utf-8",
)

aux_ids = []
env_root = pathlib.Path("/etc/edison/comfyui-workers")
for worker in workers:
    worker_id = str(worker.get("id") or "").strip()
    if not worker_id:
        raise SystemExit("Every ComfyUI worker needs an id.")
    if worker_id == str(primary.get("id")):
        continue
    cuda_visible = str(worker.get("cuda_visible_devices") or worker.get("gpu_uuid") or "").strip()
    if not cuda_visible:
        raise SystemExit(f"Worker {worker_id} is missing cuda_visible_devices/gpu_uuid.")
    port = int(worker.get("port") or 0)
    if not port:
        raise SystemExit(f"Worker {worker_id} is missing a port.")
    listen = str(worker.get("listen") or worker.get("host") or "127.0.0.1")
    if listen in {"0.0.0.0", "::"}:
        listen = "127.0.0.1"
    env_path = env_root / f"{worker_id}.env"
    env_path.write_text(
        f"EDISON_COMFYUI_LISTEN={listen}\n"
        f"EDISON_COMFYUI_PORT={port}\n"
        f"CUDA_VISIBLE_DEVICES={cuda_visible}\n",
        encoding="utf-8",
    )
    aux_ids.append(worker_id)

ids_path.write_text("\n".join(aux_ids) + ("\n" if aux_ids else ""), encoding="utf-8")
print(f"Primary worker {primary.get('id')} -> edison-comfyui.service on GPU {primary_uuid}")
print("Auxiliary workers: " + (", ".join(aux_ids) if aux_ids else "none"))
PY

mapfile -t AUX_WORKERS < "$TMP_IDS"

echo "Reloading systemd..."
systemctl daemon-reload

echo "Restarting primary ComfyUI worker on port 8188..."
systemctl enable edison-comfyui.service >/dev/null
systemctl restart edison-comfyui.service

if ((${#AUX_WORKERS[@]})); then
  echo "Starting auxiliary ComfyUI GPU workers..."
  systemctl enable edison-comfyui-workers.target >/dev/null || true
  for worker_id in "${AUX_WORKERS[@]}"; do
    [[ -n "$worker_id" ]] || continue
    systemctl enable --now "edison-comfyui-worker@${worker_id}.service"
  done
fi

echo "Restarting Edison core so it reloads worker routing config..."
systemctl restart edison-core.service || true

echo
echo "Worker service status:"
systemctl --no-pager --full status edison-comfyui.service || true
for worker_id in "${AUX_WORKERS[@]}"; do
  [[ -n "$worker_id" ]] || continue
  systemctl --no-pager --full status "edison-comfyui-worker@${worker_id}.service" || true
done

echo
echo "Queue endpoints:"
for port in 8188 8189 8190; do
  printf ":%s " "$port"
  curl -fsS "http://127.0.0.1:${port}/queue" 2>/dev/null || true
  echo
done

echo
echo "GPU snapshot:"
nvidia-smi --query-gpu=index,uuid,name,temperature.gpu,memory.used,memory.total,fan.speed --format=csv
