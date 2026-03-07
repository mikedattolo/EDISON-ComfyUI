# EDISON Containerized Dev Environment

This repository includes a containerized dev/runtime stack for EDISON.

## Files
- `Dockerfile`: builds the core runtime image.
- `docker-compose.yml`: runs `edison-core` with mapped ports and volumes.
- `scripts/edison_docker.py`: CLI helper for common workflows.

## Quick Start
```bash
python scripts/edison_docker.py init
python scripts/edison_docker.py start
```

Stop services:
```bash
python scripts/edison_docker.py stop
```

Tail logs:
```bash
python scripts/edison_docker.py logs --tail 200
```

## Volume Mounts
- `./models -> /opt/edison/models`
- `./data -> /opt/edison/data`
- `./outputs -> /opt/edison/outputs`
- `./config -> /opt/edison/config`

## Optional Toolchain Profiles
Install extra toolchains inside the running container:
```bash
python scripts/edison_docker.py profiles --name python rust
```

Supported profile names:
- `python`: installs dev Python packages/tools.
- `rust`: installs `rustc` + `cargo`.
- `node`: installs `nodejs` + `npm`.

## Firewall Controls
Container networking can be configured via environment variables or `config/edison.yaml`:

- `FIREWALL_MODE`: `allowlist` or `open`
- `ALLOWED_EGRESS_CIDRS`: comma-separated CIDR list

Compose reads these values at startup. Defaults are pulled from `config/edison.yaml` under:
```yaml
edison:
  containers:
    firewall_mode: allowlist
    allowed_egress_cidrs: ["0.0.0.0/0"]
```

## Notes
- Ports exposed: `8811` (core API), `8080` (web proxy/ui), `8188` (ComfyUI).
- Update `config/edison.yaml` to point model paths to mounted model locations as needed.
