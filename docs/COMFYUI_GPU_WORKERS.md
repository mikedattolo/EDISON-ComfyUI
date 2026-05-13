# ComfyUI Per-GPU Workers

Edison uses per-GPU ComfyUI workers for real multi-GPU throughput. This is the
preferred architecture for the Edison AI PC because the RTX 3090, RTX 5060 Ti,
and RTX 4060 Ti each have separate VRAM pools. Edison does not treat them as one
pooled 56GB device.

## Worker Layout

Default Edison AI PC routing:

- `rtx3090-primary`: `127.0.0.1:8188`, RTX 3090 24GB, primary large-job worker.
- `rtx5060ti-aux`: `127.0.0.1:8189`, RTX 5060 Ti 16GB, auxiliary image/persona worker.
- `rtx4060ti-aux`: `127.0.0.1:8190`, RTX 4060 Ti 16GB, auxiliary image/persona worker.

The primary legacy `edison-comfyui.service` remains on port `8188` for
backwards compatibility. Auxiliary workers use the templated
`edison-comfyui-worker@.service` unit and one environment file per worker under
`/etc/edison/comfyui-workers/`.

## Install Or Update Workers

From the Edison host after pulling the branch:

```bash
sudo /opt/edison/scripts/install_comfyui_gpu_workers.sh
```

The installer:

- validates configured GPU UUIDs against `nvidia-smi`
- constrains the primary ComfyUI service to the RTX 3090
- installs auxiliary worker systemd units
- starts the 5060 Ti and 4060 Ti workers
- restarts Edison core so routing config is reloaded
- probes `8188`, `8189`, and `8190`

## Edison Routing Behavior

When `edison.comfyui.workers_enabled` is true, Edison selects a worker by:

- explicit `comfyui_worker` request, if supplied
- Persona Video GPU assignment, if supplied
- estimated VRAM requirement
- worker queue depth
- primary/auxiliary role

Large estimates are routed to the 3090. Jobs that fit the 16GB cards can be
spread across auxiliary workers when their queues are lighter.

Status polling remembers the worker that accepted each image prompt, so
`/image-status/{prompt_id}` checks the correct ComfyUI port instead of always
polling `8188`.

## What This Does Not Do

This does not pool VRAM. A single ComfyUI workflow still runs inside one worker
process and one visible GPU unless the workflow/backend itself explicitly
supports model sharding. Edison uses the three cards together by dispatching
separate jobs or Persona Video segments to separate workers.

## Health Checks

Useful endpoints:

```bash
curl http://127.0.0.1:8811/comfyui/workers
curl "http://127.0.0.1:8811/system/doctor?run_cuda_allocation=true"
curl http://127.0.0.1:8188/queue
curl http://127.0.0.1:8189/queue
curl http://127.0.0.1:8190/queue
```

Useful service commands:

```bash
sudo systemctl status edison-comfyui.service
sudo systemctl status edison-comfyui-worker@rtx5060ti-aux.service
sudo systemctl status edison-comfyui-worker@rtx4060ti-aux.service
sudo journalctl -u edison-comfyui-worker@rtx5060ti-aux.service -f
```

## Troubleshooting

If a worker fails with CUDA errors, confirm:

- the NVIDIA kernel module is loaded and `nvidia-smi` sees all three cards
- the worker's `CUDA_VISIBLE_DEVICES` UUID exists
- the ComfyUI venv has a CUDA-enabled PyTorch build that supports the GPU
- no other process has consumed most of that card's VRAM

If all workers start but Edison still routes only to `8188`, check
`edison.comfyui.workers_enabled` and restart `edison-core.service`.
