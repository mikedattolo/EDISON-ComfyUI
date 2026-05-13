# System Doctor, CUDA, and Runtime Diagnostics

Open `/system-diagnostics` to run the Edison System Doctor. The UI calls `GET /api/system/doctor` and can optionally run a small CUDA allocation check with `run_cuda_allocation=true`.

The report includes:

- NVIDIA driver visibility through `nvidia-smi`
- GPU names, UUIDs, bus IDs, VRAM, temperature, utilization, power, and fan percent when exposed
- CUDA toolkit / `nvcc` visibility
- Python executable and AI package versions
- PyTorch import, CUDA availability, CUDA runtime version, GPU count, device names, and optional per-device allocation
- FFmpeg/FFprobe/Whisper binary readiness
- ComfyUI queue reachability and Persona workflow template library status
- output/upload/temp path writability
- GPU cooling/fan-health summary

The legacy lightweight endpoint remains at `/api/system/diagnostics` for older web tools. The production doctor endpoint is `/api/system/doctor`.

## Live Environment Limitation

During this implementation, SSH to `mike@192.168.1.46` timed out and SSH to `mike@100.67.221.112` reached a host but failed host-key verification. No host-key bypass was performed. Therefore no live CUDA, PyTorch, ComfyUI, or GPU fan repair was applied on the Edison PC from this session.

## What Edison Can Fix Automatically

Edison can detect and explain many configuration problems, but it does not blindly mutate Python environments or system driver packages. Use the report to identify the active Edison and ComfyUI Python environments, then reconcile packages in the correct environment only.

Typical fixes:

- install the PyTorch CUDA wheel matching the installed NVIDIA driver
- install missing `torchvision` / `torchaudio` wheels from the same family
- repair NVIDIA driver utilities when `nvidia-smi` fails on the host
- install FFmpeg when video/persona workflows need probing, cutting, or remuxing
- add missing ComfyUI custom nodes/models listed by workflow templates

