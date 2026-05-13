# System Doctor, CUDA, and Runtime Diagnostics

Open `/system-diagnostics` to run the Edison System Doctor. The UI calls `GET /api/system/doctor` and can optionally run a small CUDA allocation check with `run_cuda_allocation=true`.

The report includes:

- NVIDIA driver visibility through `nvidia-smi`
- GPU names, UUIDs, bus IDs, VRAM, temperature, utilization, power, and fan percent when exposed
- CUDA toolkit / `nvcc` visibility
- Python executable and AI package versions
- PyTorch import, CUDA availability, CUDA runtime version, GPU count, device names, supported CUDA architectures, and optional per-device allocation
- FFmpeg/FFprobe/Whisper binary readiness
- ComfyUI queue reachability and Persona workflow template library status
- output/upload/temp path writability
- GPU cooling/fan-health summary

The legacy lightweight endpoint remains at `/api/system/diagnostics` for older web tools. The production doctor endpoint is `/api/system/doctor`.

## Live Edison Audit Notes

The live Edison host was verified over SSH after matching the host ED25519 fingerprint. CUDA initially failed because Secure Boot rejected the DKMS-built NVIDIA 580 modules signed by an unenrolled local key. The host already had Canonical-signed `linux-modules-nvidia-580-open` modules for the active kernel, so the rejected DKMS modules were moved out of the module search path, `depmod` was rebuilt, and the Canonical-signed modules loaded successfully.

After repair, `nvidia-smi` detected:

- index 0: RTX 5060 Ti 16GB
- index 1: RTX 4060 Ti 16GB
- index 2: RTX 3090 24GB

PyTorch `2.6.0+cu124` can use the RTX 3090 and RTX 4060 Ti, but the RTX 5060 Ti reports `sm_120`, which is not supported by that PyTorch wheel family. Until Edison is upgraded to a PyTorch/CUDA build that supports `sm_120`, mask PyTorch services to the 3090 and 4060 Ti with `CUDA_VISIBLE_DEVICES` or equivalent service overrides. When the CUDA allocation check is enabled, System Doctor treats a successful allocation as stronger evidence than the static architecture list, which avoids false alarms on cards that run correctly even if their architecture is not listed by `torch.cuda.get_arch_list()`.

## What Edison Can Fix Automatically

Edison can detect and explain many configuration problems, but it does not blindly mutate Python environments or system driver packages. Use the report to identify the active Edison and ComfyUI Python environments, then reconcile packages in the correct environment only.

Typical fixes:

- install the PyTorch CUDA wheel matching the installed NVIDIA driver
- mask a too-new GPU from PyTorch services when the installed wheel does not support its compute capability
- install missing `torchvision` / `torchaudio` wheels from the same family
- repair NVIDIA driver utilities when `nvidia-smi` fails on the host
- install FFmpeg when video/persona workflows need probing, cutting, or remuxing
- add missing ComfyUI custom nodes/models listed by workflow templates
