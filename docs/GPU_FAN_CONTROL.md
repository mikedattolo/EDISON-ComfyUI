# GPU Cooling Diagnostics and Fan Control

EDISON includes a conservative NVIDIA GPU fan watchdog for the three-card workstation target:

- RTX 3090 24GB
- RTX 5060 Ti 16GB
- RTX 4060 Ti 16GB

It monitors GPU temperature, utilization, VRAM, power, fan percent, optional RPM telemetry, and cooling-health classification. It computes fan targets from a safe curve and can apply those targets on a properly configured Linux host. The API/UI dry-run by default so you can diagnose driver access before writing fan speeds.

## Important safety notes

- Fan control must run on the host or in a privileged GPU-enabled container with `/dev/nvidia*` devices exposed.
- NVIDIA fan writes normally require `nvidia-settings`, a running Xorg display, and Coolbits fan-control enabled.
- EDISON does not disable thermal protections. The curve clamps to a minimum fan percent and forces 100% near critical temperature.
- Modern RTX cards may stop their fans at low temperature because of zero-RPM mode. This is normal when cool; it is suspicious only when temperature rises and the fan stays at 0%.
- Edison cannot repair a physical fan, cable, shroud, or card failure in software. It can only classify symptoms, warn, and safely command fans when the driver stack permits it.

## What the service does

1. Reads telemetry through NVML, `nvidia-smi`, then sysfs fallback.
2. Applies a configured fan curve.
3. Adds a 4060 Ti guard: if the card reports fan `0%` above `42 C`, target at least `60%`.
4. Applies targets through `nvidia-settings` or NVML fan APIs when available.
5. Classifies cooling state as `normal`, `zero_rpm_idle_probably_normal`, `telemetry_unavailable`, `manual_control_unavailable`, `suspect_fan_not_spinning`, or `over_temp_warning`.
6. Reports diagnostics if `nvidia-smi` is missing or cannot communicate with the driver.

## UI and API

Open:

- `/gpu-fans`
- `/system-diagnostics` for CUDA/PyTorch/ComfyUI/path/cooling rollup

API routes through the web proxy:

- `GET /api/gpu-fans/health`
- `GET /api/gpu-fans/status`
- `GET /api/gpu-fans/diagnostics`
- `POST /api/gpu-fans/apply-once`
- `POST /api/gpu-fans/start`
- `POST /api/gpu-fans/stop`
- `POST /api/gpu-fans/reload`

Dry-run status:

```bash
python -m services.edison_core.gpu_fan_control --once
```

Apply once on the host:

```bash
python -m services.edison_core.gpu_fan_control --once --apply
```

Run continuously on the host:

```bash
python -m services.edison_core.gpu_fan_control --loop --apply --interval 5
```

## Fixing `nvidia-smi` not communicating

If EDISON reports that `nvidia-smi` is missing or cannot communicate, check the host first:

```bash
which nvidia-smi
nvidia-smi
ls -l /dev/nvidia*
ls /proc/driver/nvidia
```

If those fail on the host, reinstall or repair the NVIDIA driver package for your distro, then reboot.

On Secure Boot systems, `modprobe nvidia` may fail with `Key was rejected by service`. On Edison, the host had Canonical-signed `linux-modules-nvidia-580-open` modules installed, but DKMS-built modules in `updates/dkms` were shadowing them. Moving the rejected DKMS `nvidia*.ko*` files out of the active kernel module path and running `depmod -a` allowed the signed modules to load without a physical MOK enrollment screen.

If they work on the host but fail inside Docker/devcontainer, install NVIDIA Container Toolkit and run the container with GPU access:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Then launch the container with GPU access, for example:

```bash
docker run --gpus all --env NVIDIA_DRIVER_CAPABILITIES=all,compute,utility,graphics ...
```

A normal unprivileged dev container may not be able to see or control fans even when the host can.

## Enabling fan writes with `nvidia-settings`

NVIDIA fan speed writes usually require Coolbits in Xorg. On Ubuntu/Xorg hosts, create or update an NVIDIA Xorg snippet similar to:

```bash
sudo nvidia-xconfig --enable-all-gpus --cool-bits=12
sudo systemctl restart gdm3
```

If you use a different display manager, restart that manager instead. Verify that `nvidia-settings` can query fan attributes:

```bash
DISPLAY=:0 nvidia-settings -q fans
DISPLAY=:0 nvidia-settings -q [fan:0]/GPUTargetFanSpeed
```

If X authority blocks the service user, set `XAUTHORITY` in `services/systemd/edison-gpu-fan.service` to the correct session authority file.

## Installing the systemd service

Copy the unit file and enable it on the host:

```bash
sudo cp services/systemd/edison-gpu-fan.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now edison-gpu-fan
sudo journalctl -u edison-gpu-fan -f
```

The bundled unit assumes EDISON is installed at `/opt/edison` and runs as user `edison`. Adjust `WorkingDirectory`, `ExecStart`, `DISPLAY`, and `XAUTHORITY` for your machine.

## Configuration

Settings live in `config/edison.yaml` under `edison.gpu_fan_control`.

Key settings:

- `apply_enabled_default`: keep `false` for UI/API dry-run safety.
- `poll_interval_s`: loop interval.
- `telemetry_preference`: `nvml`, `nvidia-smi`, `sysfs`.
- `control_preference`: `nvidia-settings`, `nvml`.
- `fan_index_map`: maps GPU index to fan index.
- `curve`: temperature-to-fan-percent points.
- `gpu_overrides.4060`: zero-RPM spin-up guard for the 4060 Ti.
- `zero_rpm_idle_temp_c`: below this, 0% fan can be treated as likely idle behavior.
- `elevated_temp_c`: temperature where 0% fan becomes suspicious.
- `temperature_warning_c` / `temperature_critical_c`: UI/report thresholds for heavy-job warnings.
- `fan_anomaly_warnings_enabled`: include suspicious fan classifications in readings.
- `dashboard_refresh_interval_s`: suggested UI polling interval.

## Interpreting cooling states

- `normal`: telemetry is present and no fan/temperature anomaly is detected.
- `zero_rpm_idle_probably_normal`: fan reads 0% while the GPU is cool and not meaningfully loaded.
- `telemetry_unavailable`: Edison cannot read enough fan/temperature data from the current environment.
- `manual_control_unavailable`: telemetry is readable, but writes are not available through current Linux/NVIDIA settings.
- `suspect_fan_not_spinning`: fan reads 0% or 0 RPM while temperature/utilization is elevated or sustained.
- `over_temp_warning`: temperature is at or above the configured critical threshold.

If one GPU stays at 0%/0 RPM while temperature climbs under real load, physically inspect the fans, fan cable, shroud clearance, PCIe power, and driver/Xorg fan-control state. Do not run long renders until the card cools normally.

## Known limitations

- Some NVIDIA consumer cards expose fan control only through Xorg/Coolbits.
- Wayland-only sessions may need an Xorg session or vendor-specific fan tooling.
- NVML fan-set support varies by driver/GPU and may require elevated permissions.
- The Windows/dev environment used for this implementation cannot read the Edison host's GPU fans. Actual fan diagnosis must be performed on the Ubuntu Edison host after SSH access is verified.
