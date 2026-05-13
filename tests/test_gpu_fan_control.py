"""Tests for safe GPU fan-control logic."""

from pathlib import Path


class FakeFanControllerMixin:
    def __init__(self, *args, fake_readings=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_readings = fake_readings or []
        self.applied_targets = []

    def read_gpus(self):
        return self.fake_readings, []

    def _apply_nvidia_settings(self, gpu_index, fan_percent):
        self.applied_targets.append((gpu_index, fan_percent))
        return ["nvidia-settings", str(gpu_index), str(fan_percent)]

    def diagnostics(self):
        data = super().diagnostics()
        data["test_override"] = True
        return data


def test_interpolate_curve():
    from services.edison_core.gpu_fan_control import DEFAULT_CURVE, interpolate_curve

    assert interpolate_curve(35, DEFAULT_CURVE) == 30
    assert 39 <= interpolate_curve(50, DEFAULT_CURVE) <= 45
    assert interpolate_curve(90, DEFAULT_CURVE) == 100


def test_4060_zero_rpm_spinup_guard(tmp_path):
    from services.edison_core.gpu_fan_control import GpuFanController, GpuFanReading

    controller = GpuFanController(tmp_path, {"edison": {"gpu_fan_control": {"apply_enabled_default": False}}})
    reading = GpuFanReading(index=2, name="NVIDIA GeForce RTX 4060 Ti", temperature_c=50, fan_speed_percent=0)
    decision = controller.decide(reading, apply_requested=False)

    assert decision.target_fan_percent >= 60
    assert "spin-up" in decision.reason
    assert any("0%" in warning for warning in decision.warnings)


def test_dry_run_does_not_apply(tmp_path):
    from services.edison_core.gpu_fan_control import GpuFanController, GpuFanReading

    class FakeController(FakeFanControllerMixin, GpuFanController):
        pass

    controller = FakeController(
        tmp_path,
        {"edison": {"gpu_fan_control": {"control_preference": ["nvidia-settings"]}}},
        fake_readings=[GpuFanReading(index=0, name="RTX 3090", temperature_c=70, fan_speed_percent=40)],
    )
    result = controller.evaluate_once(apply=False)

    assert result["decisions"][0]["backend"] == "dry_run"
    assert controller.applied_targets == []


def test_apply_once_uses_control_backend(tmp_path):
    from services.edison_core.gpu_fan_control import GpuFanController, GpuFanReading

    class FakeController(FakeFanControllerMixin, GpuFanController):
        pass

    controller = FakeController(
        tmp_path,
        {"edison": {"gpu_fan_control": {"control_preference": ["nvidia-settings"], "minimum_change_interval_s": 0}}},
        fake_readings=[GpuFanReading(index=1, name="RTX 5060 Ti", temperature_c=72, fan_speed_percent=45)],
    )
    result = controller.evaluate_once(apply=True)

    assert result["decisions"][0]["applied"] is True
    assert result["decisions"][0]["backend"] == "nvidia-settings"
    assert controller.applied_targets[0][0] == 1


def test_diagnostics_explain_missing_container_gpu(monkeypatch, tmp_path):
    import services.edison_core.gpu_fan_control as fans

    original_glob = Path.glob

    def fake_glob(self, pattern):
        if str(self) == "/dev" and pattern == "nvidia*":
            return iter([])
        return original_glob(self, pattern)

    monkeypatch.setattr(fans, "_find_binary", lambda name: None)
    monkeypatch.setattr(Path, "glob", fake_glob)

    controller = fans.GpuFanController(tmp_path, {})
    monkeypatch.setattr(controller, "_probe_nvml", lambda: {"ok": False, "fan_set_supported_hint": False})
    diag = controller.diagnostics()

    assert diag["nvidia_smi_path"] is None
    assert diag["nvidia_settings_path"] is None
    assert diag["container_or_host_has_gpu_access"] is False
    assert any("NVIDIA" in item or "nvidia" in item for item in diag["recommendations"])
