from services.edison_core.system_diagnostics import (
    DiagnosticCheck,
    _torch_arch_supported,
    report_as_text,
)


def test_report_as_text_includes_recommended_fix():
    report = {
        "overall": "warn",
        "generated_at": 123,
        "checks": [
            DiagnosticCheck(
                key="torch_cuda",
                title="PyTorch CUDA",
                status="fail",
                summary="PyTorch reports CUDA is not available.",
                recommended_fix="Install a matching CUDA wheel.",
            ).to_dict()
        ],
    }

    text = report_as_text(report)

    assert "Overall: warn" in text
    assert "[FAIL] PyTorch CUDA" in text
    assert "Install a matching CUDA wheel" in text


def test_system_diagnostic_report_aggregates_check_status(monkeypatch, tmp_path):
    import services.edison_core.system_diagnostics as diag

    checks = [
        DiagnosticCheck(key="a", title="A", status="pass", summary="ok"),
        DiagnosticCheck(key="b", title="B", status="warn", summary="warned"),
    ]
    monkeypatch.setattr(diag, "collect_nvidia_smi", lambda: checks[0])
    monkeypatch.setattr(diag, "collect_cuda_toolkit", lambda: checks[0])
    monkeypatch.setattr(diag, "collect_python_packages", lambda: checks[0])
    monkeypatch.setattr(diag, "collect_torch_cuda", lambda run_allocation=False: checks[0])
    monkeypatch.setattr(diag, "collect_binary_tools", lambda: checks[0])
    monkeypatch.setattr(diag, "collect_comfyui", lambda repo_root, cfg: checks[0])
    monkeypatch.setattr(diag, "collect_paths", lambda repo_root, cfg: checks[0])
    monkeypatch.setattr(diag, "collect_cooling", lambda repo_root, cfg: checks[1])

    report = diag.build_system_diagnostic_report(tmp_path, {}, run_cuda_allocation=False)

    assert report["overall"] == "warn"
    assert report["counts"]["pass"] == 7
    assert report["counts"]["warn"] == 1


def test_torch_arch_support_matches_sm_and_compute_arches():
    assert _torch_arch_supported("sm_86", ["sm_86", "sm_90"]) is True
    assert _torch_arch_supported("sm_90", ["compute_90"]) is True
    assert _torch_arch_supported("sm_120", ["sm_86", "sm_90"]) is False
    assert _torch_arch_supported("sm_120", []) is None


def test_collect_torch_cuda_warns_for_unsupported_device_arch(monkeypatch):
    import services.edison_core.system_diagnostics as diag

    class FakeProps:
        def __init__(self, name, major, minor, total_memory):
            self.name = name
            self.major = major
            self.minor = minor
            self.total_memory = total_memory

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_arch_list():
            return ["sm_86", "sm_89", "sm_90"]

        @staticmethod
        def get_device_properties(idx):
            if idx == 0:
                return FakeProps("NVIDIA GeForce RTX 3090", 8, 6, 24 * 1024**3)
            return FakeProps("NVIDIA GeForce RTX 5060 Ti", 12, 0, 16 * 1024**3)

    class FakeVersion:
        cuda = "12.4"

    class FakeTorch:
        __version__ = "2.6.0+cu124"
        version = FakeVersion()
        cuda = FakeCuda()

    monkeypatch.setattr(diag.importlib, "import_module", lambda name: FakeTorch if name == "torch" else None)

    check = diag.collect_torch_cuda(run_allocation=False)

    assert check.status == "warn"
    assert check.details["devices"][1]["torch_arch"] == "sm_120"
    assert check.details["devices"][1]["torch_arch_supported"] is False
    assert any("sm_120" in warning for warning in check.details["warnings"])
