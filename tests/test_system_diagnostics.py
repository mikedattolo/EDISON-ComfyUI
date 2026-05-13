from services.edison_core.system_diagnostics import DiagnosticCheck, report_as_text


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
