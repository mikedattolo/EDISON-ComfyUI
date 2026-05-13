"""Tests for the additive Persona Video Studio module."""

from pathlib import Path

import pytest


def _rights_payload(**overrides):
    payload = {
        "all_visible_performers_are_consenting_adults": True,
        "source_footage_cleared_for_ai_transformation": True,
        "persona_identity_is_synthetic_or_authorized": True,
        "material_is_non_explicit_production_footage": True,
        "material_does_not_depict_or_simulate_minors": True,
        "no_unauthorized_real_person_impersonation": True,
        "local_metadata_acknowledgment_understood": True,
    }
    payload.update(overrides)
    return payload


def _service(tmp_path):
    from services.edison_core.persona_video import PersonaVideoService

    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = {
        "edison": {
            "persona_video": {
                "default_output_directory": str(repo / "outputs" / "persona_video"),
                "temp_working_directory": str(repo / "outputs" / "persona_video" / "tmp"),
                "upload_directory": str(repo / "uploads" / "persona_video"),
                "backend": "metadata_only_passthrough",
                "maximum_concurrent_segment_workers": 2,
            }
        }
    }
    return PersonaVideoService(repo, cfg)


def test_rights_consent_metadata_validation():
    from services.edison_core.persona_video import PersonaVideoValidationError, validate_rights_acknowledgement

    assert validate_rights_acknowledgement(_rights_payload())
    with pytest.raises(PersonaVideoValidationError):
        validate_rights_acknowledgement(_rights_payload(material_is_non_explicit_production_footage=False))


def test_persona_pack_schema_and_registry(tmp_path):
    service = _service(tmp_path)
    pack = service.registry.register_pack({
        "name": "Synthetic Performer A",
        "type": "lora",
        "paths": ["models/personas/synthetic-a.safetensors"],
        "notes": "test pack",
        "preferred_backend_compatibility": ["metadata_only_passthrough"],
    })
    assert pack["persona_id"].startswith("persona_")
    assert pack["pack_type"] == "lora"
    assert pack["schema_version"] == "1.1"
    assert pack["validation_state"] == "missing_files"
    assert pack["missing_paths"] == ["models/personas/synthetic-a.safetensors"]
    assert service.registry.get_pack(pack["persona_id"])["name"] == "Synthetic Performer A"


def test_video_job_schema_creation_without_autostart(tmp_path):
    service = _service(tmp_path)
    source = service.repo_root / "source.mp4"
    source.write_bytes(b"not a real mp4 but enough for queued schema validation")
    pack = service.registry.register_pack({"name": "Synthetic Actor", "type": "other", "paths": []})

    job = service.create_job({
        "project_title": "Dry Run",
        "source_path": str(source),
        "persona_id": pack["persona_id"],
        "rights_acknowledgement": _rights_payload(),
        "settings": {
            "backend": "metadata_only_passthrough",
            "transformation_scope": "metadata_validation_only",
        },
    }, autostart=False)

    assert job["job_type"] == "persona_video"
    assert job["status"] == "queued"
    assert job["rights_acknowledgement"]["material_is_non_explicit_production_footage"] is True
    assert job["persona"]["persona_id"] == pack["persona_id"]


def test_invalid_backend_handling(tmp_path):
    service = _service(tmp_path)
    source = service.repo_root / "source.mp4"
    source.write_bytes(b"x")
    pack = service.registry.register_pack({"name": "Synthetic Actor", "type": "other"})

    with pytest.raises(KeyError):
        service.create_job({
            "project_title": "Bad Backend",
            "source_path": str(source),
            "persona_id": pack["persona_id"],
            "rights_acknowledgement": _rights_payload(),
            "settings": {"backend": "missing_backend"},
        }, autostart=False)


def test_gpu_strategy_selection_logic():
    from services.edison_core.persona_video_gpu import GPUStrategySelector

    gpus = [
        {"index": 0, "name": "NVIDIA GeForce RTX 3090", "total_mb": 24576, "free_mb": 22000, "used_mb": 2000},
        {"index": 1, "name": "NVIDIA GeForce RTX 5060 Ti", "total_mb": 16384, "free_mb": 15000, "used_mb": 1000},
        {"index": 2, "name": "NVIDIA GeForce RTX 4060 Ti", "total_mb": 16384, "free_mb": 15000, "used_mb": 1000},
    ]
    capabilities = {"supports_parallel_segment_processing": True, "supports_multi_gpu": True}
    plan = GPUStrategySelector.select(gpus, capabilities, "auto", max_concurrent_workers=3)
    assert plan.primary_gpu == 0
    assert plan.effective_strategy == "parallel_all"
    assert plan.worker_gpus == [0, 1, 2]

    no_parallel = GPUStrategySelector.select(gpus, {"supports_multi_gpu": False}, "parallel_all", max_concurrent_workers=3)
    assert no_parallel.effective_strategy == "3090_only"
    assert no_parallel.worker_gpus == [0]


def test_segment_queue_scheduling():
    from services.edison_core.persona_video import build_segments
    from services.edison_core.persona_video_gpu import GPUStrategyPlan, SegmentQueueScheduler

    plan = GPUStrategyPlan(
        requested_strategy="parallel_all",
        effective_strategy="parallel_all",
        primary_gpu=0,
        worker_gpus=[0, 1, 2],
        auxiliary_gpus=[1, 2],
        max_concurrent_workers=3,
    )
    scheduler = SegmentQueueScheduler(plan, [
        {"index": 0, "name": "RTX 3090", "total_mb": 24576},
        {"index": 1, "name": "RTX 5060 Ti", "total_mb": 16384},
        {"index": 2, "name": "RTX 4060 Ti", "total_mb": 16384},
    ])
    assigned = scheduler.assign_segments(build_segments(120, preference="short"))
    assert {seg["gpu_assignment"]["gpu_index"] for seg in assigned}.issubset({0, 1, 2})
    assert any("Segment seg_001" in line for line in scheduler.stage_assignment_log(assigned))


def test_shot_detection_builds_real_cut_segments(monkeypatch, tmp_path):
    import services.edison_core.persona_video as pvs

    monkeypatch.setattr(
        pvs,
        "detect_shot_boundaries",
        lambda source: {
            "method": "ffmpeg_scene_score",
            "cut_timestamps_s": [4.0, 9.0],
            "cut_count": 2,
            "fallback": False,
        },
    )

    segments = pvs.detect_shots_and_build_segments(tmp_path / "source.mp4", 12.0)

    assert [segment["duration_s"] for segment in segments] == [4.0, 5.0, 3.0]
    assert segments[0]["shot_detection"]["method"] == "ffmpeg_scene_score"


def test_tracking_metadata_fallback_provider(tmp_path):
    from services.edison_core.persona_tracking import TrackingService

    service = TrackingService(providers=[])
    result = service.track(
        {"target_selection": {"mode": "manual", "subject_label": "performer A"}},
        [{"segment_id": "seg_001"}, {"segment_id": "seg_002"}],
        tmp_path,
    )

    assert result["fallback"] is True
    assert result["candidate_count"] == 1
    assert result["tracks"][0]["subject_label"] == "performer A"
    assert result["tracks"][0]["per_segment_coverage"]["seg_002"] == 1.0


def test_basic_segment_qc_flags_missing_output(tmp_path):
    from services.edison_core.persona_qc import basic_segment_qc

    qc = basic_segment_qc(output_path=tmp_path / "missing.mp4", segment={"duration_s": 2.0})

    assert qc.status == "failed"
    assert qc.needs_review is True
    assert "missing_output_file" in qc.warning_flags


def test_exclusive_render_snapshot_restore_with_mocked_services(monkeypatch):
    from services.edison_core.persona_video_gpu import EdisonServiceController, ExclusiveGPURenderManager, GPUDevice

    class FakeController(EdisonServiceController):
        def __init__(self):
            self.restored = False

        def snapshot(self):
            return {"available": True, "llm_fast_loaded": True}

        def suspend_for_render(self):
            return {"ok": True, "actions": ["unloaded_llm_models:fast"], "errors": []}

        def restore_after_render(self, snapshot):
            self.restored = True
            return {"ok": True, "actions": ["requested_text_model_restore"], "errors": []}

    manager = ExclusiveGPURenderManager(FakeController())
    monkeypatch.setattr(manager, "detect_gpus", lambda: [GPUDevice(index=0, name="RTX 3090", total_mb=24576, free_mb=22000, used_mb=1000)])
    monkeypatch.setattr(manager, "query_gpu_processes", lambda: [])
    state = manager.enter_exclusive_mode({"0": 18000}, wait_timeout_s=0.1, poll_interval_s=0.01)
    assert state["readiness"]["ready"] is True
    restored = manager.restore(state)
    assert restored["restore_result"]["ok"] is True


def test_reprocessing_failed_segment_state_transition():
    from services.edison_core.persona_video import mark_segment_for_retry

    job = {"job_id": "pvs_test", "status": "needs_review", "segments": [{"segment_id": "seg_001", "status": "failed"}]}
    updated = mark_segment_for_retry(job, "seg_001", reason="unit_test")
    assert updated["status"] == "running"
    assert updated["segments"][0]["retry_requested"] is True
    assert updated["rerender_history"][0]["segment_id"] == "seg_001"


def test_audio_preserve_remux_command_construction():
    from services.edison_core.persona_video import build_remux_command

    cmd = build_remux_command(Path("video.mp4"), Path("source.mp4"), Path("out.mp4"), "preserve_original")
    assert "-map" in cmd
    assert "1:a?" in cmd
    assert cmd[-1] == "out.mp4"

    strip = build_remux_command(Path("video.mp4"), Path("source.mp4"), Path("silent.mp4"), "strip")
    assert "-an" in strip


def test_metadata_report_generation(tmp_path):
    service = _service(tmp_path)
    report = service.build_metadata_report({
        "job_id": "pvs_test",
        "status": "completed",
        "rights_acknowledgement": _rights_payload(),
        "segments": [{"segment_id": "seg_001", "status": "completed"}],
        "outputs": {"final_video_path": "/tmp/out.mp4"},
    })
    assert report["schema"] == "edison.persona_video.report.v1"
    assert report["rights_acknowledgement"]["no_unauthorized_real_person_impersonation"] is True
    assert report["segments"][0]["segment_id"] == "seg_001"


def test_persona_video_router_mounts():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from services.edison_core.routes.persona_video import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    response = client.get("/persona-video/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "material_is_non_explicit_production_footage" in body["rights_ack_fields"]
