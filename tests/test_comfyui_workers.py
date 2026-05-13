from __future__ import annotations


def _worker_config():
    return {
        "edison": {
            "comfyui": {
                "host": "127.0.0.1",
                "port": 8188,
                "workers_enabled": True,
                "workers": [
                    {
                        "id": "rtx3090-primary",
                        "port": 8188,
                        "gpu_index": 2,
                        "gpu_uuid": "GPU-3090",
                        "gpu_name": "RTX 3090",
                        "role": "primary",
                        "usable_vram_mb": 22000,
                        "preferred_for": ["image", "mesh", "persona_video"],
                    },
                    {
                        "id": "rtx5060ti-aux",
                        "port": 8189,
                        "gpu_index": 0,
                        "gpu_uuid": "GPU-5060",
                        "gpu_name": "RTX 5060 Ti",
                        "role": "auxiliary",
                        "usable_vram_mb": 14500,
                        "preferred_for": ["image", "persona_video"],
                    },
                    {
                        "id": "rtx4060ti-aux",
                        "port": 8190,
                        "gpu_index": 1,
                        "gpu_uuid": "GPU-4060",
                        "gpu_name": "RTX 4060 Ti",
                        "role": "auxiliary",
                        "usable_vram_mb": 14500,
                        "preferred_for": ["image", "persona_video"],
                    },
                ],
            }
        }
    }


def test_legacy_registry_preserves_single_comfyui_config():
    from services.edison_core.comfyui_workers import ComfyUIWorkerRegistry

    registry = ComfyUIWorkerRegistry.from_config({"edison": {"comfyui": {"host": "0.0.0.0", "port": 8188}}})

    assert registry.workers_enabled is False
    assert len(registry.enabled_workers()) == 1
    assert registry.enabled_workers()[0].base_url == "http://127.0.0.1:8188"


def test_high_vram_estimate_routes_to_3090_worker():
    from services.edison_core.comfyui_workers import ComfyUIWorkerRegistry

    registry = ComfyUIWorkerRegistry.from_config(
        _worker_config(),
        queue_probe=lambda worker: {"reachable": True, "running": 0, "pending": 0, "idle": True},
    )

    selected = registry.select(job_type="image", estimated_vram_mb=18000)

    assert selected.worker.id == "rtx3090-primary"
    assert any("exceeds worker limit" in row["reason"] for row in selected.rejected)


def test_gpu_assignment_routes_persona_segment_to_matching_worker():
    from services.edison_core.comfyui_workers import ComfyUIWorkerRegistry

    registry = ComfyUIWorkerRegistry.from_config(
        _worker_config(),
        queue_probe=lambda worker: {"reachable": True, "running": 0, "pending": 0, "idle": True},
    )

    selected = registry.select(job_type="persona_video", gpu_assignment={"gpu_index": 0})

    assert selected.worker.id == "rtx5060ti-aux"
    assert selected.reason == "gpu_assignment_match"


def test_least_busy_worker_prefers_idle_aux_when_jobs_fit():
    from services.edison_core.comfyui_workers import ComfyUIWorkerRegistry

    def probe(worker):
        if worker.id == "rtx3090-primary":
            return {"reachable": True, "running": 1, "pending": 0, "idle": False}
        return {"reachable": True, "running": 0, "pending": 0, "idle": True}

    registry = ComfyUIWorkerRegistry.from_config(_worker_config(), queue_probe=probe)

    selected = registry.select(job_type="image", estimated_vram_mb=9000)

    assert selected.worker.id in {"rtx5060ti-aux", "rtx4060ti-aux"}
    assert selected.reason == "least_busy_reachable_worker"


def test_image_prompt_routes_remember_worker_url():
    from services.edison_core import app as core_app

    core_app._remember_comfyui_prompt("prompt-worker", comfyui_url="http://127.0.0.1:8189", job_type="image")

    assert core_app._comfyui_url_for_prompt("prompt-worker") == "http://127.0.0.1:8189"

    core_app._complete_image_prompt(prompt_id="prompt-worker")
    assert core_app._comfyui_url_for_prompt("prompt-worker") == core_app._comfyui_base_url()
