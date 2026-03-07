from services.edison_core.swarm_safety import SwarmMemoryPolicy


def test_swarm_policy_degraded_on_low_vram(monkeypatch):
    monkeypatch.setattr("services.edison_core.swarm_safety._get_free_vram_mb", lambda: 1000)
    policy = SwarmMemoryPolicy()
    agents = [
        {"model_name": "deep"},
        {"model_name": "medium"},
    ]
    decision = policy.assess(agents, {"deep": True, "medium": True})
    assert decision == "degraded"


def test_swarm_policy_time_slice_mid_vram(monkeypatch):
    monkeypatch.setattr("services.edison_core.swarm_safety._get_free_vram_mb", lambda: 7000)
    policy = SwarmMemoryPolicy()
    agents = [
        {"model_name": "deep"},
        {"model_name": "medium"},
    ]
    decision = policy.assess(agents, {"deep": True, "medium": True})
    assert decision == "time_slice"
