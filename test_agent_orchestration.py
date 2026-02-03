from services.edison_core.orchestration import AgentControllerBrain


def test_agent_plan_swarm_parallel():
    controller = AgentControllerBrain(config={})
    plan = controller.plan(goal="Build a demo", mode="swarm", has_image=True)
    assert plan.parallel is True
    assert plan.mode == "swarm"
    assert len(plan.tasks) >= 3
    kinds = {task.kind for task in plan.tasks}
    assert "llm" in kinds
    assert "vision" in kinds
