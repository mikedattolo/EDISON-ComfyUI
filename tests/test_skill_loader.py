from pathlib import Path

from services.edison_core.skill_loader import SkillLoader
from services.edison_core.tool_framework import ToolRegistry


def test_skill_loader_registers_skill_tools(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "demo_skill.py").write_text(
        """
from services.edison_core.tool_framework import BaseTool, ToolResult
SKILL_METADATA = {"name": "demo_skill", "enabled": True, "required_permissions": ["network"]}
class DemoTool(BaseTool):
    name = "skill.demo"
    description = "demo"
    def execute(self, **kwargs):
        return ToolResult(success=True, data={"ok": True})
def register(tool_registry):
    tool_registry.register(DemoTool())
    return ["skill.demo"]
"""
    )

    registry = ToolRegistry()
    loader = SkillLoader(
        tool_registry=registry,
        skills_dir=skills_dir,
        config_getter=lambda: {"edison": {"skills": {"allowed_permissions": ["network"]}}},
    )

    result = loader.load_all()
    assert len(result["loaded"]) == 1
    assert registry.get("skill.demo") is not None


def test_skill_loader_blocks_missing_permissions(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "locked_skill.py").write_text(
        """
SKILL_METADATA = {"name": "locked_skill", "enabled": True, "required_permissions": ["filesystem"]}
def register(tool_registry):
    return []
"""
    )

    registry = ToolRegistry()
    loader = SkillLoader(
        tool_registry=registry,
        skills_dir=skills_dir,
        config_getter=lambda: {"edison": {"skills": {"allowed_permissions": ["network"]}}},
    )

    result = loader.load_all()
    assert result["loaded"] == []
    assert len(result["skipped"]) == 1
