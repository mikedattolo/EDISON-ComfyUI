"""Example skill module for dynamic plugin loading."""

from __future__ import annotations

from services.edison_core.tool_framework import BaseTool, ToolResult, ToolRegistry


SKILL_METADATA = {
    "name": "weather_skill",
    "version": "1.0.0",
    "description": "Demo plugin that exposes a simple weather stub tool",
    "required_permissions": ["network"],
    "enabled": True,
}


class WeatherSummaryTool(BaseTool):
    name = "skill.weather_summary"
    description = "Return a short weather summary for a location (example skill tool)"

    def validate_params(self, location: str = "", **kwargs):
        if not location or not str(location).strip():
            return "location is required"
        return None

    def execute(self, location: str = "", **kwargs) -> ToolResult:
        # Intentionally simple: demonstrates the skill/plugin contract only.
        return ToolResult(
            success=True,
            data={
                "location": location,
                "summary": f"Weather integration plugin is active for {location}.",
            },
        )


def register(tool_registry: ToolRegistry):
    tool_registry.register(WeatherSummaryTool())
    return ["skill.weather_summary"]
