# Skills Plugin System

EDISON supports runtime-loadable skills under `services/edison_core/skills/`.

## Skill Module Contract
Each skill module should:
- Be named `*_skill.py`
- Export `SKILL_METADATA`
- Export `register(tool_registry: ToolRegistry)`

Example:
```python
SKILL_METADATA = {
    "name": "weather_skill",
    "version": "1.0.0",
    "required_permissions": ["network"],
    "enabled": True,
}

def register(tool_registry):
    tool_registry.register(MyTool())
    return ["my_tool_name"]
```

## Permissions
Permissions are evaluated from metadata against config:

```yaml
edison:
  skills:
    allowed_permissions: ["network"]
    disabled_skills: []
```

If a skill requests missing permissions, it is skipped.

## Runtime Reloading
The `SkillLoader` polls the skills directory and reloads on file changes.

Endpoints:
- `GET /skills` — list loaded skills and registered tools
- `POST /skills/reload` — force full reload

## Included Example
- `services/edison_core/skills/weather_skill.py`
