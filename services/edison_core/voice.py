"""
Minimal voice command parser for agent modes and tool triggers.
"""

from typing import Dict


def parse_voice_command(text: str) -> Dict[str, str]:
    text_lower = text.lower().strip()

    if "instant" in text_lower:
        return {"action": "set_mode", "mode": "instant"}
    if "thinking" in text_lower or "think" in text_lower:
        return {"action": "set_mode", "mode": "thinking"}
    if "swarm" in text_lower:
        return {"action": "set_mode", "mode": "swarm"}
    if "agent" in text_lower:
        return {"action": "set_mode", "mode": "agent"}
    if "work mode" in text_lower:
        return {"action": "set_mode", "mode": "work"}
    if "open settings" in text_lower:
        return {"action": "open_settings"}
    if "close settings" in text_lower:
        return {"action": "close_settings"}

    return {"action": "unknown", "message": "No voice command matched"}
