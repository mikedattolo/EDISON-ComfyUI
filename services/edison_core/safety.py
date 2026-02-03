"""
Confidence and safety scaffolding.
"""

from typing import Dict, Any

from .contracts import SafetyAssessmentRequest, SafetyAssessmentResponse


class SafetyGate:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config.get("safety", {}).get("thresholds", {
            "auto_block": 0.9,
            "require_confirm": 0.5
        })

    def assess(self, req: SafetyAssessmentRequest) -> SafetyAssessmentResponse:
        auto_block = self.thresholds.get("auto_block", 0.9)
        confirm = self.thresholds.get("require_confirm", 0.5)

        if req.risk_score >= auto_block:
            return SafetyAssessmentResponse(
                allowed=False,
                requires_confirmation=False,
                rationale="Risk score exceeds auto-block threshold"
            )

        if req.risk_score >= confirm:
            return SafetyAssessmentResponse(
                allowed=True,
                requires_confirmation=True,
                rationale="Risk score requires confirmation"
            )

        return SafetyAssessmentResponse(
            allowed=True,
            requires_confirmation=False,
            rationale="Risk score within safe threshold"
        )
