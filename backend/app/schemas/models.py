from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EventRecord(BaseModel):
    event_id: str
    timestamp: datetime
    user_id: str
    source_ip: str
    failed_logins: int
    login_hour: int
    geo_anomaly: bool
    asset_id: str
    asset_type: str
    asset_criticality: str
    threat_intel_score: float
    previous_alerts: int


class Correction(BaseModel):
    event_id: str
    feature: str
    old_value: Any
    new_value: Any
    timestamp: Optional[datetime] = None
    reason: str = "User supplied correction"


class DecisionRecord(BaseModel):
    event_id: str
    decision_id: str
    decision_type: str
    input_features: Dict[str, Any] = Field(default_factory=dict)
    model_version: str
    historical_output: str
    timestamp: datetime
    provenance: Dict[str, Any] = Field(default_factory=dict)


class DecisionImpact(BaseModel):
    decision_id: str
    decision_type: str
    historical_value: str
    counterfactual_value: str
    affected: bool
    recovery_action: str
    verification_status: str


class VerificationResult(BaseModel):
    verification: str
    reasons: List[str] = Field(default_factory=list)
    decision_id: str
    event_id: str


class ExplainRequest(BaseModel):
    event_id: str
    correction: Correction
    analysis: Dict[str, Any]


class AIChatRequest(BaseModel):
    question: str
    event_id: Optional[str] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)
