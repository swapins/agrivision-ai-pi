from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    STRESS = "STRESS"
    DISEASE = "DISEASE"
    UNCERTAIN = "UNCERTAIN"
    NOT_SCANNED = "NOT SCANNED"


def moisture_percent(raw: float, dry_raw: float, wet_raw: float) -> float:
    """Map a raw ADC value to 0..100%, working in either sensor direction."""
    if dry_raw == wet_raw:
        raise ValueError("dry_raw and wet_raw cannot be equal")
    pct = (raw - dry_raw) * 100.0 / (wet_raw - dry_raw)
    return max(0.0, min(100.0, pct))


def label_to_health_status(label: str, confidence: float, minimum: float) -> HealthStatus:
    """Map flexible model labels to the three exhibition statuses.

    A multiclass disease label such as ``Tomato___Late_blight`` maps to
    DISEASE, while any label containing ``healthy`` maps to HEALTHY.
    Dedicated stress labels map to STRESS. Low-confidence outputs become
    UNCERTAIN.
    """
    if confidence < minimum:
        return HealthStatus.UNCERTAIN
    low = label.strip().lower().replace("-", "_")
    if "healthy" in low or low in {"normal", "ok"}:
        return HealthStatus.HEALTHY
    if any(token in low for token in ("stress", "stressed", "wilt", "yellow")):
        return HealthStatus.STRESS
    if low in {"problem", "unhealthy", "disease", "diseased"}:
        return HealthStatus.DISEASE
    # For PlantVillage multiclass labels, every non-healthy class is a disease class.
    if "___" in label or low:
        return HealthStatus.DISEASE
    return HealthStatus.UNCERTAIN


@dataclass
class IrrigationDecision:
    should_start: bool
    reason: str


def irrigation_start_decision(
    moisture: float,
    on_below: float,
    seconds_since_last_run: float,
    cooldown_seconds: float,
) -> IrrigationDecision:
    if moisture >= on_below:
        return IrrigationDecision(False, "moisture above start threshold")
    if seconds_since_last_run < cooldown_seconds:
        return IrrigationDecision(False, "pump cooldown active")
    return IrrigationDecision(True, "dry soil")
