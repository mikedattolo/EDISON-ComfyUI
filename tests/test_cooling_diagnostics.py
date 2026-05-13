from services.edison_core.cooling_diagnostics import (
    CoolingPolicy,
    aggregate_cooling_health,
    classify_cooling_health,
)


def test_zero_rpm_idle_is_not_marked_failed():
    health = classify_cooling_health(
        temperature_c=34,
        utilization_percent=0,
        fan_speed_percent=0,
        policy=CoolingPolicy(zero_rpm_idle_temp_c=45),
    )

    assert health.state == "zero_rpm_idle_probably_normal"
    assert health.severity == "info"


def test_zero_rpm_under_load_is_suspicious():
    health = classify_cooling_health(
        temperature_c=66,
        utilization_percent=58,
        fan_speed_percent=0,
        policy=CoolingPolicy(elevated_temp_c=55, active_utilization_percent=25),
    )

    assert health.state == "suspect_fan_not_spinning"
    assert health.severity == "warning"
    assert any("elevated" in item or "utilization" in item for item in health.evidence)


def test_missing_cooling_telemetry_is_unknown_not_failure():
    health = classify_cooling_health(telemetry_available=False)

    assert health.state == "telemetry_unavailable"
    assert health.severity == "unknown"


def test_critical_temperature_is_reported_as_over_temp():
    health = classify_cooling_health(temperature_c=86, utilization_percent=10, fan_speed_percent=40)

    assert health.state == "over_temp_warning"
    assert health.severity == "critical"


def test_aggregate_cooling_health_keeps_alerts():
    normal = classify_cooling_health(temperature_c=45, utilization_percent=5, fan_speed_percent=40)
    suspect = classify_cooling_health(temperature_c=70, utilization_percent=60, fan_speed_percent=0)

    summary = aggregate_cooling_health([normal, suspect])

    assert summary["overall_severity"] == "warning"
    assert summary["counts_by_state"]["suspect_fan_not_spinning"] == 1
    assert summary["alerts"][0]["state"] == "suspect_fan_not_spinning"
