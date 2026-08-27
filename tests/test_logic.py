from agrivision.logic import HealthStatus, irrigation_start_decision, label_to_health_status, moisture_percent


def test_moisture_mapping_decreasing_sensor():
    assert moisture_percent(21000, 21000, 9000) == 0
    assert moisture_percent(9000, 21000, 9000) == 100
    assert 49 < moisture_percent(15000, 21000, 9000) < 51


def test_moisture_clamps():
    assert moisture_percent(30000, 21000, 9000) == 0
    assert moisture_percent(0, 21000, 9000) == 100


def test_label_mapping():
    assert label_to_health_status("Tomato___healthy", .95, .6) == HealthStatus.HEALTHY
    assert label_to_health_status("Tomato___Late_blight", .95, .6) == HealthStatus.DISEASE
    assert label_to_health_status("water_stress", .95, .6) == HealthStatus.STRESS
    assert label_to_health_status("problem", .95, .6) == HealthStatus.DISEASE
    assert label_to_health_status("healthy", .4, .6) == HealthStatus.UNCERTAIN
    assert label_to_health_status("unexpected_label", .99, .6) == HealthStatus.UNCERTAIN


def test_irrigation_decision():
    assert irrigation_start_decision(20, 30, 100, 20).should_start
    assert not irrigation_start_decision(40, 30, 100, 20).should_start
    assert not irrigation_start_decision(20, 30, 5, 20).should_start
