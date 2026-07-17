from src.config import SystemConfig, ADTCScoring, load_config


def test_default_config_is_qwen_only():
    config = SystemConfig()
    assert config.enable_secondary_model is False
    assert config.primary_model.name == "qwen3.5-0.8b"


def test_scoring_constants_match_adtc_spec():
    scoring = ADTCScoring()
    assert scoring.tps_reference == 15.0
    assert scoring.ram_budget_mb == 7000.0
    assert scoring.thermal_limit_c == 85.0
    assert scoring.weight_sacc == 0.50
    assert scoring.weight_sperf == 0.30
    assert scoring.weight_seff == 0.20


def test_load_config_creates_dirs(tmp_path, monkeypatch):
    config = load_config()
    assert config.log_dir.exists()
