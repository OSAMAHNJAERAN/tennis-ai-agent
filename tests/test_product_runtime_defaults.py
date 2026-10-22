"""Real environment defaults must select the requested private tennis pipeline."""
from pathlib import Path

import pytest

from src.api.jobs import REPO_ROOT, Settings
from src.api.worker import AnalysisError, check_models


def test_default_product_requires_accounts_and_selected_three_model_pipeline(monkeypatch):
    for name in ('TENNIS_PIPELINE_CONFIG', 'TENNIS_REQUIRE_AUTH', 'TENNIS_STORAGE_ROOT', 'TENNIS_SECURE_COOKIES'):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.config == (REPO_ROOT / 'configs/production/yolo26.yaml').resolve()
    assert settings.require_auth is True
    assert settings.root == (REPO_ROOT / 'outputs/product').resolve()
    with pytest.raises(AnalysisError, match='required verification are incomplete'):
        check_models(settings.config)


def test_historical_recovery_requires_explicit_environment_selection(monkeypatch, tmp_path):
    historical = REPO_ROOT / 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml'
    monkeypatch.setenv('TENNIS_PIPELINE_CONFIG', str(historical))
    monkeypatch.setenv('TENNIS_REQUIRE_AUTH', '0')
    monkeypatch.setenv('TENNIS_STORAGE_ROOT', str(tmp_path / 'recovery'))
    settings = Settings.from_env()
    assert settings.config == Path(historical).resolve()
    assert settings.require_auth is False
    assert settings.root == (tmp_path / 'recovery').resolve()
