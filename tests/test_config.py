from pathlib import Path
from agrivision.config import load_config


def test_example_config_loads():
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root/'config.example.yaml')
    assert cfg.section('pump')['gpio'] == 17
    assert cfg.section('model')['confidence_min'] == 0.60
