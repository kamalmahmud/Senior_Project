from pathlib import Path
from ivthermal.config_schema import load_config

def test_load_config_ok():
    cfg = load_config(Path("config.yaml"))
    assert cfg.contract.H == 112
    assert cfg.clip.T in (16, 32)