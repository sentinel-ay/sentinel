from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config(path=None) -> dict:
    """Load the project configuration as a plain dict."""
    with open(CONFIG_PATH if path is None else path, encoding="utf-8") as f:
        return yaml.safe_load(f)
