from pathlib import Path
from typing import Any, Dict
import hydra
from omegaconf import DictConfig, OmegaConf


def get_config() -> DictConfig:
    """Get the Hydra configuration."""
    return hydra.compose(config_name="config")


def setup_config(config_path: Path = None) -> DictConfig:
    """Initialize and return Hydra configuration."""
    if config_path is None:
        config_path = Path(__file__).parent
    return get_config()


def get_cache_dir() -> Path:
    """Get the cache directory for model downloads."""
    cache_dir = Path.home() / ".cache" / "ocrbench"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
