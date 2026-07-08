from pathlib import Path
from typing import Optional

from .base import BaseDataset
from .ufpr import UFPRDataset
from .ccpd import CCPDDataset


def get_dataset(name: str, path: Optional[Path] = None, **kwargs) -> BaseDataset:
    """Get dataset by name."""
    datasets = {
        "ufpr-alpr": UFPRDataset,
        "ccpd": CCPDDataset,
    }
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")
    return datasets[name](path=path, **kwargs)
