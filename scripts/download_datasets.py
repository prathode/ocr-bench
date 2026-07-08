#!/usr/bin/env python
"""Download script for OCR-Bench datasets."""
from pathlib import Path

DATASETS = {
    "ufpr-alpr": {
        "url": "https://web.archive.org/web/20240101000000*/UFPR-ALPR/",
        "description": "UFPR-ALPR dataset requires manual download due to licensing",
    },
    "ccpd": {
        "url": "https://github.com/detectloop/crop_license_plate",
        "description": "CCPD dataset - Chinese City Parking Dataset",
    },
}

def download_dataset(name: str, output_dir: Path = None):
    if output_dir is None:
        output_dir = Path("data") / name
    
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    
    print(f"Please download {name} manually from {DATASETS[name]['url']}")
    print(f"Expected location: {output_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python download_datasets.py <dataset_name>")
        sys.exit(1)
    download_dataset(sys.argv[1])
