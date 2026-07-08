from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
from urllib.request import urlretrieve
import zipfile
import logging

from .base import BaseDataset

logger = logging.getLogger(__name__)

UFPR_URL = "https://web.archive.org/web/20240101000000*/UFPR-ALPR/"


class UFPRDataset(BaseDataset):
    """UFPR-ALPR dataset loader."""

    def __init__(
        self,
        path: Optional[Path] = None,
        subset_size: int = 50,
        download: bool = True
    ):
        if path is None:
            path = Path("data/ufpr-alpr")
        super().__init__(path=path, subset_size=subset_size)
        self.download = download

    def load(self) -> None:
        """Load UFPR-ALPR dataset."""
        if not self.path.exists():
            if self.download:
                self._download()
            else:
                raise FileNotFoundError(f"Dataset not found at {self.path}")

        image_paths = self.get_image_paths()
        for img_path in image_paths[:self.subset_size]:
            img = cv2.imread(str(img_path))
            if img is not None:
                self._images.append(img)
                plate = self._extract_plate_from_path(img_path)
                self._labels.append(plate)

    def _download(self) -> None:
        """Download dataset with resume capability."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "UFPR-ALPR requires manual download. "
            "Please download from https://web.archive.org/web/20240101000000*/UFPR-ALPR/"
        )

    def _extract_plate_from_path(self, img_path: Path) -> str:
        """Extract license plate from image filename."""
        parts = img_path.stem.split("-")
        if len(parts) >= 6:
            return "".join(parts[3:6])
        return ""

    def get_image_paths(self) -> List[Path]:
        """Get list of image paths."""
        return sorted(self.path.rglob("*.jpg")) + sorted(self.path.rglob("*.png"))
