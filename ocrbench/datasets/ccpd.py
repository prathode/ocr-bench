from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
import logging

from .base import BaseDataset

logger = logging.getLogger(__name__)


class CCPDDataset(BaseDataset):
    """CCPD license plate dataset loader."""

    def __init__(
        self,
        path: Optional[Path] = None,
        subset_size: int = 50
    ):
        if path is None:
            path = Path("data/ccpd")
        super().__init__(path=path, subset_size=subset_size)

    def load(self) -> None:
        """Load CCPD dataset."""
        if not self.path.exists():
            raise FileNotFoundError(
                f"CCPD dataset not found at {self.path}. "
                "Please download from https://github.com/detectloop/crop_license_plate"
            )

        image_paths = self.get_image_paths()
        for img_path in image_paths[:self.subset_size]:
            img = cv2.imread(str(img_path))
            if img is not None:
                self._images.append(img)
                plate, bbox = self._parse_filename(img_path)
                self._labels.append(plate)

    def _parse_filename(self, img_path: Path) -> tuple:
        """Parse CCPD filename to extract plate and bounding box."""
        filename = img_path.stem
        parts = filename.split("-")
        if len(parts) >= 4:
            plate = parts[3]
            return plate, None
        return "", None

    def get_image_paths(self) -> List[Path]:
        """Get list of image paths."""
        return sorted(self.path.rglob("*.jpg")) + sorted(self.path.rglob("*.png"))
