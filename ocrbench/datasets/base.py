from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


class BaseDataset(ABC):
    """Base interface for OCR datasets."""

    def __init__(self, path: Optional[Path] = None, subset_size: int = 50):
        self.path = path
        self.subset_size = subset_size
        self._images: List[np.ndarray] = []
        self._labels: List[str] = []

    @abstractmethod
    def load(self) -> None:
        """Load dataset from disk or download if needed."""
        pass

    @abstractmethod
    def get_image_paths(self) -> List[Path]:
        """Get list of image paths."""
        pass

    def get_ground_truth(self) -> List[str]:
        """Get ground truth labels."""
        return self._labels

    def get_images(self) -> List[np.ndarray]:
        """Get loaded images."""
        return self._images

    def __len__(self) -> int:
        return len(self._images)
