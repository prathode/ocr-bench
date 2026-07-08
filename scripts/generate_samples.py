#!/usr/bin/env python
"""Generate sample license plate images for testing."""
from pathlib import Path
import numpy as np
import cv2


def generate_sample_images(output_dir: Path, count: int = 20):
    """Generate synthetic license plate images for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(count):
        img = np.ones((30, 120, 3), dtype=np.uint8) * 255
        
        plate_text = f"ABC-{i:03d}"
        
        cv2.putText(
            img,
            plate_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        
        filename = f"car-000000-000000-{plate_text.replace('-', '-')}-000000.jpg"
        cv2.imwrite(str(output_dir / filename), img)
    
    print(f"Generated {count} sample images in {output_dir}")


if __name__ == "__main__":
    generate_sample_images(Path("data/ufpr-alpr"), 20)