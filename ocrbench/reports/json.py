from typing import Dict, Any, List
from pathlib import Path
import json


class JSONReport:
    """Generate JSON benchmark report."""

    def generate(self, results: List[Dict], output_path: Path) -> None:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
