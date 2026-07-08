from typing import Dict, Any, List
from pathlib import Path
import pandas as pd


class CSVReport:
    """Generate CSV benchmark report."""

    def generate(self, results: List[Dict], output_path: Path) -> None:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
