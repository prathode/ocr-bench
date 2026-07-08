from typing import Dict, Any, List
from pathlib import Path
import json
import pandas as pd

from .pdf import PDFReport
from .csv import CSVReport
from .json import JSONReport
from .markdown import MarkdownReport


class ReportExporter:
    """Export benchmark results in multiple formats."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        results: List[Dict[str, Any]],
        formats: List[str] = None
    ) -> Dict[str, Path]:
        """Export results in specified formats."""
        if formats is None:
            formats = ["json", "csv", "markdown"]

        exported = {}
        for fmt in formats:
            if fmt == "json":
                exported["json"] = self._export_json(results)
            elif fmt == "csv":
                exported["csv"] = self._export_csv(results)
            elif fmt == "pdf":
                exported["pdf"] = self._export_pdf(results)
            elif fmt == "markdown":
                exported["markdown"] = self._export_markdown(results)

        return exported

    def _export_json(self, results: List[Dict]) -> Path:
        path = self.output_dir / "report.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return path

    def _export_csv(self, results: List[Dict]) -> Path:
        path = self.output_dir / "report.csv"
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)
        return path

    def _export_pdf(self, results: List[Dict]) -> Path:
        path = self.output_dir / "report.pdf"
        pdf = PDFReport()
        pdf.generate(results, path)
        return path

    def _export_markdown(self, results: List[Dict]) -> Path:
        path = self.output_dir / "report.md"
        md = MarkdownReport()
        md.generate(results, path)
        return path
