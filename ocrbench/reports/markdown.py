from typing import Dict, Any, List
from pathlib import Path


class MarkdownReport:
    """Generate Markdown benchmark report."""

    def generate(self, results: List[Dict], output_path: Path) -> None:
        lines = ["# OCRBench Benchmark Report\n"]

        for result in results:
            provider = result.get("provider_name", "Unknown")
            lines.append(f"\n## Provider: {provider}\n")

            ocr_metrics = result.get("ocr_metrics", {})
            if ocr_metrics:
                lines.append("\n### OCR Metrics\n")
                for key, value in ocr_metrics.items():
                    lines.append(f"- **{key}**: {value}\n")

            perf_metrics = result.get("performance_metrics", {})
            if perf_metrics:
                lines.append("\n### Performance Metrics\n")
                for key, value in perf_metrics.items():
                    lines.append(f"- **{key}**: {value}\n")

            resource_metrics = result.get("resource_metrics", {})
            if resource_metrics:
                lines.append("\n### Resource Metrics\n")
                for key, value in resource_metrics.items():
                    lines.append(f"- **{key}**: {value}\n")

        with open(output_path, "w") as f:
            f.writelines(lines)
