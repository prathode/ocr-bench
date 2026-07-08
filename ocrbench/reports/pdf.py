from typing import Dict, Any, List
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


class PDFReport:
    """Generate PDF benchmark report."""

    def generate(self, results: List[Dict], output_path: Path) -> None:
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("OCRBench Benchmark Report", styles["Title"]))
        story.append(Spacer(1, 12))

        for result in results:
            story.append(Paragraph(f"Provider: {result.get('provider_name', 'Unknown')}", styles["Heading1"]))
            
            ocr_metrics = result.get("ocr_metrics", {})
            if ocr_metrics:
                story.append(Paragraph("OCR Metrics:", styles["Heading2"]))
                for key, value in ocr_metrics.items():
                    story.append(Paragraph(f"  {key}: {value}", styles["Normal"]))
            
            perf_metrics = result.get("performance_metrics", {})
            if perf_metrics:
                story.append(Paragraph("Performance:", styles["Heading2"]))
                for key, value in perf_metrics.items():
                    story.append(Paragraph(f"  {key}: {value}", styles["Normal"]))
            
            story.append(Spacer(1, 12))

        doc.build(story)
