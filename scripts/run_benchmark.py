import sys
from ocrbench.benchmark import run_benchmark
from ocrbench.reports import ReportExporter
from ocrbench.recommendation import RecommendationEngine


def main():
    providers = sys.argv[1:] if len(sys.argv) > 1 else ["paddleocr"]
    
    print(f"Running benchmark with providers: {providers}")
    results = run_benchmark(providers=providers)
    
    print("\nResults:")
    for r in results:
        print(f"  {r['provider_name']}: CER={r.get('ocr_metrics', {}).get('cer_mean', 'N/A')}")
    
    exporter = ReportExporter()
    exporter.export(results)
    print("\nReport exported to outputs/")


if __name__ == "__main__":
    main()