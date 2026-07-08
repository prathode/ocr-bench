#!/usr/bin/env python
"""OCR-Bench CLI entry point."""
import argparse
from ocrbench.benchmark import run_benchmark
from ocrbench.reports import ReportExporter


def main():
    parser = argparse.ArgumentParser(description="OCR-Bench CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    benchmark_parser = subparsers.add_parser("benchmark", help="Run OCR benchmark")
    benchmark_parser.add_argument("--providers", nargs="+", default=["paddleocr"])
    benchmark_parser.add_argument("--dataset", default="ufpr-alpr")
    benchmark_parser.add_argument("--mode", default="ocr-only")
    benchmark_parser.add_argument("--batch-size", type=int, default=8)
    
    train_parser = subparsers.add_parser("train", help="Train/fine-tune models")
    train_parser.add_argument("--mode", choices=["ocr", "detector"], default="ocr")
    train_parser.add_argument("--model", default="trocr", help="Model to train (trocr, yolov8s, yolov11s, yolov12s, rtdetr)")
    train_parser.add_argument("--dataset", default="ufpr-alpr")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    if args.command == "benchmark":
        results = run_benchmark(
            providers=args.providers,
            dataset=args.dataset,
            mode=args.mode,
            batch_size=args.batch_size
        )
        print(f"Benchmark completed for {len(results)} providers")
        
        exporter = ReportExporter()
        exporter.export(results)
        print("Report exported to outputs/")
        
    elif args.command == "train":
        from ocrbench.training import run_training
        results = run_training(
            mode=args.mode,
            model_name=args.model,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        print(f"Training completed: {results}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()