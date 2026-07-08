import gradio as gr
from ocrbench.benchmark import run_benchmark
from ocrbench.training import run_training
from ocrbench.recommendation import RecommendationEngine
from ocrbench.reports import ReportExporter


def create_ui():
    """Create Gradio UI."""
    recommender = RecommendationEngine()
    exporter = ReportExporter()

    with gr.Blocks(title="OCRBench") as demo:
        gr.Markdown("# OCRBench - OCR Benchmarking Platform")

        with gr.Tab("Run Benchmark"):
            with gr.Row():
                providers = gr.CheckboxGroup(
                    choices=["paddleocr", "easyocr", "tesseract", "trocr", "rapidocr"],
                    value=["paddleocr"],
                    label="OCR Providers"
                )
                dataset = gr.Dropdown(
                    choices=["ufpr-alpr", "ccpd"],
                    value="ufpr-alpr",
                    label="Dataset"
                )

            benchmark_btn = gr.Button("Run Benchmark")
            results = gr.JSON(label="Results")

            benchmark_btn.click(
                fn=run_benchmark_ui,
                inputs=[providers, dataset],
                outputs=results
            )

        with gr.Tab("Training"):
            with gr.Row():
                train_mode = gr.Radio(
                    choices=["ocr", "detector"],
                    value="ocr",
                    label="Training Mode"
                )
                train_dataset = gr.Dropdown(
                    choices=["ufpr-alpr", "ccpd"],
                    value="ufpr-alpr",
                    label="Dataset"
                )

            with gr.Row():
                train_model = gr.Dropdown(
                    choices=["trocr", "yolov8s", "yolov11s", "yolov12s", "rtdetr"],
                    value="trocr",
                    label="Model"
                )
                epochs = gr.Slider(1, 100, value=10, step=1, label="Epochs")
                batch_size = gr.Slider(1, 32, value=8, step=1, label="Batch Size")
                learning_rate = gr.Number(value=1e-4, label="Learning Rate")

            train_btn = gr.Button("Start Training")
            train_results = gr.JSON(label="Training Results")

            train_btn.click(
                fn=train_model_ui,
                inputs=[train_mode, train_model, train_dataset, epochs, batch_size, learning_rate],
                outputs=train_results
            )

        with gr.Tab("Recommendations"):
            profile = gr.Dropdown(
                choices=["balanced", "real-time", "high-accuracy", "low-cost", "edge"],
                value="balanced",
                label="Deployment Profile"
            )
            benchmark_results = gr.State([])
            recommend_btn = gr.Button("Get Recommendations")
            recommendations = gr.JSON(label="Top Recommendations")

            recommend_btn.click(
                fn=get_recommendations_ui,
                inputs=[benchmark_results, profile],
                outputs=recommendations
            )

    return demo


def run_benchmark_ui(providers, dataset):
    results = run_benchmark(providers=providers, dataset=dataset)
    return results


def train_model_ui(mode, model_name, dataset, epochs, batch_size, learning_rate):
    results = run_training(
        mode=mode,
        model_name=model_name,
        dataset=dataset,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate)
    )
    return results


def get_recommendations_ui(results, profile):
    if not results:
        return {"error": "Run benchmark first"}
    recommender = RecommendationEngine()
    return recommender.get_recommendations(results, profile_name=profile)


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()