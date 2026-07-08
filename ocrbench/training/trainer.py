from typing import Dict, Any, Literal, List
from pathlib import Path

from ocrbench.datasets import get_dataset
from ocrbench.experiments import MLflowTracker


def run_training(
    mode: Literal["ocr", "detector"] = "ocr",
    model_name: str = None,
    dataset: str = "ufpr-alpr",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    output_dir: str = "outputs/models"
) -> Dict[str, Any]:
    """Run training on specified mode and dataset."""
    trainer = Trainer(
        mode=mode,
        model_name=model_name,
        dataset_name=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir
    )
    return trainer.train()


class Trainer:
    """Train OCR or detector models."""

    OCR_MODELS = {
        "trocr": "microsoft/trocr-base-printed",
        "easyocr": None,
        "paddleocr": None,
        "tesseract": None,
        "rapidocr": None,
    }

    DETECTOR_MODELS = {
        "yolov8s": "yolov8s.pt",
        "yolov11s": "yolov11s.pt",
        "yolov12s": "yolov12s.pt",
        "rtdetr": "rtdetr-l.pt",
    }

    def __init__(
        self,
        mode: Literal["ocr", "detector"] = "ocr",
        model_name: str = None,
        dataset_name: str = "ufpr-alpr",
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        output_dir: str = "outputs/models"
    ):
        self.mode = mode
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = MLflowTracker()

    def train(self) -> Dict[str, Any]:
        if self.mode == "ocr":
            return self._train_ocr()
        elif self.mode == "detector":
            return self._train_detector()
        else:
            raise ValueError(f"Unknown training mode: {self.mode}")

    def _train_ocr(self) -> Dict[str, Any]:
        """Fine-tune OCR model on custom dataset. Only TrOCR supports fine-tuning."""
        if self.model_name is None:
            self.model_name = "trocr"
        
        if self.model_name not in self.OCR_MODELS:
            raise ValueError(f"Unknown OCR model: {self.model_name}. Available: {list(self.OCR_MODELS.keys())}")

        if self.model_name == "trocr":
            return self._train_trocr()
        else:
            raise ValueError(
                f"{self.model_name} does not support fine-tuning. "
                "Only TrOCR (microsoft/trocr-base-printed) supports training."
            )

    def _train_trocr(self) -> Dict[str, Any]:
        """Fine-tune TrOCR model on custom dataset."""
        from PIL import Image
        
        self.tracker.start_run(run_name="trocr_finetune")
        
        dataset = get_dataset(self.dataset_name, subset_size=50)
        dataset.load()
        images = dataset.get_images()
        ground_truths = dataset.get_ground_truth()
        
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        model_name = "microsoft/trocr-base-printed"
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        
        train_losses = []
        
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(images), self.batch_size):
                batch_images = images[i:i + self.batch_size]
                batch_labels = ground_truths[i:i + self.batch_size]
                
                pil_images = [Image.fromarray(img) for img in batch_images]
                pixel_values = processor(images=pil_images, return_tensors="pt", padding=True).pixel_values.to(device)
                
                labels = processor.tokenizer(batch_labels, return_tensors="pt", padding=True).input_ids.to(device)
                
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            train_losses.append(avg_loss)
            self.tracker.log_metrics({"train_loss": avg_loss}, step=epoch)
        
        save_path = self.output_dir / f"{self.model_name}-finetuned"
        model.save_pretrained(str(save_path))
        processor.save_pretrained(str(save_path))
        
        self.tracker.end_run()
        
        return {
            "mode": "ocr",
            "model": model_name,
            "epochs": self.epochs,
            "train_losses": train_losses,
            "final_loss": train_losses[-1] if train_losses else None,
            "model_path": str(save_path),
        }

    def _train_detector(self) -> Dict[str, Any]:
        """Fine-tune YOLO detector on custom dataset."""
        import yaml
        
        if self.model_name is None:
            self.model_name = "yolov8s"
        
        if self.model_name not in self.DETECTOR_MODELS:
            raise ValueError(f"Unknown detector model: {self.model_name}. Available: {list(self.DETECTOR_MODELS.keys())}")

        self.tracker.start_run(run_name=f"{self.model_name}_finetune")
        
        dataset = get_dataset(self.dataset_name, subset_size=50)
        dataset.load()
        
        yaml_config = {
            "path": str(dataset.path),
            "train": "images",
            "val": "images",
            "nc": 1,
            "names": ["license_plate"]
        }
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_config, f)
        
        from ultralytics import YOLO
        
        model_file = self.DETECTOR_MODELS[self.model_name]
        model = YOLO(model_file)
        
        results = model.train(
            data=str(yaml_path),
            epochs=self.epochs,
            batch=self.batch_size,
            lr0=self.learning_rate,
            project=str(self.output_dir),
            name=f"{self.model_name}-plate",
        )
        
        self.tracker.end_run()
        
        return {
            "mode": "detector",
            "model": self.model_name,
            "epochs": self.epochs,
            "model_path": str(self.output_dir / f"{self.model_name}-plate"),
        }

    @classmethod
    def get_ocr_models(cls) -> List[str]:
        """Get list of OCR models that can be trained."""
        return list(cls.OCR_MODELS.keys())

    @classmethod
    def get_detector_models(cls) -> List[str]:
        """Get list of detector models that can be trained."""
        return list(cls.DETECTOR_MODELS.keys())