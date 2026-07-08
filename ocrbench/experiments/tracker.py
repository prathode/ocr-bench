from pathlib import Path
from typing import Optional, Dict, Any
import mlflow
import mlflow.pytorch
import time
import uuid


class MLflowTracker:
    """MLflow experiment tracking wrapper."""

    def __init__(
        self,
        experiment_name: str = "ocrbench",
        tracking_uri: str = None
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            mlflow.create_experiment(experiment_name, artifact_location="artifacts/")
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start a new MLflow run."""
        run = mlflow.start_run(run_name=run_name or f"run_{int(time.time())}")
        return run.info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to current run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Log metrics to current run."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: Path, artifact_path: str = None) -> None:
        """Log artifact to current run."""
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_dict(self, dictionary: Dict, artifact_file: str = "data.json") -> None:
        """Log dictionary as artifact."""
        mlflow.log_dict(dictionary, artifact_file)

    def end_run(self) -> None:
        """End current run."""
        mlflow.end_run()

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Get run data by ID."""
        return mlflow.get_run(run_id).to_dictionary()
