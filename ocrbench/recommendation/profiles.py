from typing import Dict, Any, List
from enum import Enum
from dataclasses import dataclass


class ProfileName(str, Enum):
    REAL_TIME = "real-time"
    HIGH_ACCURACY = "high-accuracy"
    LOW_COST = "low-cost"
    EDGE = "edge"
    BALANCED = "balanced"


@dataclass
class DeploymentProfile:
    """Deployment profile with weightings for scoring."""
    name: str
    description: str
    weightings: Dict[str, float]

    @classmethod
    def real_time(cls) -> "DeploymentProfile":
        return cls(
            name="real-time",
            description="Optimized for speed and low latency",
            weightings={
                "fps": 0.4,
                "latency_p95": -0.3,
                "cer": -0.2,
                "model_size_mb": -0.1,
            }
        )

    @classmethod
    def high_accuracy(cls) -> "DeploymentProfile":
        return cls(
            name="high-accuracy",
            description="Optimized for highest accuracy",
            weightings={
                "cer": -0.4,
                "wer": -0.3,
                "exact_match": 0.2,
                "precision": 0.1,
            }
        )

    @classmethod
    def low_cost(cls) -> "DeploymentProfile":
        return cls(
            name="low-cost",
            description="Optimized for minimal cost",
            weightings={
                "cost_per_image": -0.5,
                "model_size_mb": -0.2,
                "cer": -0.2,
                "fps": 0.1,
            }
        )

    @classmethod
    def edge(cls) -> "DeploymentProfile":
        return cls(
            name="edge",
            description="Optimized for edge deployment (small models)",
            weightings={
                "model_size_mb": -0.4,
                "memory_usage_gb": -0.3,
                "fps": 0.2,
                "cer": -0.1,
            }
        )

    @classmethod
    def balanced(cls) -> "DeploymentProfile":
        return cls(
            name="balanced",
            description="Balanced performance across all metrics",
            weightings={
                "fps": 0.15,
                "cer": -0.25,
                "latency_p95": -0.2,
                "model_size_mb": -0.15,
                "cost_per_image": -0.15,
                "memory_usage_gb": -0.15,
            }
        )


PROFILE_REGISTRY: Dict[str, callable] = {
    "real-time": DeploymentProfile.real_time,
    "high-accuracy": DeploymentProfile.high_accuracy,
    "low-cost": DeploymentProfile.low_cost,
    "edge": DeploymentProfile.edge,
    "balanced": DeploymentProfile.balanced,
}
