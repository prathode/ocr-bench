from typing import Dict, Any, List
from ocrbench.recommendation.profiles import DeploymentProfile, PROFILE_REGISTRY


class RecommendationEngine:
    """Rule-based recommendation engine."""

    def __init__(self):
        self.profiles = PROFILE_REGISTRY

    def rank_pipelines(
        self,
        results: List[Dict[str, Any]],
        profile_name: str = "balanced"
    ) -> List[Dict[str, Any]]:
        """Rank OCR pipelines based on deployment profile."""
        if profile_name not in self.profiles:
            raise ValueError(f"Unknown profile: {profile_name}")

        profile = self.profiles[profile_name]()
        weightings = profile.weightings

        def score(result: Dict[str, Any]) -> float:
            score_val = 0.0
            for metric, weight in weightings.items():
                value = result.get(metric, result.get("ocr_metrics", {}).get(metric, 0))
                if metric in ["cer", "wer", "latency_p95", "cost_per_image", "model_size_mb", "memory_usage_gb"]:
                    score_val += weight * (1 - value) if metric in ["cer", "wer", "latency_p95", "cost_per_image"] else weight * (1000 - value) / 1000
                else:
                    score_val += weight * value
            return score_val

        return sorted(results, key=score, reverse=True)

    def get_recommendations(
        self,
        results: List[Dict[str, Any]],
        profile_name: str = "balanced",
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Get top-k recommendations for a profile."""
        ranked = self.rank_pipelines(results, profile_name)
        return ranked[:top_k]
