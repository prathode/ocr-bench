from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class CostTracker:
    """Track API costs for LLM OCR providers."""

    costs_per_1k: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.costs_per_1k:
            self.costs_per_1k = {
                "gpt-4o": 0.03,
                "gpt-4o-mini": 0.0015,
                "gpt-4-turbo": 0.01,
                "gpt-3.5-turbo": 0.0005,
                "gemini-pro": 0.0005,
                "gemini-flash": 0.0000005,
            }

    def calculate_cost(
        self,
        provider: str,
        model: Optional[str],
        token_count: int
    ) -> float:
        """Calculate cost for token usage."""
        model_key = model if model else "default"
        if model_key in self.costs_per_1k:
            return (self.costs_per_1k[model_key] / 1000) * token_count
        return 0.0

    def get_estimated_monthly_cost(
        self,
        provider: str,
        model: Optional[str],
        token_per_image: int,
        images_per_month: int
    ) -> float:
        """Estimate monthly cost for given usage."""
        return self.calculate_cost(provider, model, token_per_image * images_per_month)
