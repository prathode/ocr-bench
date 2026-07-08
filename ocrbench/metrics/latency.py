import time
from typing import List


def calculate_fps(processing_times_ms: List[float]) -> float:
    """Calculate frames per second from processing times."""
    if not processing_times_ms:
        return 0.0
    
    avg_time_seconds = sum(processing_times_ms) / len(processing_times_ms) / 1000
    return 1.0 / avg_time_seconds if avg_time_seconds > 0 else 0.0


def calculate_latency(processing_times_ms: List[float]) -> dict:
    """Calculate latency statistics."""
    if not processing_times_ms:
        return {
            "latency_avg_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p90_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
            "latency_min_ms": 0.0,
            "latency_max_ms": 0.0,
        }
    
    import numpy as np
    arr = np.array(processing_times_ms)
    
    return {
        "latency_avg_ms": float(np.mean(arr)),
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p90_ms": float(np.percentile(arr, 90)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_p99_ms": float(np.percentile(arr, 99)),
        "latency_min_ms": float(np.min(arr)),
        "latency_max_ms": float(np.max(arr)),
    }
