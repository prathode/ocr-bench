import psutil
import torch


def get_memory_usage() -> dict:
    """Get current memory usage."""
    memory_info = psutil.virtual_memory()
    
    result = {
        "cpu_memory_percent": memory_info.percent,
        "cpu_memory_used_gb": memory_info.used / (1024**3),
        "cpu_memory_available_gb": memory_info.available / (1024**3),
    }
    
    if torch.cuda.is_available():
        result["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        result["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        result["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return result


def get_model_size_mb(model_path: str) -> float:
    """Get model file size in MB."""
    from pathlib import Path
    path = Path(model_path)
    if path.exists():
        return path.stat().st_size / (1024**2)
    return 0.0
