import pytest
import numpy as np
from ocrbench.metrics import calculate_cer, calculate_cer_batch


class TestMetrics:
    def test_cer_identical(self):
        assert calculate_cer("hello", "hello") == 0.0

    def test_cer_different(self):
        assert calculate_cer("hello", "hallo") == 0.2

    def test_cer_empty(self):
        assert calculate_cer("", "") == 0.0
        assert calculate_cer("hello", "") == 1.0

    def test_cer_batch(self):
        results = calculate_cer_batch(["hello", "world"], ["hello", "word"])
        assert "cer_mean" in results
        assert results["cer_mean"] == 0.1
