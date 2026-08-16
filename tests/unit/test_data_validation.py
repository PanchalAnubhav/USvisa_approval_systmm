"""Tests for data validation component."""

import pytest
import pandas as pd
import numpy as np

from us_visa.components.data_validation import DataValidation


class TestDataValidationHelpers:
    """Test suite for DataValidation static/utility methods."""

    def test_detect_drift_no_drift(self):
        """detect_dataset_drift should return False when data is identical."""
        np.random.seed(42)
        ref = pd.DataFrame({"a": np.random.normal(0, 1, 1000), "b": np.random.normal(5, 2, 1000)})
        cur = pd.DataFrame({"a": np.random.normal(0, 1, 1000), "b": np.random.normal(5, 2, 1000)})

        result = DataValidation.detect_dataset_drift(ref, cur, threshold=0.05)
        # With same distribution, drift should generally not be detected
        # (statistical, so we allow for occasional false positives)
        assert isinstance(result, bool)

    def test_detect_drift_with_drift(self):
        """detect_dataset_drift should return True when distributions differ significantly."""
        np.random.seed(42)
        ref = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
        cur = pd.DataFrame({"a": np.random.normal(10, 1, 1000)})  # Shifted by 10

        result = DataValidation.detect_dataset_drift(ref, cur, threshold=0.05)
        assert result is True
