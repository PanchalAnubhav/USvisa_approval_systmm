import os
import pytest
import pandas as pd
from us_visa.components.data_ingestion import DataIngestion
from us_visa.entity.config_entity import DataIngestionConfig


class TestDataIngestion:
    """Test suite for DataIngestion component and fallback mechanism."""

    def test_data_ingestion_fallback_file_exists(self):
        """Ensure the local EasyVisa.csv fallback file is available."""
        fallback_path = os.path.join("notebook", "EasyVisa.csv")
        assert os.path.exists(fallback_path), f"Fallback dataset not found at {fallback_path}"

        df = pd.read_csv(fallback_path)
        assert df.shape[0] > 0
        assert "case_status" in df.columns
