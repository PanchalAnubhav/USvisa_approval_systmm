import pytest
import pandas as pd
from us_visa.pipeline.prediction_pipeline import USvisaData, USvisaClassifier


class TestPredictionPipeline:
    """Test suite for USvisaData and USvisaClassifier prediction pipeline."""

    def test_usvisa_data_dataframe_creation(self):
        """Verify USvisaData converts inputs into valid pandas DataFrame with correct columns."""
        data = USvisaData(
            continent="Asia",
            education_of_employee="Bachelor's",
            has_job_experience="Y",
            requires_job_training="N",
            no_of_employees=500,
            region_of_employment="Northeast",
            prevailing_wage=65000.0,
            unit_of_wage="Year",
            full_time_position="Y",
            company_age=15,
        )

        df = data.get_usvisa_input_data_frame()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 10)
        assert list(df.columns) == [
            "continent",
            "education_of_employee",
            "has_job_experience",
            "requires_job_training",
            "no_of_employees",
            "region_of_employment",
            "prevailing_wage",
            "unit_of_wage",
            "full_time_position",
            "company_age",
        ]
        assert df["continent"].iloc[0] == "Asia"
        assert df["prevailing_wage"].iloc[0] == 65000.0

    def test_classifier_prediction_output(self):
        """Verify classifier loads model and returns binary 0 or 1 prediction."""
        data = USvisaData(
            continent="North America",
            education_of_employee="Master's",
            has_job_experience="Y",
            requires_job_training="N",
            no_of_employees=5000,
            region_of_employment="Northeast",
            prevailing_wage=95000.0,
            unit_of_wage="Year",
            full_time_position="Y",
            company_age=20,
        )

        df = data.get_usvisa_input_data_frame()
        classifier = USvisaClassifier()
        prediction = classifier.predict(dataframe=df)

        assert prediction is not None
        assert len(prediction) == 1
        assert int(prediction[0]) in [0, 1]
