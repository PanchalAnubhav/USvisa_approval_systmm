import os
import sys

import numpy as np
import pandas as pd
from pandas import DataFrame

from us_visa.entity.config_entity import USvisaPredictorConfig
from us_visa.entity.s3_estimator import USvisaEstimator
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file


class USvisaData:
    """
    Input data class for US Visa prediction. Matches the features used
    in the notebook after feature engineering (drop case_id, yr_of_estab;
    add company_age).
    """
    def __init__(
        self,
        continent: str,
        education_of_employee: str,
        has_job_experience: str,
        requires_job_training: str,
        no_of_employees: int,
        region_of_employment: str,
        prevailing_wage: float,
        unit_of_wage: str,
        full_time_position: str,
        company_age: int,
    ):
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_usvisa_input_data_frame(self) -> DataFrame:
        """
        Create a DataFrame from the input data for prediction.

        Output      :   DataFrame with a single row of input features.
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            usvisa_input_dict = self.get_usvisa_data_as_dict()
            return DataFrame(usvisa_input_dict)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_usvisa_data_as_dict(self) -> dict:
        """
        Return the input data as a dictionary (list of values for DataFrame construction).
        """
        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }
            logging.info("Created usvisa data dict")
            return input_data
        except Exception as e:
            raise USvisaException(e, sys) from e


class USvisaClassifier:
    """
    Prediction pipeline that loads the model from S3 and runs inference.
    """
    def __init__(self, prediction_pipeline_config: USvisaPredictorConfig = USvisaPredictorConfig()):
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise USvisaException(e, sys)

    def predict(self, dataframe: DataFrame) -> str:
        """
        Method Name :   predict
        Description :   This method loads the model from S3 and makes a prediction.

        Output      :   Prediction result string ("Visa-Approved" or "Visa-Rejected")
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Entered predict method of USvisaClassifier class")

            model = USvisaEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_s3_key,
            )

            result = model.load_model().predict(dataframe)

            return result

        except Exception as e:
            raise USvisaException(e, sys)
