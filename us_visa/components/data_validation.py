import json
import sys

import pandas as pd
from pandas import DataFrame

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise USvisaException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates if all the schema columns exist in the dataframe

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return len(missing_categorical_columns) == 0 and len(missing_numerical_columns) == 0

        except Exception as e:
            raise USvisaException(e, sys) from e

    @staticmethod
    def detect_dataset_drift(reference_df: DataFrame, current_df: DataFrame, threshold: float = 0.05) -> bool:
        """
        Method Name :   detect_dataset_drift
        Description :   This method detects data drift between reference and current dataframes
                         using a simple statistical approach (KS test per numerical column).

        Output      :   Returns True if drift is detected, False otherwise.
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            from scipy.stats import ks_2samp

            status = True
            drift_report = {}

            for column in reference_df.select_dtypes(include=["number"]).columns:
                d, p_value = ks_2samp(reference_df[column], current_df[column])
                if p_value <= threshold:
                    is_found = True
                else:
                    is_found = False
                drift_report[column] = {
                    "p_value": float(p_value),
                    "drift_status": is_found,
                }

                if is_found:
                    status = True

            # If no columns showed drift, status is False
            drift_detected = any(col_report["drift_status"] for col_report in drift_report.values())

            return drift_detected

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component

        Output      :   DataValidationArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Entered initiate_data_validation method of DataValidation class")
            validation_error_msg = ""

            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += "Columns are missing in training dataframe. "

            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += "Columns are missing in test dataframe. "

            # Validate column existence
            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += "Columns are missing in training dataframe. "

            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += "Columns are missing in test dataframe. "

            validation_status = len(validation_error_msg) == 0

            # Detect dataset drift
            if validation_status:
                drift_status = self.detect_dataset_drift(
                    reference_df=train_df, current_df=test_df
                )
                if drift_status:
                    logging.info("Drift detected between training and test dataset.")
                    validation_error_msg = "Drift detected"
                else:
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation error: {validation_error_msg}")

            # Write drift report
            drift_report = {"validation_status": validation_status, "message": validation_error_msg}
            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=drift_report,
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            logging.info("Exited initiate_data_validation method of DataValidation class")

            return data_validation_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e
