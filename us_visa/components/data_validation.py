import sys

import pandas as pd
from pandas import DataFrame

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: Configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            self._schema_config = read_yaml_file(
                file_path=SCHEMA_FILE_PATH
            )

        except Exception as e:
            raise USvisaException(e, sys) from e

    def validate_number_of_columns(
        self,
        dataframe: DataFrame
    ) -> bool:
        """
        Method Name : validate_number_of_columns

        Description :
            Validates that the dataframe contains the expected
            number of raw columns.

        Output :
            Returns True if the number of columns matches the schema.
        """
        try:
            status = (
                len(dataframe.columns)
                == len(self._schema_config["columns"])
            )

            logging.info(
                f"Is required column count present: [{status}]"
            )

            return status

        except Exception as e:
            raise USvisaException(e, sys) from e

    def is_column_exist(
        self,
        df: DataFrame
    ) -> bool:
        """
        Validate that all columns expected in the RAW dataset
        exist in the dataframe.

        Feature-engineered columns such as `company_age` are
        intentionally NOT validated here because they are created
        during the data transformation stage.
        """
        try:
            dataframe_columns = set(df.columns)

            # Columns expected directly from MongoDB / raw dataset.
            expected_columns = {
                column_name
                for column_config in self._schema_config["columns"]
                for column_name in column_config.keys()
            }

            missing_columns = expected_columns - dataframe_columns

            if missing_columns:
                logging.info(
                    f"Missing columns in dataframe: "
                    f"{sorted(missing_columns)}"
                )
                return False

            return True

        except Exception as e:
            raise USvisaException(e, sys) from e

    @staticmethod
    def detect_dataset_drift(
        reference_df: DataFrame,
        current_df: DataFrame,
        threshold: float = 0.05,
    ) -> bool:
        """
        Detect meaningful numerical data drift using the
        Kolmogorov-Smirnov (KS) test.

        The p-value alone is NOT used to reject the dataset because
        large datasets can produce very small p-values for relatively
        small distribution differences.

        Instead, a practical KS statistic threshold is used.

        Returns:
            True  -> meaningful drift detected
            False -> no meaningful drift detected
        """
        try:
            from scipy.stats import ks_2samp

            drift_report = {}

            # Practical drift threshold.
            # A KS statistic <= 0.10 is considered acceptable.
            ks_statistic_threshold = 0.10

            numerical_columns = reference_df.select_dtypes(
                include=["number"]
            ).columns

            for column in numerical_columns:

                if column not in current_df.columns:
                    logging.warning(
                        f"Column '{column}' exists in reference dataframe "
                        f"but not in current dataframe."
                    )
                    continue

                reference_values = reference_df[column].dropna()
                current_values = current_df[column].dropna()

                d_statistic, p_value = ks_2samp(
                    reference_values,
                    current_values,
                )

                is_found = (
                    d_statistic > ks_statistic_threshold
                )

                drift_report[column] = {
                    "ks_statistic": float(d_statistic),
                    "p_value": float(p_value),
                    "drift_status": is_found,
                }

                logging.info(
                    f"Drift check - {column}: "
                    f"KS={d_statistic:.4f}, "
                    f"p_value={p_value:.6f}, "
                    f"drift={is_found}"
                )

            drift_detected = any(
                report["drift_status"]
                for report in drift_report.values()
            )

            return drift_detected

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_validation(
        self
    ) -> DataValidationArtifact:
        """
        Method Name : initiate_data_validation

        Description :
            Initiates the data validation component.

        Steps:
            1. Load train and test dataframes.
            2. Validate number of raw columns.
            3. Validate existence of raw columns.
            4. Detect potential numerical data drift.
            5. Write validation/drift report.
            6. Create DataValidationArtifact.

        Output:
            DataValidationArtifact
        """
        try:
            logging.info(
                "Entered initiate_data_validation method "
                "of DataValidation class"
            )

            validation_error_msg = ""

            # ---------------------------------------------------------
            # Load train and test datasets
            # ---------------------------------------------------------

            train_df = pd.read_csv(
                self.data_ingestion_artifact.trained_file_path
            )

            test_df = pd.read_csv(
                self.data_ingestion_artifact.test_file_path
            )

            logging.info(
                f"Training dataframe shape: {train_df.shape}"
            )

            logging.info(
                f"Testing dataframe shape: {test_df.shape}"
            )

            logging.info(
                f"Training dataframe columns: "
                f"{list(train_df.columns)}"
            )

            logging.info(
                f"Testing dataframe columns: "
                f"{list(test_df.columns)}"
            )

            # ---------------------------------------------------------
            # Validate number of columns
            # ---------------------------------------------------------

            status = self.validate_number_of_columns(
                dataframe=train_df
            )

            logging.info(
                f"All required columns present in "
                f"training dataframe: {status}"
            )

            if not status:
                validation_error_msg += (
                    "Columns are missing in training dataframe. "
                )

            status = self.validate_number_of_columns(
                dataframe=test_df
            )

            logging.info(
                f"All required columns present in "
                f"testing dataframe: {status}"
            )

            if not status:
                validation_error_msg += (
                    "Columns are missing in test dataframe. "
                )

            # ---------------------------------------------------------
            # Validate raw column existence
            # ---------------------------------------------------------

            status = self.is_column_exist(
                df=train_df
            )

            if not status:
                validation_error_msg += (
                    "Columns are missing in training dataframe. "
                )

            status = self.is_column_exist(
                df=test_df
            )

            if not status:
                validation_error_msg += (
                    "Columns are missing in test dataframe. "
                )

            # At this point we have validated the actual RAW dataset.
            validation_status = (
                len(validation_error_msg) == 0
            )

            # ---------------------------------------------------------
            # Detect dataset drift
            # ---------------------------------------------------------

            if validation_status:

                drift_status = self.detect_dataset_drift(
                    reference_df=train_df,
                    current_df=test_df,
                )

                if drift_status:

                    logging.warning(
                        "Potential data drift detected between "
                        "training and test datasets."
                    )

                    # Drift is treated as a warning.
                    # It does NOT invalidate the dataset.
                    validation_error_msg = (
                        "Validation passed with potential drift"
                    )

                else:

                    logging.info(
                        "No significant drift detected between "
                        "training and test datasets."
                    )

                    validation_error_msg = "Validation passed"

            else:

                logging.error(
                    f"Validation error: {validation_error_msg}"
                )

            # ---------------------------------------------------------
            # Write validation / drift report
            # ---------------------------------------------------------

            drift_report = {
                "validation_status": validation_status,
                "message": validation_error_msg,
            }

            write_yaml_file(
                file_path=(
                    self.data_validation_config
                    .drift_report_file_path
                ),
                content=drift_report,
            )

            # ---------------------------------------------------------
            # Create validation artifact
            # ---------------------------------------------------------

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=(
                    self.data_validation_config
                    .drift_report_file_path
                ),
            )

            logging.info(
                f"Data validation artifact: "
                f"{data_validation_artifact}"
            )

            logging.info(
                "Exited initiate_data_validation method "
                "of DataValidation class"
            )

            return data_validation_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e