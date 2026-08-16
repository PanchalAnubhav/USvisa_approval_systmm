import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from us_visa.entity.estimator import TargetValueMapping


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        :param data_validation_artifact: Output reference of data validation artifact stage
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns the data transformation pipeline
                         matching the notebook's ColumnTransformer exactly.

        Output      :   ColumnTransformer pipeline object
        On Failure  :   Write an exception log and then raise an exception

        Pipeline from notebook (Cell 63):
            - OneHotEncoder for: continent, unit_of_wage, region_of_employment
            - OrdinalEncoder for: has_job_experience, requires_job_training, full_time_position, education_of_employee
            - PowerTransformer(yeo-johnson) for: no_of_employees, company_age
            - StandardScaler for: no_of_employees, prevailing_wage, company_age
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info("Got numerical cols from schema config")

            oh_columns = self._schema_config["oh_columns"]
            or_columns = self._schema_config["or_columns"]
            transform_columns = self._schema_config["transform_columns"]
            num_features = self._schema_config["numerical_columns"]

            logging.info("Initialized OneHotEncoder, OrdinalEncoder, StandardScaler, PowerTransformer")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ordinal_encoder = OrdinalEncoder()

            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features),
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )

            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component of the pipeline.
                         1. Read train/test CSVs
                         2. Feature engineer: drop case_id, compute company_age = CURRENT_YEAR - yr_of_estab, drop yr_of_estab
                         3. Encode target (Certified=0, Denied=1)
                         4. Apply ColumnTransformer preprocessing
                         5. Apply SMOTE (only on train set)
                         6. Save transformed arrays and preprocessor object

        Output      :   DataTransformationArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Entered initiate_data_transformation method of DataTransformation class")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            train_df = DataTransformation.read_data(
                file_path=self.data_ingestion_artifact.trained_file_path
            )
            test_df = DataTransformation.read_data(
                file_path=self.data_ingestion_artifact.test_file_path
            )

            # --- Feature Engineering (from notebook cells 16, 23, 25) ---
            # Drop case_id
            drop_cols = self._schema_config.get("drop_columns", [])
            if drop_cols:
                train_df = drop_columns(df=train_df, cols=drop_cols)
                test_df = drop_columns(df=test_df, cols=drop_cols)

            # Compute company_age = CURRENT_YEAR - yr_of_estab
            if "yr_of_estab" in train_df.columns:
                train_df["company_age"] = CURRENT_YEAR - train_df["yr_of_estab"]
                test_df["company_age"] = CURRENT_YEAR - test_df["yr_of_estab"]
                train_df.drop("yr_of_estab", inplace=True)
                test_df.drop("yr_of_estab", inplace=True)

            logging.info("Feature engineering complete: dropped case_id, created company_age, dropped yr_of_estab")

            # --- Target encoding ---
            target_mapping = TargetValueMapping()
            train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].map(target_mapping._asdict())
            test_df[TARGET_COLUMN] = test_df[TARGET_COLUMN].map(target_mapping._asdict())

            logging.info("Mapped target column: Certified->0, Denied->1")

            # Split X and y
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("Split input and target features for train and test datasets")

            # --- Preprocessing ---
            preprocessor = self.get_data_transformer_object()

            logging.info("Got the preprocessor object")

            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            logging.info("Used the preprocessor object to transform the datasets")

            # --- SMOTE: applied ONLY to training data (notebook cell 69) ---
            smt = SMOTE(random_state=42, sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )
            logging.info("Applied SMOTE on training dataset")

            input_feature_test_final = transformed_input_test_feature
            target_feature_test_final = target_feature_test_df

            logging.info("Created train and test arrays")

            # Combine X and y into single arrays for saving
            train_arr = np.c_[
                input_feature_train_final, np.array(target_feature_train_final)
            ]
            test_arr = np.c_[
                input_feature_test_final, np.array(target_feature_test_final)
            ]

            # --- Save artifacts ---
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor_object,
            )

            # Save the preprocessor object separately
            save_object(
                file_path="final_model/preprocessor.pkl",
                obj=preprocessor_object,
            )

            logging.info("Saved the preprocessor object")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            logging.info("Exited initiate_data_transformation method of DataTransformation class")

            return data_transformation_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e
