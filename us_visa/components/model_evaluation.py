import os
import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from sklearn.metrics import f1_score

from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact,
)
from us_visa.exception import USvisaException
from us_visa.constants import TARGET_COLUMN, CURRENT_YEAR
from us_visa.logger import logging
from us_visa.entity.estimator import USvisaModel, TargetValueMapping
from us_visa.utils.main_utils import load_object

# Local path where the production-accepted model is stored
LOCAL_MODEL_PATH = os.path.join("final_model", "model.pkl")


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_best_model(self) -> Optional[USvisaModel]:
        """
        Load the current production model from the local model store.
        Returns None if no production model exists yet (first run).
        """
        try:
            if os.path.exists(LOCAL_MODEL_PATH):
                logging.info(f"Loading existing production model from {LOCAL_MODEL_PATH}")
                return load_object(file_path=LOCAL_MODEL_PATH)
            logging.info("No existing production model found — this is the first run.")
            return None
        except Exception as e:
            raise USvisaException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluates the newly trained model against the current local production model.
        Accepts the new model if its F1 score exceeds the old one by the threshold.
        """
        try:
            logging.info("Entered evaluate_model method of ModelEvaluation class")

            # Read test data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Feature engineering (same as data_transformation)
            if "case_id" in test_df.columns:
                test_df = test_df.drop(columns=["case_id"])
            if "yr_of_estab" in test_df.columns:
                test_df["company_age"] = CURRENT_YEAR - test_df["yr_of_estab"]
                test_df = test_df.drop(columns=["yr_of_estab"])

            x_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # Encode target
            target_mapping = TargetValueMapping()
            y_test = y_test.map(target_mapping._asdict())

            # Use F1 score from trainer artifact (avoids re-running inference)
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            best_model_f1_score = 0.0

            # Try to load existing local production model and score it
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x_test)
                best_model_f1_score = f1_score(y_test, y_hat_best_model)
                logging.info(f"Existing production model F1: {best_model_f1_score:.4f}")

            # Determine if new model is accepted
            difference = trained_model_f1_score - best_model_f1_score
            is_model_accepted = difference >= self.model_eval_config.changed_threshold_score

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_model_accepted,
                difference=difference,
            )

            logging.info(f"Evaluation result: {result}")
            logging.info("Exited evaluate_model method of ModelEvaluation class")

            return result

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Initiates the model evaluation component.
        """
        try:
            logging.info("Entered initiate_model_evaluation method of ModelEvaluation class")

            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                changed_accuracy=evaluate_model_response.difference,
                s3_model_path=LOCAL_MODEL_PATH,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            logging.info("Exited initiate_model_evaluation method of ModelEvaluation class")

            return model_evaluation_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e
