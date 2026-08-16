import sys
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_numpy_array_data, load_object, save_object
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from us_visa.entity.estimator import USvisaModel


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    @staticmethod
    def _evaluate_clf(y_true, y_pred) -> Tuple[float, float, float, float]:
        """
        Evaluate a classifier and return (accuracy, f1, precision, recall).
        Matches the notebook's evaluate_clf function (cell 71).
        """
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        return acc, f1, precision, recall

    def get_model_object_and_report(
        self, train: np.ndarray, test: np.ndarray
    ) -> Tuple[object, ClassificationMetricArtifact, float]:
        """
        Method Name :   get_model_object_and_report
        Description :   Trains multiple models (from notebook cell 86 — best hyperparams),
                         evaluates them on the test set, and selects the best one by F1 score.

        Models and best hyperparams from notebook (cell 84):
            - XGBoost: min_child_weight=1, max_depth=9
            - RandomForest: n_estimators=200, max_features='sqrt', max_depth=None
            - KNN: weights='distance', n_neighbors=4, algorithm='auto'
            - CatBoost: learning_rate=0.1, l2_leaf_reg=3, iterations=300, depth=10

        Output      :   Best model object, metric artifact, and accuracy score
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Entered get_model_object_and_report method of ModelTrainer class")

            X_train, y_train, X_test, y_test = (
                train[:, :-1],
                train[:, -1],
                test[:, :-1],
                test[:, -1],
            )

            # Define models with best hyperparameters from notebook
            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=200, max_features="sqrt", max_depth=None
                ),
                "KNeighborsClassifier": KNeighborsClassifier(
                    weights="distance", n_neighbors=4, algorithm="auto"
                ),
                "XGBClassifier": XGBClassifier(
                    min_child_weight=1, max_depth=9, n_jobs=-1
                ),
                "CatBoostClassifier": CatBoostClassifier(
                    learning_rate=0.1, l2_leaf_reg=3, iterations=300, depth=10, verbose=False
                ),
            }

            best_model = None
            best_model_name = None
            best_f1 = -1.0
            best_accuracy = -1.0
            best_metric_artifact = None

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_acc, train_f1, train_precision, train_recall = self._evaluate_clf(y_train, y_train_pred)
                test_acc, test_f1, test_precision, test_recall = self._evaluate_clf(y_test, y_test_pred)

                logging.info(f"{model_name} — Train: acc={train_acc:.4f}, f1={train_f1:.4f}")
                logging.info(f"{model_name} — Test:  acc={test_acc:.4f}, f1={test_f1:.4f}, "
                             f"precision={test_precision:.4f}, recall={test_recall:.4f}")

                # Select best model by test F1 score
                if test_f1 > best_f1:
                    best_f1 = test_f1
                    best_accuracy = test_acc
                    best_model = model
                    best_model_name = model_name
                    best_metric_artifact = ClassificationMetricArtifact(
                        f1_score=test_f1,
                        precision_score=test_precision,
                        recall_score=test_recall,
                    )

            logging.info(f"Best model: {best_model_name} with F1={best_f1:.4f}, Accuracy={best_accuracy:.4f}")
            logging.info("Exited get_model_object_and_report method of ModelTrainer class")

            return best_model, best_metric_artifact, best_accuracy

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Method Name :   initiate_model_trainer
        Description :   Initiates the model training stage:
                         1. Load transformed train/test numpy arrays
                         2. Train and select the best model
                         3. Verify model meets expected accuracy threshold
                         4. Load preprocessor and wrap into USvisaModel
                         5. Save the model artifact

        Output      :   ModelTrainerArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            best_model, metric_artifact, accuracy = self.get_model_object_and_report(
                train=train_arr, test=test_arr
            )

            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            if accuracy < self.model_trainer_config.expected_accuracy:
                logging.info(
                    f"No model found with accuracy >= {self.model_trainer_config.expected_accuracy}. "
                    f"Best accuracy: {accuracy}"
                )
                raise Exception(
                    f"No model found with accuracy >= {self.model_trainer_config.expected_accuracy}. "
                    f"Best accuracy: {accuracy}"
                )

            usvisa_model = USvisaModel(
                preprocessing_object=preprocessing_obj, trained_model_object=best_model
            )

            logging.info("Created USvisaModel object with preprocessor and trained model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=usvisa_model,
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            logging.info("Exited initiate_model_trainer method of ModelTrainer class")

            return model_trainer_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e
