import os
import sys
import shutil

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from us_visa.entity.config_entity import ModelPusherConfig

# Local destination for the accepted production model
LOCAL_MODEL_DIR = "final_model"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.pkl")


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   Copies the accepted trained model to the local final_model directory.
                        (Local-only mode — no S3 upload required)

        Output      :   ModelPusherArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            trained_model_path = self.model_evaluation_artifact.trained_model_path
            os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

            logging.info(f"Copying trained model from {trained_model_path} to {LOCAL_MODEL_PATH}")
            shutil.copy(src=trained_model_path, dst=LOCAL_MODEL_PATH)
            logging.info("Model copied successfully to final_model/model.pkl")

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name="local",
                s3_model_path=LOCAL_MODEL_PATH,
            )

            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e
