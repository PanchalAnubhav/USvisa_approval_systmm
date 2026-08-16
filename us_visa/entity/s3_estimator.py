import sys

from us_visa.cloud_storage.aws_storage import SimpleStorageService
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.entity.estimator import USvisaModel


class USvisaEstimator:
    """
    This class is used to save and retrieve the US visa model from S3 bucket
    and to make predictions on the data.
    """

    def __init__(self, bucket_name: str, model_path: str):
        """
        :param bucket_name: Name of your S3 model bucket
        :param model_path:  Location of your model in S3 bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model: USvisaModel = None

    def is_model_present(self, model_path: str) -> bool:
        """
        Check if the model is present in S3 bucket.
        """
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except USvisaException:
            return False
        except Exception as e:
            raise USvisaException(e, sys)

    def load_model(self) -> USvisaModel:
        """
        Load the model from the model_path in S3.
        """
        return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Save the model to S3 bucket.

        :param from_file: local file path of the model to upload.
        :param remove: whether to remove the local model file after uploading.
        """
        try:
            self.s3.upload_file(
                from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove,
            )
        except Exception as e:
            raise USvisaException(e, sys)
