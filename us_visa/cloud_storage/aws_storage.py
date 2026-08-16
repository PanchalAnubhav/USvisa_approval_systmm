import os
import sys
from io import StringIO
from typing import List, Union

import dill
from botocore.exceptions import ClientError
from mypy_boto3_s3.service_resource import Bucket

from us_visa.configuration.aws_connection import S3Client
from us_visa.exception import USvisaException
from us_visa.logger import logging


class SimpleStorageService:
    """
    Wrapper around S3Client that provides high-level operations for
    uploading, downloading, listing, and loading model artifacts from S3.
    """

    def __init__(self):
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        """
        Check if a given key path exists in the specified S3 bucket.
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [
                file_object for file_object in bucket.objects.filter(Prefix=s3_key)
            ]
            if len(file_objects) > 0:
                return True
            return False
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_object(object_name: str, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str]:
        """
        Read the content of an S3 object.
        """
        logging.info("Entered the read_object method of SimpleStorageService class")

        try:
            func = (
                lambda: object_name.get()["Body"].read().decode()
                if decode
                else object_name.get()["Body"].read()
            )
            conv_func = lambda: StringIO(func()) if make_readable else func()
            return conv_func()
        except Exception as e:
            raise USvisaException(e, sys)

    def get_bucket(self, bucket_name: str) -> Bucket:
        """
        Get the Bucket resource object.
        """
        logging.info("Entered the get_bucket method of SimpleStorageService class")

        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logging.info("Exited the get_bucket method of SimpleStorageService class")
            return bucket
        except Exception as e:
            raise USvisaException(e, sys)

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List, object]:
        """
        Get file objects with a given prefix from the bucket.
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [
                file_object for file_object in bucket.objects.filter(Prefix=filename)
            ]
            return file_objects
        except Exception as e:
            raise USvisaException(e, sys)

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        """
        Load a dill-serialized model from S3.
        """
        logging.info("Entered the load_model method of SimpleStorageService class")

        try:
            func = (
                lambda: model_name
                if model_dir is None
                else model_dir + "/" + model_name
            )

            model_file = func()
            file_object = self.get_file_object(model_file, bucket_name)

            if len(file_object) > 0:
                file_obj = file_object[0]
                file_content = self.read_object(file_obj, decode=False)
                model = dill.loads(file_content)
                logging.info("Exited the load_model method of SimpleStorageService class")
                return model
            else:
                logging.info("Model not found in S3 bucket")
                return None

        except Exception as e:
            raise USvisaException(e, sys)

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """
        Create a folder (prefix) in the S3 bucket.
        """
        logging.info("Entered the create_folder method of SimpleStorageService class")

        try:
            self.s3_resource.Object(bucket_name, folder_name).load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                folder_obj = folder_name + "/"
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj)
            else:
                raise USvisaException(e, sys)
        logging.info("Exited the create_folder method of SimpleStorageService class")

    def upload_file(
        self,
        from_filename: str,
        to_filename: str,
        bucket_name: str,
        remove: bool = True,
    ) -> None:
        """
        Upload a local file to S3.
        """
        logging.info("Entered the upload_file method of SimpleStorageService class")

        try:
            logging.info(
                f"Uploading {from_filename} to {to_filename} in bucket {bucket_name}"
            )

            self.s3_resource.meta.client.upload_file(
                from_filename, bucket_name, to_filename
            )

            logging.info(
                f"Uploaded {from_filename} to {to_filename} in bucket {bucket_name}"
            )

            if remove:
                os.remove(from_filename)
                logging.info(f"Removed the local file {from_filename} after upload.")

        except Exception as e:
            raise USvisaException(e, sys)

    def download_file(self, bucket_name: str, output_file_path: str, key: str) -> None:
        """
        Download a file from S3 to a local path.
        """
        try:
            self.s3_client.download_file(bucket_name, key, output_file_path)
        except Exception as e:
            raise USvisaException(e, sys)
