# from us_visa.logger import logging
# from us_visa.exception import USvisaException
# import sys

# logging.info("Welcome to our custom log")
# try:
#     a = 1 / "10"
# except Exception as e:
#     raise USvisaException(e, sys) from e




from us_visa.pipeline.training_pipeline import TrainPipeline


pipeline = TrainPipeline()
pipeline.run_pipeline()