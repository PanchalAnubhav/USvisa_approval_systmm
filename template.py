import os
from pathlib import Path


project_name = "us_visa"
list_of_files = [
    f"{project_name}/__init__.py",
   
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",

    f"{project_name}/configuration/__init__.py",

    f"{project_name}/constants/__init__.py",

    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",

    f"{project_name}/exception/__init__.py",

    f"{project_name}/logger/__init__.py",

    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",

    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",

    "app.py",
    "requirements.txt",
    "DockerFile",
    ".dockerignore",
    "demo.py",
    "setup.py",
    "config/model.yaml",
    "config/schema.yaml",
]


for filepath in list_of_files:

    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)           # to split the path into directory and file name

    if filedir != "":                                    # to check if directory is not empty
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):  # to check if file exists or not
        with open(filepath, "w") as f:
            pass
    else:
        print(f"{filename} already exists")