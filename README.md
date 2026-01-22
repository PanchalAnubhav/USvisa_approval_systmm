Markdown

# 🇺🇸 US Visa Approval Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-red)
![ML](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Docker](https://img.shields.io/badge/Container-Docker-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
The **US Visa Approval Prediction System** is an end-to-end Machine Learning solution designed to predict whether a visa application will be **Certified** or **Denied** based on applicant and employer data.

The project handles the full lifecycle of an ML application: from data ingestion and sophisticated preprocessing to model training and deployment. It addresses key challenges like **severe class imbalance** and **categorical complexity** to achieve a **96.83% accuracy**.

## 🚀 Key Features
* **End-to-End Pipeline:** Modular code structure covering Data Ingestion → Data Transformation → Model Training → Prediction.
* **Robust Preprocessing:** * **Imbalance Handling:** Utilized **SMOTEENN** (combining oversampling and undersampling) to manage skewed data.
    * **Feature Engineering:** Applied Power Transformer, OneHot Encoding, and Ordinal Encoding.
* **Model Selection:** Extensive experimentation with Random Forest, XGBoost, CatBoost, and KNN. 
    * 🏆 **Winner:** **K-Nearest Neighbors (KNN)** outperformed ensemble methods after hyperparameter tuning.
* **Deployment Ready:** Containerized using **Docker** and served via a **Flask API**.
* **Database Integration:** MongoDB used for storing training metadata and logs.

## 📊 Model Performance
This model was rigorously tested, yielding high performance across all critical metrics:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **96.83%** |
| **F1-Score** | 97.1% |
| **Precision** | 95.8% |
| **Recall (Denied Class)** | **98.5%** |
| **ROC-AUC** | ~96.7% |

*The high recall on the "Denied" class ensures that the model is exceptionally good at flagging potential rejections, a critical requirement for this domain.*

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Imbalanced-learn
* **Web Framework:** Flask
* **Containerization:** Docker
* **Database:** MongoDB
* **CI/CD:** GitHub Actions (configured in workflows)

## 📂 Project Structure
* **config/**: Configuration files
* **notebook/**: Jupyter notebooks for EDA and experiments
* **us_visa/**: Main source code package
  * `components/`: Data Ingestion, Transformation, Training modules
  * `pipeline/`: Training and Prediction pipelines
  * `entity/`: Dataclasses for artifacts and config
  * `logger.py`: Logging configuration
* **app.py**: Flask application entry point
* **Dockerfile**: Docker configuration
* **requirements.txt**: Project dependencies

# 🏃‍♂️ Getting Started

## Prerequisites
Python 3.8+

MongoDB Atlas Account (or local MongoDB)

Docker (optional, for containerization)

## Installation
Clone the repository
git clone [https://github.com/PanchalAnubhav/USvisa_approval_systmm.git](https://github.com/PanchalAnubhav/USvisa_approval_systmm.git)
cd USvisa_approval_systmm

## Create a Virtual Environment

conda create -n visa python=3.8 -y
conda activate visa

## Install Dependencies

pip install -r requirements.txt
Set Environment Variables Create a .env file or export your MongoDB URL:
export MONGODB_URL="your_mongodb_connection_string"


## 🐳 Running with Docker
Build the Image
docker build -t us-visa-app .
Run the Container
docker run -p 5000:5000 us-visa-app

📜 License
Distributed under the MIT License. See LICENSE for more information.
