# U.S. Visa Approval Prediction System 🇺🇸✈️

## 📌 Project Overview

The **U.S. Visa Approval Prediction System** is a machine learning project designed to classify and predict the approval outcomes of U.S. Visa applications based on historical applicant data. By leveraging advanced data preprocessing and foundational classification algorithms, this project uncovers patterns in applicant features that influence visa decisions.

## 🚀 Features

* **Exploratory Data Analysis (EDA):** Deep dive into the dataset to uncover trends, correlations, and outliers using Matplotlib and Seaborn.

* **Robust Data Preprocessing:** Automated handling of missing values and categorical encoding.

* **Class Imbalance Handling:** Corrected skewed data distributions to prevent model bias towards the majority class.

* **Model Training & Optimization:** Evaluated multiple classification algorithms (e.g., Logistic Regression, Decision Trees/Random Forest) to determine the highest predictive accuracy.

* **Performance Metrics:** Evaluated using Accuracy, Precision, Recall, and F1-Score.

## 🛠️ Technology Stack

* **Language:** Python

* **Data Manipulation:** Pandas, NumPy

* **Machine Learning:** Scikit-Learn

* **Data Visualization:** Matplotlib, Seaborn

## 📊 Model Performance

This model was rigorously tested, yielding high performance across all critical metrics:

| **Metric** | **Score** | 
| **Accuracy** | **96.83%** | 
| **F1-Score** | 97.1% | 
| **Precision** | 95.8% | 
| **Recall (Denied Class)** | **98.5%** | 
| **ROC-AUC** | ~96.7% | 

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/PanchalAnubhav/USvisa_approval_systmm.git](https://github.com/PanchalAnubhav/USvisa_approval_systmm.git)
   cd USvisa_approval_systmm
   
Run the Jupyter Notebook:
jupyter notebook

Open the main notebook file to step through the EDA, preprocessing, and model training pipeline.
