# U.S. Visa Approval Prediction System ✈️

## 📌 Project Overview
The **U.S. Visa Approval Prediction System** is a machine learning project designed to classify and predict the approval outcomes of U.S. Visa applications based on historical applicant data. By leveraging careful data preprocessing and a fair comparison across classification algorithms, this project uncovers patterns in applicant features that influence visa decisions.

> **Update:** An earlier version of this project reported 96-97%+ accuracy. That number was the result of a data leakage bug in the resampling pipeline (SMOTEENN was applied to the full dataset *before* the train/test split, letting synthetic points in the test set be built from real points in training). This has been found, fixed, and documented below — the honest, corrected result is **73.33% accuracy**. See [Model Performance](#-model-performance) for full details.

## 🚀 Features
* **Exploratory Data Analysis (EDA):** Deep dive into the dataset to uncover trends, correlations, and outliers using Matplotlib and Seaborn.
* **Robust Data Preprocessing:** Automated handling of missing values and categorical encoding.
* **Class Imbalance Handling:** SMOTE resampling applied correctly — to the training set only, after the train/test split, to avoid data leakage.
* **Model Training & Comparison:** Evaluated nine classification algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, SVC, KNN, XGBoost, CatBoost) at baseline, then tuned the top four with `RandomizedSearchCV`.
* **Performance Metrics:** Evaluated using Accuracy, Precision, Recall, and F1-Score — on an untouched, real-world-distributed test set.

## 🛠️ Technology Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost, CatBoost, imbalanced-learn (SMOTE)
* **Data Visualization:** Matplotlib, Seaborn

## 📊 Model Performance

**Final model: CatBoostClassifier (SMOTE-resampled, tuned via RandomizedSearchCV)**

| **Metric** | **Score** |
|---|---|
| **Accuracy** | **73.33%** |
| **Macro F1-Score** | 0.69 |
| **Precision (Certified)** | 0.61 |
| **Recall (Certified)** | 0.54 |
| **Precision (Denied)** | 0.79 |
| **Recall (Denied)** | 0.83 |

Evaluated against a naive baseline (always predicting "Certified") of ~66.8% accuracy — so this model is doing genuine, meaningful work, not just exploiting class imbalance.

**Known limitation:** recall on the "Certified" class is 0.54 — a resampling variant (SMOTEENN instead of SMOTE) trades ~3 points of accuracy for more balanced per-class recall (0.67-0.68 on both classes). Both variants are in the notebook; SMOTE was chosen as the default for this README's headline metric, but check the notebook if the precision/recall tradeoff matters for your use case.

**Model comparison (top 4, tuned, SMOTE-resampled):**

| Model | Accuracy | Macro F1 |
|---|---|---|
| **CatBoostClassifier** | **73.33%** | **0.69** |
| XGBClassifier | 70.41% | 0.66 |
| Random Forest Classifier | ~70% | ~0.66 |
| KNeighborsClassifier | 65.13% | 0.62 |

## 🐛 What Went Wrong (and How It Was Fixed)
The original version of this project applied SMOTEENN resampling to the entire dataset *before* splitting into train and test sets, and selected KNN as the "best model" without re-validating that choice after tuning. This produced:
- An inflated, leakage-driven accuracy of 96-97%+
- A test set with an artificially balanced class distribution instead of the real ~66.7% / 33.2% split
- A model (KNN) that, once evaluated fairly, turned out to be the *weakest* of the models tested, not the strongest

The fix: the train/test split now happens first, on the original unresampled data (stratified, so the test set keeps the real class distribution). Resampling is applied only to the training set afterward. All four candidate models were then re-tuned and re-compared on this corrected pipeline — CatBoost came out ahead.

This is documented here deliberately, not hidden, since it's a common and instructive mistake worth showing rather than erasing.

## 💻 How to Run Locally
1. **Clone the repository:**
```bash
   git clone https://github.com/PanchalAnubhav/USvisa_approval_systmm.git
   cd USvisa_approval_systmm
```

2. **Set up a virtual environment and install dependencies:**
```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost catboost jupyter nbconvert
```

3. **Run the Jupyter Notebook:**
```bash
   jupyter notebook
```
   Open the main notebook file to step through the EDA, preprocessing, corrected train/test split, resampling, model training, and evaluation.

## ⚠️ Usage Limitations
This model is intended for **educational and research purposes only** — not for actual immigration decision-making. Visa outcomes depend on complex legal, policy, and human factors well beyond the scope of this dataset, and the model has not undergone formal fairness auditing across demographic subgroups.
