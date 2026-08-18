<div align="center">

# 🇺🇸 US Visa Approval Predictor

**AI-powered visa approval prediction system — built end-to-end with a production MLOps pipeline**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-usvisa--demo.onrender.com-4f9eff?style=for-the-badge)](https://usvisa-demo.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47a248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![AWS](https://img.shields.io/badge/AWS-ECR_Ready-ff9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com)

</div>

---

## 🔗 Live Demo

> **Try it now →** [https://usvisa-demo.onrender.com](https://usvisa-demo.onrender.com)

Fill in the applicant's details and get an instant AI-powered visa approval prediction.

> ⚠️ **First load may take ~30 seconds** — the free-tier server wakes up on demand.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [ML Pipeline](#-ml-pipeline)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Getting Started](#-getting-started)
- [Docker](#-docker)
- [Deployment Branches](#-deployment-branches)
- [API Reference](#-api-reference)
- [Environment Variables](#-environment-variables)

---

## 🧠 Overview

This project predicts whether a US visa application will be **approved or denied** based on applicant and employer details. It is built as a **full-stack MLOps system** — not just a trained model, but a complete pipeline that:

- Ingests data from **MongoDB Atlas**
- Validates data for **schema drift**
- Transforms features with **SMOTE oversampling** to handle class imbalance
- Trains **4 classifiers** and selects the best by F1 score
- Evaluates against the production model before deploying
- Serves predictions via a **FastAPI REST API**
- Renders a **modern glassmorphism UI**
- Supports **background model retraining** via a dedicated endpoint

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER / RECRUITER                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTPS
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              FastAPI Application  (app.py)                      │
│                                                                  │
│   GET  /          → Glassmorphism Web UI (usvisa.html)          │
│   POST /predict   → JSON Prediction API                         │
│   POST /retrain   → Trigger background training pipeline        │
│   GET  /retrain/status → Poll training progress                 │
│   GET  /health    → Health check                                │
└──────────┬──────────────────────────────┬────────────────────────┘
           │ predict()                    │ retrain()
           ▼                              ▼
┌─────────────────────┐       ┌──────────────────────────────────┐
│  Prediction Pipeline │       │      Training Pipeline           │
│                     │       │                                  │
│  USvisaData         │       │  DataIngestion (MongoDB/CSV)     │
│  → DataFrame        │       │  → DataValidation (drift check)  │
│  → Preprocessor     │       │  → DataTransformation (SMOTE)    │
│  → Model.pkl        │       │  → ModelTrainer (4 classifiers)  │
│  → Prediction       │       │  → ModelEvaluation (vs prod)     │
└─────────────────────┘       │  → ModelPusher (to final_model/) │
                              └──────────────────────────────────┘
                                           │
                              ┌────────────▼──────────┐
                              │    MongoDB Atlas       │
                              │  (raw dataset store)  │
                              └───────────────────────┘
```

---

## 🔬 ML Pipeline

The training pipeline consists of **6 sequential stages**:

| Stage | Component | What it does |
|-------|-----------|--------------|
| 1️⃣ | **Data Ingestion** | Fetches the EasyVisa dataset from MongoDB Atlas; falls back to local CSV if offline |
| 2️⃣ | **Data Validation** | Checks schema, column types, and runs an Evidently data drift report |
| 3️⃣ | **Data Transformation** | Encodes categoricals, scales numerics, applies **SMOTE** to fix class imbalance |
| 4️⃣ | **Model Trainer** | Trains RandomForest, KNN, XGBoost, and CatBoost; selects best by **F1 score** |
| 5️⃣ | **Model Evaluation** | Compares the new model against the production model — only promotes if better |
| 6️⃣ | **Model Pusher** | Saves accepted model to `final_model/` and optionally pushes to **AWS S3** |

---

## 📊 Model Performance

> Results from the last full training run (`2026-08-17`):

| Model | Test Accuracy | Test F1 | Precision | Recall |
|-------|:---:|:---:|:---:|:---:|
| Random Forest | 70.25% | 0.587 | 0.565 | 0.610 |
| KNN | 66.74% | 0.552 | 0.517 | 0.592 |
| XGBoost | 71.09% | 0.605 | 0.574 | 0.640 |
| **CatBoost ✅ (selected)** | **71.94%** | **0.617** | **0.584** | **0.653** |

**CatBoostClassifier** was selected as the production model with hyperparameters:
- `learning_rate=0.1`, `depth=10`, `iterations=300`, `l2_leaf_reg=3`

> The model is saved to `final_model/model.pkl` with the preprocessing pipeline bundled (`final_model/preprocessor.pkl`) into a single `USvisaModel` object using `dill` serialisation.

---

## 🛠️ Tech Stack

### Machine Learning
- **CatBoost** — production model (gradient boosting)
- **XGBoost, RandomForest, KNN** — challenger models
- **scikit-learn** — preprocessing, metrics, SMOTE
- **imbalanced-learn** — SMOTE oversampling
- **Evidently AI** — data drift detection

### Backend
- **FastAPI** — async REST API
- **Uvicorn** — ASGI server
- **Pydantic** — data validation & schema
- **dill** — serialisation for complex Python objects

### Data
- **MongoDB Atlas** — cloud dataset store
- **pymongo** — MongoDB driver
- **pandas / numpy** — data processing

### Infrastructure
- **Docker** — containerisation
- **Render.com** — free-forever cloud deployment
- **AWS ECR + ECS** — production-grade deployment (branch: `aws-deploy`)
- **GitHub Actions** — CI/CD pipeline

### Frontend
- Vanilla HTML/CSS/JavaScript
- Glassmorphism design system
- Async fetch API for real-time predictions

---

## 📁 Project Structure

```
USvisa_approval_systmm/
│
├── app.py                          # FastAPI application (routes, retrain, predict)
├── Dockerfile                      # Production Docker image
├── docker-compose.yml              # Local orchestration
├── render.yaml                     # Render.com deployment config (free-cloud branch)
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Dev + test dependencies
├── Makefile                        # Helper commands
├── setup.py                        # Package config
│
├── us_visa/                        # Core Python package
│   ├── components/                 # Pipeline stage implementations
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrates all 6 stages
│   │   └── prediction_pipeline.py  # USvisaClassifier for inference
│   │
│   ├── entity/
│   │   ├── config_entity.py        # Typed config dataclasses
│   │   ├── artifact_entity.py      # Typed artifact dataclasses
│   │   └── estimator.py            # USvisaModel wrapper
│   │
│   ├── cloud_storage/
│   │   └── aws_storage.py          # S3 upload/download helpers
│   │
│   ├── data_access/
│   │   └── usvisa_data.py          # MongoDB data access layer
│   │
│   ├── constants/                  # App-wide constants & config paths
│   ├── logger/                     # Structured logging
│   ├── exception/                  # Custom exception with traceback
│   └── utils/                      # Shared utilities (load/save object, YAML)
│
├── templates/
│   └── usvisa.html                 # Glassmorphism prediction UI
├── static/                         # CSS, JS, icons
├── config/                         # schema.yaml, model.yaml
├── notebook/                       # EDA & model selection notebooks
├── tests/                          # Unit tests
├── final_model/                    # Production model artifacts
└── .github/workflows/aws.yaml      # GitHub Actions CI/CD
```

---

## ✨ Features

- ✅ **End-to-end MLOps pipeline** — data ingestion → validation → transformation → training → evaluation → deployment
- ✅ **Automatic model selection** — trains 4 classifiers, selects the best by F1 score
- ✅ **Model gating** — new model only promoted to production if it outperforms the current one
- ✅ **Background retraining** — trigger pipeline via API; poll status without blocking the UI
- ✅ **Data drift detection** — Evidently AI reports on each training run
- ✅ **SMOTE** — handles class imbalance without discarding data
- ✅ **Docker-first** — single `Dockerfile` runs locally and in cloud
- ✅ **Health check endpoint** — Render and ECS both use `/health`
- ✅ **Premium UI** — glassmorphism design, animated result card, toast notifications
- ✅ **AWS-ready** — separate `aws-deploy` branch with ECR push workflow

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- MongoDB Atlas URI (or use the local CSV fallback)
- AWS credentials (optional — only needed for S3/ECR features)

### 1. Clone the repo

```bash
git clone https://github.com/PanchalAnubhav/USvisa_approval_systmm.git
cd USvisa_approval_systmm
git checkout free-cloud   # demo branch
```

### 2. Install dependencies

```bash
pip install -r requirements-dev.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env and fill in MONGODB_URL, AWS credentials, etc.
```

### 4. Run locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Open [http://localhost:8080](http://localhost:8080)

---

## 🐳 Docker

### Build and run

```bash
docker build -t usvisa-app -f Dockerfile .
docker compose up -d
```

### Verify

```bash
curl http://localhost:8080/health
# → {"status": "ok"}
```

---

## 🌿 Deployment Branches

| Branch | Purpose | URL |
|--------|---------|-----|
| `main` | Source of truth / development | — |
| `free-cloud` | **Live demo on Render.com (no credit card)** | [usvisa-demo.onrender.com](https://usvisa-demo.onrender.com) |
| `aws-deploy` | AWS ECR + ECS via GitHub Actions | Triggered by CI push |

> The `free-cloud` and `aws-deploy` branches are **intentionally kept isolated** — changes to one never affect the other.

---

## 📡 API Reference

### `GET /health`
Returns service health status.
```json
{"status": "ok"}
```

### `POST /predict`
Predict visa approval outcome.

**Request body:**
```json
{
  "continent": "Asia",
  "education_of_employee": "Master's",
  "has_job_experience": "Y",
  "requires_job_training": "N",
  "no_of_employees": 5000,
  "region_of_employment": "South",
  "prevailing_wage": 80000,
  "unit_of_wage": "Year",
  "full_time_position": "Y",
  "company_age": 20
}
```

**Response:**
```json
{
  "status": true,
  "prediction": "Visa-Approved",
  "prediction_value": 1
}
```

### `POST /retrain`
Triggers the full 6-stage training pipeline in the background. Returns `202 Accepted` immediately.

### `GET /retrain/status`
Poll training progress.
```json
{
  "status": "running",
  "started_at": "2026-08-17T18:00:00Z",
  "finished_at": null,
  "message": "Pipeline started. This may take 2–5 minutes."
}
```

---

## 🔐 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MONGODB_URL` | ✅ | MongoDB Atlas connection string |
| `DATABASE_NAME` | ✅ | MongoDB database name |
| `COLLECTION_NAME` | ✅ | MongoDB collection name |
| `AWS_ACCESS_KEY_ID` | ⚠️ Optional | For S3 model push & ECR |
| `AWS_SECRET_ACCESS_KEY` | ⚠️ Optional | For S3 model push & ECR |
| `AWS_REGION` | ⚠️ Optional | AWS region (e.g. `us-east-1`) |
| `APP_HOST` | ✅ | Server host (default `0.0.0.0`) |
| `APP_PORT` | ✅ | Server port (default `8080`) |

---

## 👤 Author

**Anubhav Panchal**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0a66c2?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/PanchalAnubhav)

---

<div align="center">

Made with ❤️ | [Live Demo →](https://usvisa-demo.onrender.com)

</div>
