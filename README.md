# US Visa Approval Prediction System

An end-to-end MLOps production-ready machine learning project that predicts whether a US visa application will be **Certified** or **Denied** based on applicant and employer features.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────────┐    ┌──────────────┐
│  MongoDB     │───▶│ Data         │───▶│ Data              │───▶│ Data         │
│  (Raw Data)  │    │ Ingestion    │    │ Validation        │    │ Transform    │
└─────────────┘    └──────────────┘    └───────────────────┘    └──────┬───────┘
                                                                       │
┌─────────────┐    ┌──────────────┐    ┌───────────────────┐          │
│  AWS S3     │◀───│ Model        │◀───│ Model             │◀─────────┘
│  (Registry) │    │ Pusher       │    │ Trainer            │
└─────────────┘    └──────────────┘    └───────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  FastAPI Prediction Service     │
│  /predict  /train  /health      │
└─────────────────────────────────┘
```

## ML Pipeline

| Stage | Description |
|-------|-------------|
| **Data Ingestion** | Exports data from MongoDB, splits into train/test |
| **Data Validation** | Schema validation + KS-test drift detection |
| **Data Transformation** | Feature engineering (company_age), encoding (OHE/Ordinal), scaling (StandardScaler/PowerTransformer), SMOTE resampling |
| **Model Training** | Trains XGBoost, RandomForest, KNN, CatBoost with tuned hyperparams; selects best by F1 |
| **Model Evaluation** | Compares new model against S3 production model |
| **Model Pusher** | Pushes accepted model to AWS S3 registry |

### Best Model (from notebook)
- **CatBoostClassifier** — Accuracy: 73.3%, F1: 0.575
- Hyperparameters: `learning_rate=0.1, l2_leaf_reg=3, iterations=300, depth=10`

## Quick Start

### Prerequisites
- Python 3.10+
- MongoDB Atlas account
- AWS account (for S3 model registry)

### Setup

```bash
# Clone and enter project
git clone <your-repo-url>
cd USvisa_approval_systmm

# Create virtual environment
conda create -n visa python=3.10 -y
conda activate visa

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your MongoDB URL and AWS credentials
```

### Environment Variables

```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@..."
export AWS_ACCESS_KEY_ID=<your_key>
export AWS_SECRET_ACCESS_KEY=<your_secret>
```

### Run

```bash
# Start the API server
uvicorn app:app --host 0.0.0.0 --port 8080 --reload

# Or use Makefile
make run

# Trigger training pipeline
curl http://localhost:8080/train

# Health check
curl http://localhost:8080/health
```

### Docker

```bash
# Build
docker build -t usvisa-app -f DockerFile .

# Run
docker-compose up -d
```

## Project Structure

```
USvisa_approval_systmm/
├── app.py                      # FastAPI application
├── config/
│   ├── model.yaml              # Model hyperparameter configs
│   └── schema.yaml             # Data schema & feature groups
├── us_visa/
│   ├── cloud_storage/          # AWS S3 operations
│   ├── components/             # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── configuration/          # MongoDB & AWS connections
│   ├── constants/              # Project-wide constants
│   ├── data_access/            # MongoDB data access layer
│   ├── entity/                 # Data classes (configs, artifacts, estimators)
│   ├── exception/              # Custom exception handling
│   ├── logger/                 # Logging configuration
│   ├── pipeline/               # Training & prediction pipelines
│   └── utils/                  # Utility functions
├── tests/                      # Unit & integration tests
├── notebook/                   # EDA & model training notebooks
├── DockerFile                  # Container configuration
├── docker-compose.yml          # Local orchestration
├── Makefile                    # Development commands
├── requirements.txt            # Python dependencies
└── .github/workflows/aws.yaml # CI/CD pipeline
```

## Workflow

1. `constants` → 2. `entity` → 3. `components` → 4. `pipeline` → 5. `app.py`

## CI/CD

GitHub Actions workflow (`.github/workflows/aws.yaml`):
1. **CI**: Install deps → Run tests → Build Docker image → Push to ECR
2. **CD**: Pull image on EC2 → Stop old container → Start new container

## License

MIT License
