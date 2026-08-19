# Contributing to US Visa Approval System


Thank you for your interest in contributing to the **US Visa Approval System**.


This repository is primarily a personal **Machine Learning / MLOps portfolio project**, but suggestions, bug reports, improvements, and meaningful contributions are welcome.


## Getting Started


1. Fork the repository.
2. Clone your fork:


```bash
git clone https://github.com/PanchalAnubhav/USvisa_approval_systmm.git
cd USvisa_approval_systmm
```
Create a new branch for your changes:
```
git checkout -b feature/your-feature-name
```
Create and activate a virtual environment:
```
python -m venv .venv
```
Windows
```
.venv\Scripts\activate
```
Install the project dependencies:
```
pip install -r requirements.txt
```
Configure your environment variables using .env.example.

Do not commit your actual .env file, database credentials, API keys, passwords, or other secrets.

## Project Structure

The project follows a modular ML pipeline architecture:

- Data Ingestion — Retrieves and prepares data from MongoDB.
- Data Validation — Validates schema, columns, and dataset drift.
- Data Transformation — Performs feature engineering and preprocessing.
- Model Training — Trains and evaluates machine learning models.
- Model Evaluation — Compares model performance.
- Model Pusher — Handles the trained model artifacts.
- Prediction Pipeline — Uses the trained model for predictions.
- Application/API — Provides the user-facing prediction interface.
- Deployment — Supports the live demonstration environment and an AWS-ready deployment branch.

When making changes, try to preserve the separation between these components.

## Development Guidelines

Please:

- Keep changes focused and relevant.
- Follow the existing project structure and coding style.
- Use clear and descriptive names.
- Keep functions and classes reasonably modular.
- Add logging where appropriate.
- Handle exceptions using the project's existing exception-handling approach.
- Update documentation when your changes affect project usage or behavior.
- Test changes locally before submitting a pull request.
- Avoid unnecessary changes to unrelated parts of the project.

For ML-related changes, verify that changes to preprocessing, feature engineering, schemas, or model training remain compatible with the prediction pipeline.

## Data and Security

Never commit:

- .env files
- MongoDB credentials
- API keys
- Passwords
- Access tokens
- Private certificates
- Personal or confidential datasets
- Generated secrets

Use .env.example to document required environment variables without exposing their values.

If you discover a security vulnerability, please follow the instructions in SECURITY.md rather than publicly disclosing sensitive details in an issue.

## Testing

Before submitting changes, make sure the relevant functionality works locally.

Depending on the change, verify:

- Data ingestion
- Data validation
- Data transformation
- Model training
- Model evaluation
- Prediction pipeline
- API/application behavior
- Deployment configuration

For deployment-related changes, also verify that required environment variables are configured correctly and that the application starts successfully.

Commit Guidelines

Use clear and descriptive commit messages.

Examples:

- fix: resolve MongoDB connection configuration
- feat: add prediction validation
- fix: correct data transformation pipeline
- docs: update deployment instructions
- refactor: improve model evaluation component

Avoid vague commit messages such as:

- update
- changes
- fixed stuff
- final
- new code
- Pull Requests

Before opening a pull request:

- Make sure your branch contains only the changes related to the contribution.
- Test your changes locally.
- Check that no secrets or unnecessary generated files are included.
- Update relevant documentation.
- Provide a clear description of the changes.
- Explain any important implementation decisions.
- Include screenshots, logs, or test results when useful.

A pull request should clearly communicate:

- What was changed.
- Why it was changed.
- How it was tested.
- Any known limitations or follow-up work.
- Issues
- Bug Reports

When reporting a bug, provide:

- A clear description of the problem.
- Steps to reproduce it.
- Expected behavior.
- Actual behavior.
- Relevant error messages or logs.
- Python/OS/environment information when relevant.
- Screenshots when they help explain the issue.
- Feature Requests

For feature requests, explain:

- What you would like to add.
- What problem it solves.
- Why it would be useful.
- How it could fit into the existing project.
- Any implementation ideas you may have.

## Branches

The repository uses separate branches for different purposes.

## Main Branch

The main branch contains the primary project code and documentation.

## Free Deployment Branch

The free-deployment branch contains configuration required for the publicly accessible demonstration deployment.

## AWS Deployment Branch

The AWS deployment branch is maintained separately as an AWS-ready version of the project.

Changes that affect the core application should be considered carefully before being applied independently to deployment-specific branches.

## Before You Submit

Please check the following:

 The project runs successfully.
 I tested the functionality affected by my changes.
 I did not commit secrets or credentials.
 I followed the existing project structure.
 I updated documentation where necessary.
 My commit messages are clear.
 My pull request clearly explains the changes.
 I checked that unrelated files were not modified.

## Thank You

Thank you for taking the time to contribute to the US Visa Approval System.

Whether you are reporting a bug, suggesting an improvement, improving documentation, or contributing code, your feedback helps make the project better.
