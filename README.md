# Customer Churn Prediction System

A machine learning-based Flask web application that predicts whether a telecom customer is likely to churn (leave the service) or stay.

## Features

- Customer churn prediction
- Data cleaning and preprocessing
- Categorical feature encoding with `LabelEncoder`
- Numerical feature scaling with `StandardScaler`
- Comparison of Logistic Regression, Random Forest, and Decision Tree
- Automatic selection of the best model using F1-score
- Model performance metrics
- Churn probability prediction
- Flask REST API
- Saved model artifacts using Pickle
- Health-check endpoint

## Technologies Used

- Python
- Flask
- Flask-CORS
- Pandas
- NumPy
- Scikit-learn
- KaggleHub
- Pickle

## Dataset

This project uses the Telco Customer Churn dataset:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains customer information such as tenure, contract type, internet service, monthly charges, total charges, and churn status.

During preprocessing:

- `customerID` is removed.
- `TotalCharges` is converted to numeric.
- Missing values are handled.
- Duplicate records are removed.
- Categorical columns are encoded.
- Numerical columns are standardized.
- `Churn` is converted to `1` (Yes) and `0` (No).

## Machine Learning Models

The application trains and compares:

1. Logistic Regression
2. Random Forest
3. Decision Tree

The model with the highest F1-score is selected as the best model.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── server.py
├── model_artifacts.pkl
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── templates/
│   └── index.html
└── README.md
```

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate it

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install flask flask-cors numpy pandas scikit-learn kagglehub
```

## Run the Application

Start the Flask server:

```bash
python server.py
```

Then open:

```text
http://localhost:5000
```

If `model_artifacts.pkl` does not exist, the application automatically loads the dataset, trains the models, evaluates them, and saves the best model.

## API Endpoints

### Home

```http
GET /
```

Serves the main application page.

### Predict Churn

```http
POST /predict
```

Accepts customer information as JSON and returns the predicted churn status and probability.

Example response:

```json
{
  "prediction": "No Churn",
  "probability": "25.4%",
  "probability_raw": 25.4,
  "success": true
}
```

### Model Metrics

```http
GET /api/metrics
```

Returns model performance metrics, the selected best model, and top features.

### Train Model

```http
POST /api/train
```

Retrains the models and saves the best-performing model.

### Health Check

```http
GET /api/health
```

Returns the application health status.

Example:

```json
{
  "status": "healthy",
  "success": true
}
```

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

F1-score is used to select the best-performing model.

## Model Artifacts

`model_artifacts.pkl` stores:

- Trained model
- Scaler
- Label encoders
- Feature columns
- Best model name
- Model metrics
- Top features

Keeping this file in the project allows the application to load the trained model without retraining.

## Important Note

The application uses KaggleHub to load the dataset when training is required, so internet access may be required.

Do not commit passwords, API keys, tokens, or other private credentials to GitHub.

## Future Improvements

- Add an interactive dashboard
- Add model-performance charts
- Add customer risk categories
- Improve feature engineering
- Add hyperparameter tuning
- Add automated tests
- Add Docker support
- Deploy the application to a cloud platform


