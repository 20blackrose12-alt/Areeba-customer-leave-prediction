# server.py
"""
Customer Churn Prediction System
A complete Flask web application for predicting customer churn using machine learning.
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
import kagglehub
from kagglehub import KaggleDatasetAdapter
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model and encoders
model = None
scaler = None
label_encoders = {}
feature_columns = []
best_model_name = ""
model_metrics = {}

def load_and_preprocess_data():
    """
    Load and preprocess the Telco Customer Churn dataset.
    Handles missing values, encoding, and feature engineering.
    """
    print("Loading dataset from Kaggle...")
    
    try:
        # Load dataset directly from Kaggle
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "blastchar/telco-customer-churn",
            "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove customerID as it's not useful for prediction
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop('customerID', axis=1)
        
        # Handle missing values
        print("Handling missing values...")
        
        # Convert TotalCharges to numeric
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Fill missing TotalCharges with median
        df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
        
        # Handle any other missing values
        df_clean = df_clean.dropna()
        
        # Remove duplicate records
        df_clean = df_clean.drop_duplicates()
        print(f"Dataset after cleaning: {df_clean.shape}")
        
        return df_clean
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def prepare_features_and_target(df):
    """
    Prepare features and target for model training.
    Encode categorical variables and scale numerical features.
    """
    print("Preparing features and target...")
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Store feature columns
    feature_columns = X.columns.tolist()
    
    return X, y, label_encoders, scaler, feature_columns

def train_and_evaluate_models(X, y):
    """
    Train multiple models and select the best performing one.
    """
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = 0
    best_name = ""
    
    print("\nTraining and evaluating models...")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        
        # Select best model based on F1 score
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name
    
    # Get feature importance for best model
    feature_importance = None
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        feature_importance = np.abs(best_model.coef_[0])
    
    # Get top 5 features
    if feature_importance is not None:
        feature_names = X.columns
        feature_importance_dict = dict(zip(feature_names, feature_importance))
        top_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    else:
        top_features = []
    
    print(f"\nBest model: {best_name} with F1 Score: {best_score:.4f}")
    
    return best_model, best_name, results, top_features, X_train, X_test, y_train, y_test

def save_model_artifacts(model, scaler, label_encoders, feature_columns, best_model_name, model_metrics, top_features):
    """
    Save model artifacts for prediction.
    """
    artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns,
        'best_model_name': best_model_name,
        'model_metrics': model_metrics,
        'top_features': top_features
    }
    
    with open('model_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    print("Model artifacts saved successfully!")

def load_model_artifacts():
    """
    Load model artifacts from disk.
    """
    try:
        with open('model_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except FileNotFoundError:
        return None

@app.route('/')
def index():
    """
    Serve the main application page.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make churn prediction based on customer data.
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Load model artifacts
        artifacts = load_model_artifacts()
        if not artifacts:
            return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 500
        
        model = artifacts['model']
        scaler = artifacts['scaler']
        label_encoders = artifacts['label_encoders']
        feature_columns = artifacts['feature_columns']
        
        # Create DataFrame with all required columns
        input_df = pd.DataFrame([data])
        
        # Ensure all columns exist
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value
        
        # Select only the required columns
        input_df = input_df[feature_columns]
        
        # Identify categorical and numerical columns
        categorical_cols = list(label_encoders.keys())
        numerical_cols = [col for col in feature_columns if col not in categorical_cols]
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in input_df.columns:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                except ValueError as e:
                    return jsonify({'error': f'Invalid value for {col}: {str(e)}'}), 400
        
        # Scale numerical features
        if numerical_cols:
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Prepare response
        churn_probability = prediction_proba[1] * 100
        prediction_label = 'Churn' if prediction[0] == 1 else 'No Churn'
        
        return jsonify({
            'prediction': prediction_label,
            'probability': f'{churn_probability:.1f}%',
            'probability_raw': churn_probability,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    Get model performance metrics.
    """
    try:
        artifacts = load_model_artifacts()
        if not artifacts:
            return jsonify({'error': 'Model not trained yet.'}), 500
        
        metrics = artifacts.get('model_metrics', {})
        best_model_name = artifacts.get('best_model_name', '')
        top_features = artifacts.get('top_features', [])
        
        return jsonify({
            'metrics': metrics,
            'best_model': best_model_name,
            'top_features': top_features,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Train the model and return results.
    """
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Prepare features and target
        X, y, label_encoders, scaler, feature_columns = prepare_features_and_target(df)
        
        # Train and evaluate models
        best_model, best_name, model_metrics, top_features, X_train, X_test, y_train, y_test = train_and_evaluate_models(X, y)
        
        # Save model artifacts
        save_model_artifacts(
            best_model, scaler, label_encoders, feature_columns,
            best_name, model_metrics, top_features
        )
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully!',
            'best_model': best_name,
            'metrics': model_metrics,
            'top_features': top_features
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({'status': 'healthy', 'success': True})

if __name__ == '__main__':
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION SYSTEM")
    print("=" * 60)
    
    # Check if model already exists, if not, train it
    if not os.path.exists('model_artifacts.pkl'):
        print("No existing model found. Training new model...")
        try:
            # Load and preprocess data
            df = load_and_preprocess_data()
            
            # Prepare features and target
            X, y, label_encoders, scaler, feature_columns = prepare_features_and_target(df)
            
            # Train and evaluate models
            best_model, best_name, model_metrics, top_features, X_train, X_test, y_train, y_test = train_and_evaluate_models(X, y)
            
            # Save model artifacts
            save_model_artifacts(
                best_model, scaler, label_encoders, feature_columns,
                best_name, model_metrics, top_features
            )
            print("\nModel training completed successfully!")
            
        except Exception as e:
            print(f"\nError during model training: {str(e)}")
            print("Please check your internet connection and try again.")
    else:
        print("Existing model found. Loading model artifacts...")
        artifacts = load_model_artifacts()
        if artifacts:
            print(f"Loaded model: {artifacts.get('best_model_name', 'Unknown')}")
    
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)