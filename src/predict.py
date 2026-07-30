"""
predict.py - Interactive & Modular Customer Churn Prediction Engine

This script allows users to make real-time churn predictions for individual
or batch customer profiles. It outputs predicted probabilities, risk levels,
confidence metrics, dynamic ASCII probability visualizers, and targeted business actions.
"""

import os
import joblib
import pandas as pd
import numpy as np
from utils import categorize_risk

def load_champion_model(model_path: str = None):
    """
    Load the serialized model artifact containing the trained pipeline and feature schema.
    """
    if model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "saved_model", "churn_prediction_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please run train_model.py first.")

    artifact = joblib.load(model_path)
    return artifact['model_pipeline'], artifact['feature_names'], artifact.get('model_name', 'Trained Model')

def format_input_data(customer_dict: dict, feature_names: list) -> pd.DataFrame:
    """
    Transform raw user input dictionary into the exact engineered feature format expected by the model.
    """
    # Baseline default values for full schema
    raw_record = {
        'SeniorCitizen': 1 if customer_dict.get('Age', 30) >= 65 else 0,
        'Age': customer_dict.get('Age', 40),
        'Tenure': customer_dict.get('Tenure', 12),
        'MonthlyCharges': customer_dict.get('MonthlyCharges', 70.0),
        'TotalCharges': customer_dict.get('Tenure', 12) * customer_dict.get('MonthlyCharges', 70.0),
        'Gender': customer_dict.get('Gender', 'Male'),
        'InternetService': customer_dict.get('InternetService', 'Fiber optic'),
        'Contract': customer_dict.get('Contract', 'Month-to-month'),
        'PaymentMethod': customer_dict.get('PaymentMethod', 'Electronic check'),
        'TechSupport': customer_dict.get('TechSupport', 'No'),
        'OnlineSecurity': customer_dict.get('OnlineSecurity', 'No'),
        'StreamingTV': customer_dict.get('StreamingTV', 'No'),
        'StreamingMovies': customer_dict.get('StreamingMovies', 'No'),
        'PhoneService': customer_dict.get('PhoneService', 'Yes'),
        'MultipleLines': customer_dict.get('MultipleLines', 'No'),
        'Dependents': customer_dict.get('Dependents', 'No'),
        'Complaints': customer_dict.get('Complaints', 0),
        'SatisfactionScore': customer_dict.get('SatisfactionScore', 3),
        'SupportTickets': customer_dict.get('SupportTickets', 1),
        'UsageHours': customer_dict.get('UsageHours', 100.0),
        'DataConsumption': customer_dict.get('UsageHours', 100.0) * 1.5
    }

    # Convert to DataFrame and apply One-Hot Encoding
    df_raw = pd.DataFrame([raw_record])
    cat_cols = df_raw.select_dtypes(include=['object', 'str']).columns
    df_encoded = pd.get_dummies(df_raw, columns=cat_cols, drop_first=True)
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    # Reindex columns to match model training feature schema exactly
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)
    return df_aligned

def generate_visual_bar(probability: float, length: int = 25) -> str:
    """Generate a terminal-safe ASCII visual progress bar for probability visualization."""
    filled_length = int(round(length * probability))
    bar = '=' * filled_length + '-' * (length - filled_length)
    return f"[{bar}] {probability * 100:.1f}%"

def predict_customer_churn(customer_dict: dict, model_path: str = None) -> dict:
    """
    Predict churn for a single customer profile.
    Returns structured results including prediction, probability, risk tier, and recommended action.
    """
    pipeline, feature_names, model_name = load_champion_model(model_path)
    X_input = format_input_data(customer_dict, feature_names)

    prediction = pipeline.predict(X_input)[0]
    prob_churn = pipeline.predict_proba(X_input)[0][1]

    risk_info = categorize_risk(prob_churn)
    visual_bar = generate_visual_bar(prob_churn)

    result = {
        'Prediction': 'WILL CHURN' if prediction == 1 else 'WILL NOT CHURN',
        'Model Used': model_name,
        'Churn Probability': prob_churn,
        'Probability Bar': visual_bar,
        'Risk Percentage': risk_info['Risk Percentage'],
        'Risk Level': risk_info['Risk Tier'],
        'Confidence Level': risk_info['Confidence Level'],
        'Recommended Action': risk_info['Recommended Action']
    }

    return result

def print_prediction_report(result: dict):
    """Format and print a professional prediction summary."""
    print("\n" + "=" * 60)
    print("            CUSTOMER CHURN PREDICTION REPORT")
    print("=" * 60)
    print(f" Prediction Target  : {result['Prediction']}")
    print(f" Churn Probability  : {result['Probability Bar']}")
    print(f" Risk Tier          : {result['Risk Level']}")
    print(f" Confidence Level   : {result['Confidence Level']}")
    print("-" * 60)
    print(f" RECOMMENDED BUSINESS RETENTION STRATEGY:")
    print(f" >> {result['Recommended Action']}")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    print("[INFO] Running Sample Interactive Prediction Test...")

    # Sample High-Risk Customer Profile
    high_risk_sample = {
        'Age': 45,
        'Gender': 'Female',
        'Tenure': 3,
        'MonthlyCharges': 95.50,
        'Contract': 'Month-to-month',
        'InternetService': 'Fiber optic',
        'TechSupport': 'No',
        'Complaints': 1,
        'SupportTickets': 4,
        'SatisfactionScore': 1,
        'UsageHours': 45.0
    }

    # Sample Low-Risk Customer Profile
    low_risk_sample = {
        'Age': 32,
        'Gender': 'Male',
        'Tenure': 48,
        'MonthlyCharges': 45.00,
        'Contract': 'Two year',
        'InternetService': 'DSL',
        'TechSupport': 'Yes',
        'Complaints': 0,
        'SupportTickets': 0,
        'SatisfactionScore': 5,
        'UsageHours': 210.0
    }

    print("\n--- SAMPLE 1: HIGH RISK PROFILE TEST ---")
    res_high = predict_customer_churn(high_risk_sample)
    print_prediction_report(res_high)

    print("\n--- SAMPLE 2: LOW RISK PROFILE TEST ---")
    res_low = predict_customer_churn(low_risk_sample)
    print_prediction_report(res_low)
