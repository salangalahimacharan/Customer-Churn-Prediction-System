"""
utils.py - Reusable utilities for Customer Churn Prediction ML System

This module provides essential helper functions for data loading, data cleaning,
feature preprocessing, model evaluation metrics, visual plotting, and automated
customer churn risk categorization with actionable business strategy suggestions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Apply consistent aesthetic style for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from specified CSV filepath with exception handling.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded successfully from '{filepath}'. Shape: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform comprehensive data cleaning:
    - Drop duplicate rows
    - Fill missing values in TotalCharges with Tenure * MonthlyCharges
    - Ensure correct numerical data types
    """
    df_clean = df.copy()
    
    # 1. Handle Duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    if duplicates_removed > 0:
        print(f"[INFO] Removed {duplicates_removed} duplicate row(s).")

    # 2. Fix Data Types & Missing Values for TotalCharges
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        missing_total = df_clean['TotalCharges'].isnull().sum()
        if missing_total > 0:
            # Impute missing TotalCharges based on Tenure * MonthlyCharges
            imputed_values = df_clean['Tenure'] * df_clean['MonthlyCharges']
            df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(imputed_values)
            print(f"[INFO] Imputed {missing_total} missing value(s) in 'TotalCharges'.")
            
    return df_clean

def prepare_features(df: pd.DataFrame, target_col: str = 'Churn'):
    """
    Encode target variable and categorical features.
    Returns:
    - X (DataFrame): Engineered features
    - y (Series): Binary target (1 for Yes, 0 for No)
    - feature_names (list): List of feature names after dummy encoding
    """
    df_prep = df.copy()
    
    # Remove Identifier Column
    if 'CustomerID' in df_prep.columns:
        df_prep = df_prep.drop(columns=['CustomerID'])
        
    # Map Target Variable
    if target_col in df_prep.columns:
        y = df_prep[target_col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
        X_raw = df_prep.drop(columns=[target_col])
    else:
        y = None
        X_raw = df_prep
        
    # One-Hot Encoding for Categorical Features using pd.get_dummies()
    cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns
    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    
    # Standardize column names (boolean to int)
    bool_cols = X_encoded.select_dtypes(include=['bool']).columns
    X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)
    
    return X_encoded, y

def evaluate_model(y_true, y_pred, y_prob=None, model_name: str = "Model"):
    """
    Compute comprehensive classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'ROC-AUC': roc_auc,
        'Confusion Matrix': cm
    }
    
    return metrics

def display_metrics_table(metrics: dict):
    """
    Format and print a professional performance summary table.
    """
    print("\n" + "=" * 50)
    print(f"       MODEL PERFORMANCE EVALUATION ({metrics['Model'].upper()})")
    print("=" * 50)
    print(f" Accuracy    : {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
    print(f" Precision   : {metrics['Precision']:.4f}")
    print(f" Recall      : {metrics['Recall']:.4f}")
    print(f" F1 Score    : {metrics['F1 Score']:.4f}")
    print(f" ROC-AUC     : {metrics['ROC-AUC']:.4f}")
    print("=" * 50 + "\n")

def categorize_risk(churn_prob: float):
    """
    Categorize churn risk into discrete tiers with confidence levels
    and provide contextual business recommendations.
    """
    if churn_prob < 0.25:
        tier = "Low Risk"
        confidence = "High (Stable Customer)"
        action = "Maintain regular touchpoints. No immediate intervention required."
        color = "green"
    elif churn_prob < 0.50:
        tier = "Medium Risk"
        confidence = "Moderate"
        action = "Send proactive feature usage tips, loyalty newsletters, and service check-ins."
        color = "orange"
    elif churn_prob < 0.75:
        tier = "High Risk"
        confidence = "High (Action Required)"
        action = "Offer personalized contract upgrade discount (15-20%), priority tech support, or free add-ons."
        color = "crimson"
    else:
        tier = "Very High Risk"
        confidence = "Very High (Immediate Priority)"
        action = "Assign dedicated customer retention specialist immediately. Schedule urgent feedback call and offer tailor-made retention bundle."
        color = "darkred"
        
    return {
        'Churn Probability': churn_prob,
        'Risk Percentage': f"{churn_prob * 100:.1f}%",
        'Risk Tier': tier,
        'Confidence Level': confidence,
        'Recommended Action': action,
        'Color': color
    }

# Visual Plotting Helper Functions
def save_plot_churn_distribution(df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x='Churn', palette=['#2ecc71', '#e74c3c'], hue='Churn', legend=False)
    plt.title('Customer Churn Class Distribution', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Churn Status', fontsize=12, labelpad=10)
    plt.ylabel('Customer Count', fontsize=12, labelpad=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = f'{100 * height / total:.1f}%'
        ax.annotate(f'{height}\n({percentage})', (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_plot_confusion_matrix(cm: np.ndarray, model_name: str, output_path: str):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Retained (0)', 'Churned (1)'],
                yticklabels=['Retained (0)', 'Churned (1)'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('Actual Label', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_plot_roc_curve(y_true, y_prob, model_name: str, output_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2980b9', lw=2.5, label=f'{model_name} (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, labelpad=10)
    plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_plot_precision_recall_curve(y_true, y_prob, model_name: str, output_path: str):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#8e44ad', lw=2.5, label=f'{model_name}')
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Recall (Sensitivity)', fontsize=12, labelpad=10)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12, labelpad=10)
    plt.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_plot_feature_importance(feature_names: list, importances: np.ndarray, output_path: str, top_n: int = 15):
    plt.figure(figsize=(10, 7))
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(top_n)
    
    sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
    plt.title(f'Top {top_n} Key Driver Features for Churn Prediction', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Absolute Importance / Coefficient Magnitude', fontsize=12, labelpad=10)
    plt.ylabel('Feature Name', fontsize=12, labelpad=10)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_plot_correlation_heatmap(df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(12, 9))
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix Heatmap', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
