"""
train_model.py - Model Training, Benchmarking, Tuning, and Artifact Export

This script executes the complete machine learning pipeline:
1. Loads and cleans raw data
2. Performs stratified 80/20 train-test split
3. Constructs leak-free Pipelines (StandardScaler + Estimator)
4. Evaluates and benchmarks 5 ML models (Logistic Regression, Decision Tree, Random Forest, KNN, SVM)
5. Performs Hyperparameter Tuning via GridSearchCV and 5-Fold Cross Validation
6. Generates high-resolution diagnostic plots saved to images/
7. Automatically selects and saves the champion model to saved_model/churn_prediction_model.pkl
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from utils import (
    load_data, clean_data, prepare_features, evaluate_model,
    display_metrics_table, save_plot_churn_distribution,
    save_plot_confusion_matrix, save_plot_roc_curve,
    save_plot_precision_recall_curve, save_plot_feature_importance,
    save_plot_correlation_heatmap
)

def run_training_pipeline():
    # Setup Directory Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "customer_churn.csv")
    images_dir = os.path.join(base_dir, "images")
    saved_model_dir = os.path.join(base_dir, "saved_model")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("      CUSTOMER CHURN PREDICTION - ENTERPRISE ML PIPELINE")
    print("=" * 60 + "\n")

    # Step 2 & 3: Load and Clean Data
    df_raw = load_data(dataset_path)
    df = clean_data(df_raw)

    # Save EDA Plots
    print("\n[STEP 4] Generating EDA Visualizations...")
    save_plot_churn_distribution(df, os.path.join(images_dir, "churn_distribution.png"))
    save_plot_correlation_heatmap(df, os.path.join(images_dir, "correlation_heatmap.png"))

    # Step 5: Feature Engineering
    print("\n[STEP 5] Performing Feature Engineering & One-Hot Encoding...")
    X, y = prepare_features(df, target_col='Churn')
    feature_names = X.columns.tolist()
    print(f"[INFO] Processed Features: {len(feature_names)} columns.")

    # Step 6: Train / Test Split
    print("\n[STEP 6] Splitting Dataset (80% Train, 20% Test, Stratified, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Step 7: Build Primary Baseline Pipeline (Logistic Regression)
    print("\n[STEP 7 & 8] Building and Training Primary Baseline (Logistic Regression Pipeline)...")
    primary_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    )
    primary_pipeline.fit(X_train, y_train)

    y_pred_primary = primary_pipeline.predict(X_test)
    y_prob_primary = primary_pipeline.predict_proba(X_test)[:, 1]

    # Step 9: Primary Model Evaluation
    primary_metrics = evaluate_model(y_test, y_pred_primary, y_prob_primary, model_name="Primary Logistic Regression")
    display_metrics_table(primary_metrics)

    # Step 11: Compare Multiple Algorithms & Cross Validation
    print("\n[STEP 11] Benchmarking Multiple Algorithms & Performing Cross Validation...")

    algorithms = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Decision Tree': Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier(max_depth=6, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'K-Nearest Neighbors': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=7))
        ]),
        'Support Vector Machine': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(probability=True, random_state=42))
        ])
    }

    comparison_results = []
    trained_pipelines = {}

    for name, pipeline in algorithms.items():
        start_time = time.time()
        
        # Fit model
        pipeline.fit(X_train, y_train)
        exec_time = time.time() - start_time
        
        # Evaluate
        train_acc = pipeline.score(X_train, y_train)
        test_pred = pipeline.predict(X_test)
        test_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        
        m = evaluate_model(y_test, test_pred, test_prob, model_name=name)
        
        # 5-Fold Cross-Validation on F1 Score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
        
        comparison_results.append({
            'Algorithm': name,
            'Train Accuracy': round(train_acc, 4),
            'Test Accuracy': round(m['Accuracy'], 4),
            'Precision': round(m['Precision'], 4),
            'Recall': round(m['Recall'], 4),
            'F1 Score': round(m['F1 Score'], 4),
            'ROC-AUC': round(m['ROC-AUC'], 4),
            '5-Fold CV F1': round(cv_scores.mean(), 4),
            'Execution Time (s)': round(exec_time, 3)
        })
        trained_pipelines[name] = pipeline

    comparison_df = pd.DataFrame(comparison_results)
    print("\n" + "=" * 80)
    print("                    ALGORITHM COMPARISON BENCHMARK TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80 + "\n")

    # Select Best Model based on ROC-AUC & F1 Score
    best_algo_row = comparison_df.sort_values(by=['ROC-AUC', 'F1 Score'], ascending=False).iloc[0]
    best_model_name = best_algo_row['Algorithm']
    print(f"[WINNER] Best Performing Algorithm Selected: >>> {best_model_name} <<< (ROC-AUC: {best_algo_row['ROC-AUC']:.4f}, F1: {best_algo_row['F1 Score']:.4f})")

    # Hyperparameter Tuning for Winner via GridSearchCV
    print(f"\n[BONUS FEATURE] Running Hyperparameter Tuning (GridSearchCV) for {best_model_name}...")
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [8, 12, None],
            'model__min_samples_split': [2, 5]
        }
    elif best_model_name == 'Logistic Regression':
        param_grid = {
            'model__C': [0.1, 1.0, 10.0],
            'model__solver': ['liblinear', 'lbfgs']
        }
    elif best_model_name == 'Support Vector Machine':
        param_grid = {
            'model__C': [0.5, 1.0, 5.0],
            'model__kernel': ['rbf', 'linear']
        }
    else:
        param_grid = {}

    best_pipeline = trained_pipelines[best_model_name]
    if param_grid:
        grid_search = GridSearchCV(best_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        print(f"[INFO] Best Parameters found: {grid_search.best_params_}")

    # Evaluate Final Tuned Model
    final_pred = best_pipeline.predict(X_test)
    final_prob = best_pipeline.predict_proba(X_test)[:, 1]
    final_metrics = evaluate_model(y_test, final_pred, final_prob, model_name=best_model_name)

    print("\n" + "=" * 50)
    print(f"       FINAL TUNED CHAMPION MODEL REPORT ({best_model_name})")
    print("=" * 50)
    print(classification_report(y_test, final_pred, target_names=['Retained (No)', 'Churned (Yes)']))
    print("=" * 50)

    # Step 10: Generate and Save Final Diagnostic Visual Plots
    print("\n[STEP 10] Generating and Saving Final Performance Visualizations...")
    
    save_plot_confusion_matrix(final_metrics['Confusion Matrix'], best_model_name, os.path.join(images_dir, "confusion_matrix.png"))
    save_plot_roc_curve(y_test, final_prob, best_model_name, os.path.join(images_dir, "roc_curve.png"))
    save_plot_precision_recall_curve(y_test, final_prob, best_model_name, os.path.join(images_dir, "precision_recall_curve.png"))

    # Extract & Plot Feature Importances / Coefficients
    clf_step = best_pipeline.named_steps['model'] if 'model' in best_pipeline.named_steps else best_pipeline.steps[-1][1]
    if hasattr(clf_step, 'feature_importances_'):
        importances = clf_step.feature_importances_
    elif hasattr(clf_step, 'coef_'):
        importances = np.abs(clf_step.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    save_plot_feature_importance(feature_names, importances, os.path.join(images_dir, "feature_importance.png"))

    # Additional diagnostic plots: Probability Distribution & Confidence Chart
    plt.figure(figsize=(8, 5))
    sns.histplot(final_prob, bins=30, kde=True, color='#3498db')
    plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
    plt.title('Predicted Churn Probability Distribution', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Probability of Churn', fontsize=12)
    plt.ylabel('Customer Count', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "probability_distribution.png"), dpi=300)
    plt.close()

    # Step 12: Save Champion Model Bundle (Pipeline + Feature Metadata)
    model_export_path = os.path.join(saved_model_dir, "churn_prediction_model.pkl")
    model_artifact = {
        'model_pipeline': best_pipeline,
        'feature_names': feature_names,
        'model_name': best_model_name,
        'metrics': final_metrics,
        'comparison_df': comparison_df
    }
    joblib.dump(model_artifact, model_export_path)
    print(f"\n[STEP 12] Champion Model and preprocessor exported successfully to: '{model_export_path}'")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    run_training_pipeline()
