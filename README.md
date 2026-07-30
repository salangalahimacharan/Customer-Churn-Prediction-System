# Customer Churn Prediction using Machine Learning

Live project : https://customer-churn-prediction-system-nv3k.onrender.com

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, end-to-end Machine Learning pipeline for predicting customer churn in subscription and telecom businesses. Designed with modular code architecture, rigorous cross-validation, automated hyperparameter tuning, and real-time actionable retention recommendation engines.

---

## 📌 Project Overview

Customer churn occurs when customers stop doing business with a company or cancel subscription services. Acquiring new customers costs 5x to 25x more than retaining existing ones. 

This repository delivers a production-ready machine learning framework to identify customers at risk of churn before they leave, enabling retention teams to take targeted, cost-effective proactive measures.

---

## 🎯 Business Objectives

1. **Predict High-Risk Churners**: Accurately predict whether a customer will churn (`Yes` / `No`) using usage, demographic, and support interaction features.
2. **Quantify Risk Tiers**: Assign a continuous probability score and categorize accounts into discrete risk tiers (`Low`, `Medium`, `High`, `Very High`).
3. **Automate Business Retention Strategies**: Provide real-time, context-aware operational recommendations for retention teams.
4. **Identify Key Churn Drivers**: Extract feature importance ranking to uncover top operational drivers influencing customer attrition.

---

## 📂 Project Structure

```
Customer-Churn-Prediction/
│
├── dataset/
│   └── customer_churn.csv                # 5,500+ realistic customer records
│
├── notebook/
│   └── Customer_Churn_Prediction.ipynb   # Executable Jupyter Notebook (Steps 1-17 + Bonus)
│
├── src/
│   ├── train_model.py                     # Pipeline benchmark, cross-validation & tuning script
│   ├── predict.py                         # Interactive prediction CLI & library engine
│   └── utils.py                           # Preprocessing, metrics, plotting & risk logic
│
├── images/                                # High-resolution EDA and performance plots
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── feature_importance.png
│   └── probability_distribution.png
│
├── saved_model/
│   └── churn_prediction_model.pkl        # Serialized pipeline & champion model bundle
│
├── requirements.txt                       # Python dependencies
├── README.md                              # Comprehensive documentation
└── LICENSE                                # MIT License
```

---

## 📊 Dataset Schema

The dataset contains 5,500+ realistic customer records featuring realistic business relationships and an imbalanced churn ratio (~33% churn):

| Feature Name | Type | Description |
|---|---|---|
| `CustomerID` | String | Unique identifier for each customer |
| `Gender` | Categorical | Customer gender (`Male` / `Female`) |
| `SeniorCitizen` | Binary | Senior citizen status (`1` / `0`) |
| `Age` | Numerical | Customer age in years (18 to 80) |
| `Tenure` | Numerical | Months customer has stayed with company (1 to 72) |
| `MonthlyCharges` | Numerical | Monthly bill amount in USD |
| `TotalCharges` | Numerical | Cumulative charges over tenure |
| `InternetService` | Categorical | Internet service type (`Fiber optic`, `DSL`, `No`) |
| `Contract` | Categorical | Contract term (`Month-to-month`, `One year`, `Two year`) |
| `PaymentMethod` | Categorical | Payment method used |
| `TechSupport` | Categorical | Technical support subscription status |
| `OnlineSecurity` | Categorical | Online security add-on status |
| `StreamingTV` | Categorical | TV streaming add-on status |
| `StreamingMovies` | Categorical | Movie streaming add-on status |
| `PhoneService` | Categorical | Phone service subscription |
| `MultipleLines` | Categorical | Multiple line subscription |
| `Dependents` | Categorical | Customer dependents status |
| `Complaints` | Binary | Whether customer lodged a complaint recently (`1` / `0`) |
| `SatisfactionScore` | Numerical | Survey satisfaction score (1 to 5) |
| `SupportTickets` | Numerical | Number of support tickets logged (0 to 10) |
| `UsageHours` | Numerical | Monthly app / service usage hours |
| `DataConsumption` | Numerical | Monthly data consumption in GB |
| **`Churn`** | Binary Target | **Target Variable (`Yes` / `No`)** |

---

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn
- **Model Persistence**: Joblib
- **Environment**: Jupyter Notebook / Python CLI

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A: Train and Benchmark Models
Run `train_model.py` to clean data, train 5 algorithms, run 5-fold cross validation, perform hyperparameter grid search, export plots to `images/`, and serialize the champion model:
```bash
python src/train_model.py
```

### Option B: Run Interactive Inference / Predictions
Run `predict.py` to evaluate sample high-risk and low-risk customer profiles and view visual risk bars and retention recommendations:
```bash
python src/predict.py
```

### Option C: Explore Jupyter Notebook
Open the step-by-step notebook in Jupyter:
```bash
jupyter notebook notebook/Customer_Churn_Prediction.ipynb
```

---

## 📈 Model Comparison & Benchmark Results

All models were evaluated on an 80/20 stratified test split:

| Algorithm | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score | ROC-AUC | 5-Fold CV F1 |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 82.5% | 82.1% | 0.7420 | 0.7280 | **0.7349** | **0.8872** | 0.7310 |
| **Random Forest** | 99.8% | 83.4% | 0.7710 | 0.7150 | 0.7419 | 0.8920 | 0.7380 |
| **Decision Tree** | 84.1% | 80.2% | 0.7100 | 0.6900 | 0.6998 | 0.8310 | 0.6920 |
| **K-Nearest Neighbors**| 85.2% | 79.8% | 0.7020 | 0.6810 | 0.6913 | 0.8420 | 0.6880 |
| **Support Vector Machine**| 84.0% | 82.3% | 0.7510 | 0.7120 | 0.7309 | 0.8810 | 0.7290 |

---

## 🖼️ Generated Visualizations

The pipeline generates publication-quality diagnostic plots in `images/`:

- **Churn Distribution**: `images/churn_distribution.png`
- **Feature Correlation Matrix**: `images/correlation_heatmap.png`
- **Confusion Matrix**: `images/confusion_matrix.png`
- **ROC Curve**: `images/roc_curve.png`
- **Precision-Recall Curve**: `images/precision_recall_curve.png`
- **Key Feature Importances**: `images/feature_importance.png`
- **Probability Distribution**: `images/probability_distribution.png`

---

## 💡 Evaluation Metrics Explained

- **Accuracy**: Overall proportion of correct predictions. (High accuracy can be misleading in imbalanced datasets).
- **Precision**: Out of all predicted churners, how many actually churned? Important for minimizing wasted retention budget.
- **Recall (Sensitivity)**: Out of all actual churners, how many did the model identify? Critical to prevent lost revenue.
- **F1 Score**: Harmonic mean of Precision and Recall. The primary metric for balancing false positives and false negatives.
- **ROC-AUC Score**: Measures how well the model separates churners from non-churners across all classification thresholds.
- **Confusion Matrix**: 2x2 matrix detailing True Positives, True Negatives, False Positives, and False Negatives.

---

## 🔮 Future Improvements

1. **SHAP (SHapley Additive exPlanations)** integration for local feature explanations on individual customer predictions.
2. **REST API Deployment**: Wrap `predict.py` with FastAPI or Flask for production cloud deployment.
3. **Streamlit Web Application**: Interactive dashboard for business users to simulate customer churn risk dynamically.
4. **Automated ML (AutoML)**: Integrate Optuna for advanced Bayesian hyperparameter optimization.

---

## 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
