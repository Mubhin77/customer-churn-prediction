# 📊 Customer Churn Prediction

## Overview
This project is an end-to-end AI/ML solution that predicts customer churn for a subscription-based business.  
It enables businesses to proactively detect customers likely to churn so they can implement targeted retention strategies, reducing revenue loss.

## Dataset
We use the **Telco Customer Churn Dataset** (Kaggle):  
[Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

**Key Features:**
- Customer demographics
- Account information
- Service usage statistics
- Churn label (Yes/No)

---

## Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/Mubhin77/customer-churn-prediction.git
cd customer-churn-prediction/app


2. Create Virtual Environment
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Run Application
uvicorn main:app --reload





# Two prediction modes:

- CSV Upload Prediction

- Upload a CSV containing customer data

Predictions generated for all records

Download prediction results

Manual Input Prediction

Fill form fields manually

Get churn prediction instantly

Backend Endpoints
Endpoint	Method	Description
/	GET	Loads the prediction interface
/upload_csv	POST	Accepts CSV file for batch prediction
/download_predictions	GET	Downloads predictions as CSV
/predict	POST	Accepts manual input for single prediction
Model Details

Model: Logistic Regression (best performing model)

Threshold tuning performed for optimal performance

Risk Levels:

High Risk: Probability ≥ 0.7

Medium Risk: Probability ≥ threshold

Low Risk: Probability < threshold

EDA & Model Development

Exploratory Data Analysis and model training are performed in /notebooks/final_eda.ipynb and /model/final.ipynb.

Key steps:

Data cleaning & preprocessing

Feature engineering

Model training and tuning

Threshold analysis

Model evaluation (Accuracy, Precision, Recall, F1-score, ROC-AUC)

MLOps & Deployment Plan

Pipeline:

Data Ingestion → Data Cleaning → Model Training → Threshold Tuning → Deployment → Monitoring


Deployment includes:

FastAPI backend

HTML/JS frontend

Downloadable prediction CSV

Manual input prediction form

Monitoring KPIs:

Model accuracy drift

Precision stability

Data distribution changes

Roadmap (3–6 Months)

Automate monthly retraining

Add real-time churn analytics dashboard

Integrate model monitoring tools

Version control with MLflow/DVC

Expand to multi-channel prediction

References

Telco Customer Churn Dataset — Kaggle

Scikit-learn Documentation

FastAPI Documentation

Author

Mubhin Basnet
King’s College — BSc IT