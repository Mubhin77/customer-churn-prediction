# 📊 Customer Churn Prediction

An end-to-end AI/ML solution that predicts customer churn for subscription-based businesses. This system enables proactive detection of at-risk customers, allowing businesses to implement targeted retention strategies and reduce revenue loss.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [MLOps & Deployment](#mlops--deployment)
- [Roadmap](#roadmap)
- [References](#references)
- [Author](#author)

## 🎯 Overview

This project provides a comprehensive churn prediction system that helps businesses identify customers likely to discontinue their service. By leveraging machine learning algorithms, the system analyzes customer demographics, account information, and service usage patterns to predict churn probability.

## 📊 Dataset

**Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn) (Kaggle)

**Key Features:**
- Customer demographics (gender, age, dependents)
- Account information (contract type, payment method, tenure)
- Service usage statistics (internet service, phone service, add-ons)
- Churn label (Yes/No)

## ✨ Features

### Prediction Modes

1. **CSV Upload Prediction**
   - Upload CSV files containing multiple customer records
   - Batch prediction for all records
   - Download results as CSV

2. **Manual Input Prediction**
   - Interactive form for single customer prediction
   - Real-time churn probability calculation
   - Risk level classification (High/Medium/Low)

### Risk Classification

- **High Risk:** Churn probability ≥ 0.7
- **Medium Risk:** Churn probability ≥ threshold
- **Low Risk:** Churn probability < threshold

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Step 1: Clone Repository

```bash
git clone https://github.com/Mubhin77/customer-churn-prediction.git
cd customer-churn-prediction/app
```

### Step 2: Create Virtual Environment

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Application

```bash
uvicorn main:app --reload
```

The application will be available at `http://localhost:8000`

## 💻 Usage

### CSV Batch Prediction

1. Navigate to the web interface
2. Click "Upload CSV" section
3. Select your customer data CSV file
4. Click "Upload & Predict"
5. Download the results with predictions

### Manual Single Prediction

1. Navigate to the "Manual Prediction" section
2. Fill in customer information fields
3. Click "Predict Churn"
4. View the prediction result and risk level

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Loads the prediction interface |
| `/upload_csv` | POST | Accepts CSV file for batch prediction |
| `/download_predictions` | GET | Downloads predictions as CSV |
| `/predict` | POST | Accepts manual input for single prediction |

## 🤖 Model Details

**Algorithm:** Logistic Regression (selected as best performing model)

**Performance Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

**Optimization:**
- Threshold tuning performed for optimal classification
- Cross-validation for robust performance estimation

**Development Location:** 
- EDA: `/notebooks/final_eda.ipynb`
- Model Training: `/model/final.ipynb`

## 📁 Project Structure

```
customer-churn-prediction/
├── app/
│   ├── main.py                    # FastAPI backend application
│   └── prediction_churn.csv       # Downloaded predictions (generated)
├── data/
│   └── churn.csv                  # Training dataset
├── frontend/
│   ├── index.html                 # Web interface
│   ├── script.js                  # Frontend JavaScript
│   └── styles.css                 # Styling
├── model/
│   ├── final.ipynb                # Model training notebook
│   └── *.pkl                      # Trained model files
├── notebooks/
│   └── final_eda.ipynb            # Exploratory Data Analysis
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
└── sample_customers.csv           # Sample data for testing
```

## 🔄 MLOps & Deployment

### Pipeline Architecture

```
Data Ingestion → Data Cleaning → Feature Engineering → Model Training → 
Threshold Tuning → Deployment → Monitoring
```

### Deployment Components

- **Backend:** FastAPI
- **Frontend:** HTML/CSS/JavaScript
- **Model Serving:** Real-time and batch prediction
- **Output:** Downloadable CSV predictions

### Monitoring KPIs

- Model accuracy drift detection
- Precision/recall stability tracking
- Data distribution changes
- Prediction latency
- API performance metrics

## 🗺️ Roadmap (3–6 Months)

- [ ] Implement automated monthly model retraining
- [ ] Develop real-time churn analytics dashboard
- [ ] Integrate model monitoring tools (e.g., Evidently AI)
- [ ] Add version control with MLflow or DVC
- [ ] Expand to multi-channel prediction (email, SMS, push notifications)
- [ ] Implement A/B testing framework for model versions
- [ ] Add explainability features (SHAP values)

## 📚 References

- [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 👤 Author

**Mubhin Basnet**  
BSc IT – King's College  
[GitHub](https://github.com/Mubhin77)

---

⭐ If you find this project helpful, please consider giving it a star!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
