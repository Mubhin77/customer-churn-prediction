import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analysis & Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stMetric {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;}
    h1 {color: #1f77b4;}
    h2 {color: #2c3e50; margin-top: 2rem;}
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk {background-color: #ffebee; border-left: 5px solid #f44336;}
    .medium-risk {background-color: #fff3e0; border-left: 5px solid #ff9800;}
    .low-risk {background-color: #e8f5e9; border-left: 5px solid #4caf50;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Customer Churn Analysis & Prediction System")
st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ System Mode")
mode = st.sidebar.radio("Select Mode:", ["🎓 Training & Analysis", "🔮 Prediction"])

# Initialize session state for model
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}

# Helper functions
def prepare_data_for_training(df):
    """Prepare data for model training"""
    model_df = df.copy()
    
    # Detect and rename target column
    target_col = None
    for col in df.columns:
        if col.lower() in ['churn', 'churned', 'attrition', 'exited']:
            target_col = col
            break
    
    if target_col is None:
        return None, None, None, "No churn column found"
    
    if target_col != 'Churn':
        model_df = model_df.rename(columns={target_col: 'Churn'})
    
    # Clean data
    if 'TotalCharges' in model_df.columns:
        model_df['TotalCharges'] = pd.to_numeric(model_df['TotalCharges'], errors='coerce')
    model_df = model_df.dropna()
    
    # Create binary target
    if model_df['Churn'].dtype == 'object':
        model_df['Churn_flag'] = model_df['Churn'].map(lambda x: 1 if str(x).lower() in ['yes', 'true', '1'] else 0)
    else:
        model_df['Churn_flag'] = model_df['Churn'].astype(int)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = model_df.select_dtypes(include=['object']).columns.tolist()
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')
    if 'customerID' in categorical_cols:
        categorical_cols.remove('customerID')
    
    for col in categorical_cols:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        label_encoders[col] = le
    
    # Prepare features and target
    exclude_cols = ['Churn', 'Churn_flag', 'customerID']
    feature_cols = [col for col in model_df.columns if col not in exclude_cols]
    X = model_df[feature_cols]
    y = model_df['Churn_flag']
    
    return X, y, label_encoders, None

def train_model(X, y):
    """Train Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

def predict_churn_risk(model, data, feature_columns, label_encoders):
    """Predict churn risk for new data"""
    pred_df = data.copy()
    
    # Clean data
    if 'TotalCharges' in pred_df.columns:
        pred_df['TotalCharges'] = pd.to_numeric(pred_df['TotalCharges'], errors='coerce')
    
    # Store customer IDs
    customer_ids = pred_df['customerID'] if 'customerID' in pred_df.columns else None
    
    # Encode categorical variables
    for col, le in label_encoders.items():
        if col in pred_df.columns:
            # Handle unseen categories
            pred_df[col] = pred_df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Ensure we have all required features
    for col in feature_columns:
        if col not in pred_df.columns:
            pred_df[col] = 0
    
    # Select only the features used in training
    X_pred = pred_df[feature_columns]
    
    # Get predictions and probabilities
    predictions = model.predict(X_pred)
    probabilities = model.predict_proba(X_pred)[:, 1]  # Probability of churn
    
    # Classify risk levels
    risk_levels = []
    for prob in probabilities:
        if prob >= 0.7:
            risk_levels.append('High')
        elif prob >= 0.4:
            risk_levels.append('Medium')
        else:
            risk_levels.append('Low')
    
    # Create results dataframe
    results = pd.DataFrame({
        'Churn_Probability': probabilities,
        'Churn_Risk': risk_levels,
        'Will_Churn': ['Yes' if p == 1 else 'No' for p in predictions]
    })
    
    if customer_ids is not None:
        results.insert(0, 'customerID', customer_ids.values)
    
    return results

# =========================
# TRAINING & ANALYSIS MODE
# =========================
if mode == "🎓 Training & Analysis":
    st.header("Training & Analysis Mode")
    st.info("Upload a CSV file with historical customer data including the 'Churn' column to train the model and perform EDA.")
    
    uploaded_file = st.file_uploader("Upload Training Data (CSV with Churn column)", type=['csv'], key='training')
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Prepare data
        with st.spinner("Preparing data and training model..."):
            X, y, label_encoders, error = prepare_data_for_training(df)
            
            if error:
                st.error(f"⚠️ {error}")
                st.stop()
            
            # Train model
            model, accuracy = train_model(X, y)
            
            # Save to session state
            st.session_state.trained_model = model
            st.session_state.feature_columns = X.columns.tolist()
            st.session_state.label_encoders = label_encoders
            
            st.success(f"✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")
        
        # Add download button for model
        col1, col2 = st.columns([1, 3])
        with col1:
            model_data = pickle.dumps({
                'model': model,
                'feature_columns': X.columns.tolist(),
                'label_encoders': label_encoders
            })
            st.download_button(
                label="💾 Download Trained Model",
                data=model_data,
                file_name="churn_model.pkl",
                mime="application/octet-stream"
            )
        
        st.markdown("---")
        
        # Prepare display dataframe
        churn = df.copy()
        if 'TotalCharges' in churn.columns:
            churn['TotalCharges'] = pd.to_numeric(churn['TotalCharges'], errors='coerce')
        churn = churn.dropna()
        if 'Churn' not in churn.columns:
            for col in churn.columns:
                if col.lower() in ['churn', 'churned', 'attrition', 'exited']:
                    churn = churn.rename(columns={col: 'Churn'})
                    break
        if churn['Churn'].dtype == 'object':
            churn['Churn_flag'] = churn['Churn'].map(lambda x: 1 if str(x).lower() in ['yes', 'true', '1'] else 0)
        else:
            churn['Churn_flag'] = churn['Churn'].astype(int)
        
        # Navigation
        st.sidebar.markdown("---")
        st.sidebar.title("📊 Analysis Navigation")
        pages = ["📈 Overview", "👥 Demographics", "💼 Services", "💰 Financial", "🔍 Feature Importance"]
        page = st.sidebar.radio("Go to:", pages)
        
        # === OVERVIEW PAGE ===
        if page == "📈 Overview":
            st.header("Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            churn_count = churn['Churn_flag'].sum()
            churn_rate = (churn_count / len(churn)) * 100
            
            with col1:
                st.metric("Total Customers", f"{len(churn):,}")
            with col2:
                st.metric("Churned", f"{churn_count:,}")
            with col3:
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
            with col4:
                st.metric("Model Accuracy", f"{accuracy*100:.1f}%")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Churn Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(x='Churn', data=churn, palette=['#2ecc71', '#e74c3c'], ax=ax)
                ax.set_title('Churn Distribution', fontweight='bold')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Sample Data")
                st.dataframe(churn.head(10))
        
        # === DEMOGRAPHICS PAGE ===
        elif page == "👥 Demographics":
            st.header("Customer Demographics")
            
            col1, col2 = st.columns(2)
            
            if 'gender' in churn.columns:
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(x='gender', hue='Churn', data=churn, palette='Set2', ax=ax)
                    ax.set_title('Gender vs Churn')
                    st.pyplot(fig)
            
            if 'SeniorCitizen' in churn.columns:
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(x='SeniorCitizen', hue='Churn', data=churn, palette='Set2', ax=ax)
                    ax.set_title('Senior Citizen vs Churn')
                    st.pyplot(fig)
            
            if 'tenure' in churn.columns:
                st.markdown("---")
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.histplot(x='tenure', hue='Churn', data=churn, multiple='stack', bins=30, palette=['#2ecc71', '#e74c3c'], ax=ax)
                ax.set_title('Tenure Distribution')
                st.pyplot(fig)
        
        # === SERVICES PAGE ===
        elif page == "💼 Services":
            st.header("Services & Contract")
            
            col1, col2 = st.columns(2)
            
            if 'Contract' in churn.columns:
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(x='Contract', hue='Churn', data=churn, palette='Set1', ax=ax)
                    ax.set_title('Contract Type vs Churn')
                    plt.xticks(rotation=15)
                    st.pyplot(fig)
            
            if 'InternetService' in churn.columns:
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(x='InternetService', hue='Churn', data=churn, palette='Set3', ax=ax)
                    ax.set_title('Internet Service vs Churn')
                    st.pyplot(fig)
        
        # === FINANCIAL PAGE ===
        elif page == "💰 Financial":
            st.header("Financial Analysis")
            
            if 'MonthlyCharges' in churn.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(x='Churn', y='MonthlyCharges', data=churn, palette=['#2ecc71', '#e74c3c'], ax=ax)
                    ax.set_title('Monthly Charges vs Churn')
                    st.pyplot(fig)
                
                with col2:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Avg Charges (Churned)", 
                                f"${churn[churn['Churn_flag']==1]['MonthlyCharges'].mean():.2f}")
                    with col_b:
                        st.metric("Avg Charges (Retained)", 
                                f"${churn[churn['Churn_flag']==0]['MonthlyCharges'].mean():.2f}")
        
        # === FEATURE IMPORTANCE PAGE ===
        elif page == "🔍 Feature Importance":
            st.header("Feature Importance")
            
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 8))
                top_15 = feature_importance.head(15)
                sns.barplot(x='Importance', y='Feature', data=top_15, palette='viridis', ax=ax)
                ax.set_title('Top 15 Features by Importance')
                st.pyplot(fig)
            
            with col2:
                st.subheader("All Features")
                st.dataframe(feature_importance, height=500)

# =========================
# PREDICTION MODE
# =========================
else:  # Prediction mode
    st.header("Prediction Mode")
    st.info("Upload a CSV file with new customer data (without Churn column) to predict churn risk.")
    
    # Check if model is trained
    if st.session_state.trained_model is None:
        st.warning("⚠️ No trained model found! Please train a model first in 'Training & Analysis' mode.")
        
        st.markdown("### Or upload a pre-trained model:")
        model_file = st.file_uploader("Upload Model File (.pkl)", type=['pkl'])
        
        if model_file is not None:
            try:
                model_data = pickle.load(model_file)
                st.session_state.trained_model = model_data['model']
                st.session_state.feature_columns = model_data['feature_columns']
                st.session_state.label_encoders = model_data['label_encoders']
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    
    if st.session_state.trained_model is not None:
        st.success("✅ Model is ready for predictions!")
        
        uploaded_file = st.file_uploader("Upload Customer Data for Prediction (CSV)", type=['csv'], key='prediction')
        
        if uploaded_file is not None:
            # Load data
            pred_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(pred_data)} records")
            
            # Show sample
            st.subheader("Input Data Sample")
            st.dataframe(pred_data.head())
            
            # Make predictions
            with st.spinner("Making predictions..."):
                results = predict_churn_risk(
                    st.session_state.trained_model,
                    pred_data,
                    st.session_state.feature_columns,
                    st.session_state.label_encoders
                )
            
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            high_risk = len(results[results['Churn_Risk'] == 'High'])
            medium_risk = len(results[results['Churn_Risk'] == 'Medium'])
            low_risk = len(results[results['Churn_Risk'] == 'Low'])
            
            with col1:
                st.metric("Total Customers", len(results))
            with col2:
                st.metric("🔴 High Risk", high_risk, 
                         delta=f"{high_risk/len(results)*100:.1f}%", delta_color="inverse")
            with col3:
                st.metric("🟡 Medium Risk", medium_risk,
                         delta=f"{medium_risk/len(results)*100:.1f}%", delta_color="off")
            with col4:
                st.metric("🟢 Low Risk", low_risk,
                         delta=f"{low_risk/len(results)*100:.1f}%", delta_color="normal")
            
            st.markdown("---")
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                risk_counts = results['Churn_Risk'].value_counts()
                colors = {'High': '#f44336', 'Medium': '#ff9800', 'Low': '#4caf50'}
                bars = ax.bar(risk_counts.index, risk_counts.values, 
                             color=[colors[x] for x in risk_counts.index])
                ax.set_title('Customer Risk Distribution', fontweight='bold')
                ax.set_ylabel('Number of Customers')
                ax.set_xlabel('Risk Level')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Probability Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(results['Churn_Probability'], bins=30, color='#1f77b4', alpha=0.7)
                ax.axvline(x=0.7, color='red', linestyle='--', label='High Risk Threshold')
                ax.axvline(x=0.4, color='orange', linestyle='--', label='Medium Risk Threshold')
                ax.set_title('Churn Probability Distribution', fontweight='bold')
                ax.set_xlabel('Churn Probability')
                ax.set_ylabel('Number of Customers')
                ax.legend()
                st.pyplot(fig)
            
            st.markdown("---")
            
            # Full results table
            st.subheader("Detailed Predictions")
            
            # Add styling
            def highlight_risk(row):
                if row['Churn_Risk'] == 'High':
                    return ['background-color: #ffebee'] * len(row)
                elif row['Churn_Risk'] == 'Medium':
                    return ['background-color: #fff3e0'] * len(row)
                else:
                    return ['background-color: #e8f5e9'] * len(row)
            
            styled_results = results.style.apply(highlight_risk, axis=1)
            st.dataframe(styled_results, use_container_width=True)
            
            # Download button
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Combine original data with predictions
                final_output = pd.concat([pred_data.reset_index(drop=True), 
                                         results.reset_index(drop=True)], axis=1)
                
                csv = final_output.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
            
            # Risk-specific insights
            st.markdown("---")
            st.header("🎯 Risk-Specific Recommendations")
            
            tab1, tab2, tab3 = st.tabs(["🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"])
            
            with tab1:
                high_risk_data = results[results['Churn_Risk'] == 'High']
                st.write(f"**{len(high_risk_data)} customers** identified as high risk")
                st.markdown("""
                **Recommended Actions:**
                - 🎯 Immediate retention campaign
                - 💰 Offer special discounts or loyalty rewards
                - 📞 Personal outreach from customer success team
                - 🔄 Consider contract renegotiation
                """)
                if len(high_risk_data) > 0:
                    st.dataframe(high_risk_data.head(10))
            
            with tab2:
                medium_risk_data = results[results['Churn_Risk'] == 'Medium']
                st.write(f"**{len(medium_risk_data)} customers** identified as medium risk")
                st.markdown("""
                **Recommended Actions:**
                - 📧 Engagement email campaigns
                - 🎁 Introduce them to new features/services
                - 📊 Monitor usage patterns closely
                - 💬 Gather feedback surveys
                """)
                if len(medium_risk_data) > 0:
                    st.dataframe(medium_risk_data.head(10))
            
            with tab3:
                low_risk_data = results[results['Churn_Risk'] == 'Low']
                st.write(f"**{len(low_risk_data)} customers** identified as low risk")
                st.markdown("""
                **Recommended Actions:**
                - ⭐ Leverage for referrals and testimonials
                - 🎯 Upsell premium features
                - 💚 Maintain current satisfaction levels
                - 🏆 Consider loyalty program enrollment
                """)
                if len(low_risk_data) > 0:
                    st.dataframe(low_risk_data.head(10))

st.sidebar.markdown("---")
st.sidebar.markdown("""
**💡 Tips:**
- Train model with historical data first
- Then use prediction mode for new customers
- Download model to reuse later
""")