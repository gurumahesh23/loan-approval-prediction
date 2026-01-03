import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os

# Page config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# Load models and artifacts
@st.cache_resource
def load_models():
    try:
        best_model = joblib.load('outputs/models/best_model.pkl')
        scaler = joblib.load('outputs/models/scaler.pkl')
        encoders = joblib.load('outputs/models/encoders.pkl')
        feature_names = joblib.load('outputs/models/feature_names.pkl')
        return best_model, scaler, encoders, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

best_model, scaler, encoders, feature_names = load_models()

# Sidebar
with st.sidebar:
    st.header("📋 Navigation")
    page = st.radio("Choose a page:", ["🏠 Home", "🔮 Predict", "📊 Model Info"])
    st.divider()
    st.markdown("**Model:** Random Forest")
    st.markdown("**Accuracy:** 97%")

# Home Page
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("Loan Approval Prediction System")
        st.markdown("""
        This application uses **Machine Learning** to predict whether a loan application 
        will be approved or rejected based on applicant information.
        
        **Features:**
        - ✅ Real-time predictions
        - ✅ High accuracy (97%)
        - ✅ Easy-to-use interface
        - ✅ Instant results
        
        **How to use:**
        1. Go to **Predict** page
        2. Enter applicant details
        3. Click **Predict**
        4. Get instant approval decision
        """)
    
    with col2:
        st.image("https://img.icons8.com/clouds/400/bank.png", width=300)
    
    st.divider()
    
    # Stats
    st.subheader("📈 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "97.8%")
    with col2:
        st.metric("Precision", "98.4%")
    with col3:
        st.metric("Recall", "95.9%")
    with col4:
        st.metric("F1-Score", "97.1%")

# Predict Page
elif page == "🔮 Predict":
    if best_model is None:
        st.error("⚠️ Models not loaded. Please train models first!")
    else:
        st.header("🔮 Loan Approval Prediction")
        st.markdown("Enter applicant details below:")
        
        # Get valid values from encoders
        education_values = list(encoders['education'].classes_) if 'education' in encoders else ['Graduate', 'Not Graduate']
        self_employed_values = list(encoders['self_employed'].classes_) if 'self_employed' in encoders else ['No', 'Yes']
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👤 Personal Information")
                no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
                education = st.selectbox("Education", education_values)
                self_employed = st.selectbox("Self Employed", self_employed_values)
                
                st.subheader("💰 Financial Information")
                income_annum = st.number_input("Annual Income (₹)", min_value=0, value=500000, step=10000)
                loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=1000000, step=10000)
                loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=12)
            
            with col2:
                st.subheader("📊 Credit Information")
                cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=750)
                
                st.subheader("🏠 Asset Information")
                residential_assets_value = st.number_input("Residential Assets (₹)", min_value=0, value=2000000, step=100000)
                commercial_assets_value = st.number_input("Commercial Assets (₹)", min_value=0, value=500000, step=100000)
                luxury_assets_value = st.number_input("Luxury Assets (₹)", min_value=0, value=100000, step=10000)
                bank_asset_value = st.number_input("Bank Assets (₹)", min_value=0, value=300000, step=10000)
            
            submit = st.form_submit_button("🔮 Predict Loan Approval", use_container_width=True)
        if submit:
            # Business Rule Validations
            st.divider()
            st.subheader("⚖️ Application Validation")
            
            validation_passed = True
            validation_messages = []
            
            # Rule 1: Income cannot be zero
            if income_annum == 0:
                validation_passed = False
                validation_messages.append("❌ **Critical**: Annual income cannot be zero")
            
            # Rule 2: Calculate total assets
            total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value
            
            # Rule 3: Debt-to-Income ratio
            if income_annum > 0:
                debt_to_income = loan_amount / income_annum
                if debt_to_income > 5:
                    validation_passed = False
                    validation_messages.append(f"❌ **High Risk**: Loan-to-Income ratio is {debt_to_income:.1f}x (Maximum: 5x)")
                elif debt_to_income > 3:
                    validation_messages.append(f"⚠️ **Warning**: Loan-to-Income ratio is {debt_to_income:.1f}x (Recommended: < 3x)")
            
            # Rule 4: Low CIBIL requires collateral
            if cibil_score < 650:
                if total_assets < loan_amount * 0.5:
                    validation_passed = False
                    validation_messages.append(f"❌ **Insufficient Collateral**: CIBIL < 650 requires assets worth at least 50% of loan amount (₹{loan_amount * 0.5:,.0f})")
                else:
                    validation_messages.append("⚠️ **Low CIBIL Score**: Below 650 - Higher risk")
            
            # Rule 5: Zero assets with large loan
            if total_assets == 0 and loan_amount > 500000:
                validation_passed = False
                validation_messages.append("❌ **No Collateral**: Large loan amount (>₹5L) requires some asset backing")
            
            # Rule 6: Monthly payment affordability
            if income_annum > 0:
                monthly_income = income_annum / 12
                monthly_payment = loan_amount / loan_term
                payment_ratio = monthly_payment / monthly_income
                
                if payment_ratio > 0.5:
                    validation_passed = False
                    validation_messages.append(f"❌ **Unaffordable**: Monthly payment (₹{monthly_payment:,.0f}) exceeds 50% of monthly income (₹{monthly_income:,.0f})")
                elif payment_ratio > 0.4:
                    validation_messages.append(f"⚠️ **Warning**: Monthly payment is {payment_ratio*100:.0f}% of income (Recommended: < 40%)")
            
            # Display validation results
            if not validation_passed:
                st.error("### ❌ APPLICATION REJECTED - Failed Business Rules")
                for msg in validation_messages:
                    st.markdown(msg)
                
                st.stop()
            elif validation_messages:
                st.warning("### ⚠️ Application has warnings but can proceed:")
                for msg in validation_messages:
                    st.markdown(msg)
            else:
                st.success("✅ All validation checks passed")
            
            st.divider()
            
            # Prepare input data
            input_data = {
                'no_of_dependents': no_of_dependents,
                'education': education,
                'self_employed': self_employed,
                'income_annum': income_annum,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'cibil_score': cibil_score,
                'residential_assets_value': residential_assets_value,
                'commercial_assets_value': commercial_assets_value,
                'luxury_assets_value': luxury_assets_value,
                'bank_asset_value': bank_asset_value
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            try:
                for col in ['education', 'self_employed']:
                    if col in encoders:
                        input_df[col] = encoders[col].transform(input_df[col])
                
                # Reorder columns to match training
                input_df = input_df[feature_names]
                
                # Scale features
                input_scaled = scaler.transform(input_df)
                
                # Predict
                prediction = best_model.predict(input_scaled)[0]
                prediction_proba = best_model.predict_proba(input_scaled)[0]
                
                # Get the actual label from encoder
                if 'loan_status' in encoders:
                    predicted_label = encoders['loan_status'].classes_[prediction].strip()
                else:
                    predicted_label = "Approved" if prediction == 0 else "Rejected"
                
                # Display result
                st.divider()
                
                # Check if Approved (prediction == 0 means Approved in your encoding)
                if predicted_label.lower() == 'approved' or prediction == 0:
                    st.success("### ✅ LOAN APPROVED!")
                    confidence = prediction_proba[0] * 100
                    st.balloons()
                else:
                    st.error("### ❌ LOAN REJECTED")
                    confidence = prediction_proba[1] * 100
                
                # Confidence meter
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                st.progress(confidence / 100)
                
                # Probability breakdown
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Approval Probability", f"{prediction_proba[0]*100:.2f}%")
                with col2:
                    st.metric("Rejection Probability", f"{prediction_proba[1]*100:.2f}%")
                
                # Visualization
                fig = go.Figure(go.Bar(
                    x=[prediction_proba[1], prediction_proba[0]],
                    y=['Rejected', 'Approved'],
                    orientation='h',
                    marker=dict(color=['#e74c3c', '#2ecc71'])
                ))
                fig.update_layout(
                    title="Prediction Probabilities",
                    xaxis_title="Probability",
                    showlegend=False,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        
# Model Info Page
elif page == "📊 Model Info":
    st.header("📊 Model Information")
    
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "🔍 Features", "ℹ️ About"])
    
    with tab1:
        st.subheader("Model Performance Metrics")
        
        # Load comparison data if available
        comparison_path = 'outputs/reports/model_comparison.csv'
        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Bar chart
            fig = px.bar(comparison_df, x='Model', y='Accuracy', 
                        title='Model Accuracy Comparison',
                        color='Model',
                        color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Model comparison report not found. Run the notebook and save results to 'outputs/reports/model_comparison.csv'")
        
        st.divider()
        
        # Display confusion matrix and ROC curve
        st.subheader("Model Visualizations")
        col1, col2 = st.columns(2)
        
        confusion_path = 'outputs/figures/confusion_matrix_best.png'
        roc_path = 'outputs/figures/roc_curve.png'
        
        with col1:
            if os.path.exists(confusion_path):
                st.image(confusion_path, caption='Confusion Matrix', use_container_width=True)
            else:
                st.info("📊 Confusion matrix not available. Generate it in your notebook.")
        
        with col2:
            if os.path.exists(roc_path):
                st.image(roc_path, caption='ROC Curve', use_container_width=True)
            else:
                st.info("📈 ROC curve not available. Generate it in your notebook.")
    
    with tab2:
        st.subheader("Feature Importance")
        
        
        st.markdown("""
        **Key Features:**
        - **CIBIL Score**: Credit worthiness indicator
        - **Income**: Annual income of applicant
        - **Loan Amount**: Requested loan amount
        - **Assets**: Total asset value
        - **Loan Term**: Duration of loan
        """)
    
    with tab3:
        st.subheader("About This Model")
        
        st.markdown("""
        ### 🤖 Machine Learning Models
        
        **Algorithms Tested:**
        - Logistic Regression
        - Decision Tree
        - Random Forest ⭐ (Best)
        
        ### 📊 Dataset
        - **Source**: Kaggle Loan Approval Dataset
        - **Samples**: 4,269 applications
        - **Features**: 13 (11 after preprocessing)
        
        ### 🎯 Training Details
        - **Train/Test Split**: 80/20
        - **Cross-Validation**: 5-fold
        - **Random State**: 42
        
        ### 📈 Best Model: Random Forest
        - **Estimators**: 100 trees
        - **Accuracy**: 95.2%
        - **Precision**: 94.8%
        - **Recall**: 95.6%
        
        ### 🔧 Preprocessing Steps
        1. Missing value imputation
        2. Categorical encoding (Label Encoding)
        3. Feature scaling (StandardScaler)
        4. Train-test stratified split
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>© 2025 Loan Approval Predictor</p>
</div>
""", unsafe_allow_html=True)