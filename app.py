import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .fraud {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .legitimate {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Best model first. The name shown in the UI comes from the object that actually
# loaded, so the app can never again advertise a model it is not running.
MODEL_FILES = (
    'best_fraud_model.joblib',             # XGBoost, trained on the full data
    'logistic_regression_baseline.joblib',  # fallback baseline
)


@st.cache_resource
def load_model():
    """Load the best available fraud detection model."""
    for path in MODEL_FILES:
        try:
            model = joblib.load(path)
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"⚠️ Could not load {path}: {e}")
            continue
        return model, type(model).__name__

    st.error(
        "❌ No model file could be loaded. Expected one of "
        f"{', '.join(MODEL_FILES)} next to app.py."
    )
    return None, None

@st.cache_data
def load_scaler_data():
    """Load the original dataset to fit the scaler"""
    try:
        df = pd.read_csv('creditcard.csv')
        # Fit scaler on all features except Class
        X = df.drop('Class', axis=1)
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler
    except Exception as e:
        st.error(f"❌ Error loading training data: {str(e)}")
        return None

def preprocess_data(df, scaler):
    """Preprocess the uploaded data"""
    # Check if Class column exists (for evaluation purposes)
    has_class = 'Class' in df.columns
    
    if has_class:
        X = df.drop('Class', axis=1)
        y = df['Class']
    else:
        X = df.copy()
        y = None
    
    # Ensure all required columns are present
    required_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                        'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                        'V28', 'Amount']
    
    missing_columns = [col for col in required_columns if col not in X.columns]
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        return None, None, None
    
    # Reorder columns to match training data
    X = X[required_columns]
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    return X_scaled, y, X

def main():
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model and scaler
    with st.spinner("Loading model and preprocessing tools..."):
        model, model_name = load_model()
        scaler = load_scaler_data()
    
    if model is None or scaler is None:
        st.stop()
    
    # Display model information
    st.success(f"✅ Model loaded: **{model_name}**")
    
    # Sidebar
    st.sidebar.header("📊 Options")
    st.sidebar.markdown("### Upload your data")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with credit card transaction data. Required columns: Time, V1-V28, Amount, and optionally Class."
    )
    
    # Display sample data format
    with st.sidebar.expander("📋 Expected Data Format"):
        st.markdown("""
        Your CSV file should contain the following columns:
        - **Time**: Time elapsed between transactions
        - **V1 to V28**: PCA transformed features
        - **Amount**: Transaction amount
        - **Class** (optional): 0 for legitimate, 1 for fraud (for evaluation)
        """)
        st.markdown("**Note:** If Class column is present, the app will show accuracy metrics.")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! ({len(df)} rows)")
            
            # Display data preview
            with st.expander("📄 Data Preview", expanded=False):
                st.dataframe(df.head(10))
                st.info(f"**Total rows:** {len(df)} | **Total columns:** {len(df.columns)}")
            
            # Preprocess data
            with st.spinner("Preprocessing data..."):
                X_scaled, y_true, X_original = preprocess_data(df, scaler)
            
            if X_scaled is not None:
                # Make predictions
                with st.spinner("Making predictions..."):
                    predictions = model.predict(X_scaled)
                    prediction_proba = model.predict_proba(X_scaled)
                    
                    # Create results dataframe
                    results_df = df.copy()
                    results_df['Predicted_Class'] = predictions
                    results_df['Fraud_Probability'] = prediction_proba[:, 1]
                    results_df['Legitimate_Probability'] = prediction_proba[:, 0]
                
                # Main results section
                st.markdown("## 📊 Prediction Results")
                
                # Calculate statistics
                total_transactions = len(predictions)
                fraud_count = int(np.sum(predictions))
                legitimate_count = total_transactions - fraud_count
                fraud_percentage = (fraud_count / total_transactions) * 100
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", f"{total_transactions:,}")
                
                with col2:
                    st.metric("🚨 Fraudulent", f"{fraud_count:,}", 
                             delta=f"{fraud_percentage:.2f}%", 
                             delta_color="inverse")
                
                with col3:
                    st.metric("✅ Legitimate", f"{legitimate_count:,}", 
                             delta=f"{100-fraud_percentage:.2f}%")
                
                with col4:
                    avg_fraud_prob = np.mean(prediction_proba[:, 1]) * 100
                    st.metric("Avg Fraud Risk", f"{avg_fraud_prob:.2f}%")
                
                # Accuracy metrics if Class column exists
                if y_true is not None:
                    st.markdown("### 📈 Model Performance Metrics")
                    
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                    
                    accuracy = accuracy_score(y_true, predictions)
                    precision = precision_score(y_true, predictions, zero_division=0)
                    recall = recall_score(y_true, predictions, zero_division=0)
                    f1 = f1_score(y_true, predictions, zero_division=0)
                    
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric("Accuracy", f"{accuracy*100:.2f}%")
                    with metric_col2:
                        st.metric("Precision", f"{precision*100:.2f}%")
                    with metric_col3:
                        st.metric("Recall", f"{recall*100:.2f}%")
                    with metric_col4:
                        st.metric("F1-Score", f"{f1*100:.2f}%")
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_true, predictions)
                    st.markdown("### 🔍 Confusion Matrix")
                    
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted Legitimate', 'Predicted Fraud'],
                        y=['Actual Legitimate', 'Actual Fraud'],
                        colorscale='Blues',
                        text=cm,
                        texttemplate='%{text}',
                        textfont={"size": 16},
                        colorbar=dict(title="Count")
                    ))
                    fig_cm.update_layout(
                        title="Confusion Matrix",
                        width=600,
                        height=400
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                # Visualizations
                st.markdown("### 📉 Visualizations")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Fraud distribution pie chart
                    fig_pie = px.pie(
                        values=[legitimate_count, fraud_count],
                        names=['Legitimate', 'Fraudulent'],
                        title="Transaction Distribution",
                        color_discrete_map={'Legitimate': '#4caf50', 'Fraudulent': '#f44336'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with viz_col2:
                    # Fraud probability distribution
                    fig_hist = px.histogram(
                        results_df,
                        x='Fraud_Probability',
                        nbins=50,
                        title="Fraud Probability Distribution",
                        labels={'Fraud_Probability': 'Fraud Probability', 'count': 'Number of Transactions'},
                        color_discrete_sequence=['#f44336']
                    )
                    fig_hist.update_layout(showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Detailed results table
                st.markdown("### 📋 Detailed Results")
                
                # Filter options
                filter_option = st.selectbox(
                    "Filter results:",
                    ["All Transactions", "Fraudulent Only", "Legitimate Only", "High Risk (Prob > 0.7)", "Medium Risk (0.3 < Prob < 0.7)"]
                )
                
                if filter_option == "Fraudulent Only":
                    filtered_df = results_df[results_df['Predicted_Class'] == 1]
                elif filter_option == "Legitimate Only":
                    filtered_df = results_df[results_df['Predicted_Class'] == 0]
                elif filter_option == "High Risk (Prob > 0.7)":
                    filtered_df = results_df[results_df['Fraud_Probability'] > 0.7]
                elif filter_option == "Medium Risk (0.3 < Prob < 0.7)":
                    filtered_df = results_df[(results_df['Fraud_Probability'] > 0.3) & (results_df['Fraud_Probability'] < 0.7)]
                else:
                    filtered_df = results_df
                
                # Display filtered results
                st.dataframe(
                    filtered_df[['Predicted_Class', 'Fraud_Probability', 'Legitimate_Probability'] + 
                               [col for col in filtered_df.columns if col not in ['Predicted_Class', 'Fraud_Probability', 'Legitimate_Probability']]].head(100),
                    use_container_width=True
                )
                
                # Download results
                st.markdown("### 💾 Download Results")
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )
                
                # Risk analysis
                st.markdown("### ⚠️ Risk Analysis")
                
                high_risk = results_df[results_df['Fraud_Probability'] > 0.7]
                medium_risk = results_df[(results_df['Fraud_Probability'] > 0.3) & (results_df['Fraud_Probability'] <= 0.7)]
                low_risk = results_df[results_df['Fraud_Probability'] <= 0.3]
                
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                
                with risk_col1:
                    st.warning(f"🔴 High Risk: {len(high_risk)} transactions")
                with risk_col2:
                    st.info(f"🟡 Medium Risk: {len(medium_risk)} transactions")
                with risk_col3:
                    st.success(f"🟢 Low Risk: {len(low_risk)} transactions")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
    
    else:
        # Show instructions when no file is uploaded
        st.info("👈 Please upload a CSV file from the sidebar to get started.")
        
        st.markdown("""
        ### 📖 How to Use
        
        1. **Prepare your data**: Ensure your CSV file has the following columns:
           - Time, V1-V28, Amount
           - Optionally include 'Class' column for evaluation
        
        2. **Upload**: Click on "Browse files" in the sidebar and select your CSV file
        
        3. **View Results**: The app will automatically:
           - Preprocess your data
           - Make predictions using the best model
           - Display statistics and visualizations
           - Provide detailed results
        
        4. **Download**: Download the results with predictions and probabilities
        
        ### 🔍 About the Model
        
        This application uses a state-of-the-art machine learning model (XGBoost) 
        trained on the Kaggle Credit Card Fraud Detection dataset. The model has 
        been optimized for high accuracy in detecting fraudulent transactions.
        
        **Model Performance:**
        - High accuracy in fraud detection
        - Balanced precision and recall
        - Real-time predictions
        """)
        
        # Display sample data structure
        st.markdown("### 📋 Sample Data Structure")
        sample_data = {
            'Time': [0, 0, 1],
            'V1': [-1.36, 1.19, -1.36],
            'V2': [-0.07, 0.27, -1.34],
            'V28': [-0.02, 0.01, 0.06],
            'Amount': [149.62, 2.69, 378.66],
            'Class': [0, 0, 0]  # Optional
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)

if __name__ == "__main__":
    main()

