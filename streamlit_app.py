import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import io

# Set page configuration for a wide layout and attractive theme
st.set_page_config(page_title="Air Quality Prediction", layout="wide", page_icon="🌬️")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stSlider>div>div {
        color: #4CAF50;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# Set random seed for reproducibility
np.random.seed(42)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Overview", "EDA", "Model Results", "Predict"])

# Initialize session state for data and models
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.model_trained = False
    st.session_state.rf = None
    st.session_state.xgb = None
    st.session_state.le = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.y_pred_rf = None
    st.session_state.y_pred_xgb = None
    st.session_state.features = None

# Load and preprocess dataset from repository
if st.session_state.data is None:
    try:
        data = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')
        # Data Cleaning
        data = data.iloc[:, :-2]
        data.replace(-200, np.nan, inplace=True)
        data = data.dropna()
        data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')
        
        # Create Target Variable
        def categorize_quality(value):
            if value < 1000:
                return 'Low'
            elif value < 1500:
                return 'Medium'
            else:
                return 'High'
        data['Quality_level'] = data['PT08.S1(CO)'].apply(categorize_quality)
        
        # Feature Engineering
        data['Hour'] = data['Datetime'].dt.hour
        data['Pollutant_Ratio'] = data['CO(GT)'] / data['NO2(GT)'].replace(0, np.nan)
        
        # Store processed data
        st.session_state.data = data
        st.sidebar.success("Dataset loaded successfully!")
    except FileNotFoundError:
        st.error("Error: 'AirQualityUCI.csv' not found in the repository. Please ensure the file is included in the same directory as streamlit_app.py.")
        st.stop()

# Home Page
if page == "Home":
    st.title("🌬️ Air Quality Prediction Dashboard")
    st.markdown("""
        Welcome to the **Air Quality Prediction Dashboard**! This application predicts air quality levels (Low, Medium, High) 
        using machine learning models trained on the UCI Air Quality dataset. Navigate through the sections to explore the data, 
        view exploratory data analysis, evaluate model performance, and make predictions.
        
        **Features:**
        - 📊 Interactive visualizations of data and model results
        - 🤖 Random Forest and XGBoost models for accurate predictions
        - 🕹️ User-friendly interface for custom predictions
        - 📈 Detailed statistical outputs
        
        **Instructions:**
        1. Use the navigation menu to explore different sections.
        2. Check the 'Predict' page to input custom values and get air quality predictions.
    """)
    st.image("https://via.placeholder.com/800x200.png?text=Air+Quality+Banner", use_column_width=True)

# Data Overview Page
elif page == "Data Overview":
    st.title("📊 Data Overview")
    if st.session_state.data is not None:
        st.subheader("Dataset Information")
        buffer = io.StringIO()
        st.session_state.data.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.subheader("First 5 Rows")
        st.dataframe(st.session_state.data.head())
        
        st.subheader("Summary Statistics")
        st.dataframe(st.session_state.data.describe())
    else:
        st.error("Dataset could not be loaded. Please check the error message in the sidebar.")

# EDA Page
elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    if st.session_state.data is not None:
        data = st.session_state.data
        
        st.subheader("Distribution of PT08.S1(CO)")
        fig = px.histogram(data, x='PT08.S1(CO)', nbins=30, marginal="rug", title="Distribution of PT08.S1(CO)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Distribution of Quality Levels")
        quality_counts = data['Quality_level'].value_counts().reindex(['Low', 'Medium', 'High'])
        fig = px.bar(x=quality_counts.index, y=quality_counts.values, labels={'x': 'Quality Level', 'y': 'Count'},
                     title="Distribution of Quality Levels", color=quality_counts.index)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlation Heatmap")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr = data[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu',
                                        zmin=-1, zmax=1, text=corr.values.round(2), texttemplate="%{text}"))
        fig.update_layout(title="Correlation Heatmap of Numeric Features", width=len(numeric_cols)*50 + 100)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Dataset could not be loaded. Please check the error message in the sidebar.")

# Model Results Page
elif page == "Model Results":
    st.title("📈 Model Results")
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Train models if not already trained
        if not st.session_state.model_trained:
            features = ['CO(GT)', 'NO2(GT)', 'T', 'RH', 'Hour', 'Pollutant_Ratio']
            X = data[features]
            y = data['Quality_level']
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
            )
            
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            
            xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
            xgb.fit(X_train, y_train)
            y_pred_xgb = xgb.predict(X_test)
            
            # Store results in session state
            st.session_state.rf = rf
            st.session_state.xgb = xgb
            st.session_state.le = le
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred_rf = y_pred_rf
            st.session_state.y_pred_xgb = y_pred_xgb
            st.session_state.features = features
            st.session_state.model_trained = True
        
        # Display model performance
        st.subheader("Random Forest Performance")
        st.write("**Accuracy:**", accuracy_score(st.session_state.y_test, st.session_state.y_pred_rf))
        st.text("Classification Report:")
        st.text(classification_report(st.session_state.y_test, st.session_state.y_pred_rf, 
                                     target_names=st.session_state.le.classes_))
        
        st.subheader("XGBoost Performance")
        st.write("**Accuracy:**", accuracy_score(st.session_state.y_test, st.session_state.y_pred_xgb))
        st.text("Classification Report:")
        st.text(classification_report(st.session_state.y_test, st.session_state.y_pred_xgb, 
                                     target_names=st.session_state.le.classes_))
        
        # Confusion Matrix for XGBoost
        st.subheader("Confusion Matrix (XGBoost)")
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred_xgb)
        fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=st.session_state.le.classes_, y=st.session_state.le.classes_,
                        title="Confusion Matrix (XGBoost)", color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance for XGBoost
        st.subheader("Feature Importances (XGBoost)")
        importances = st.session_state.xgb.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = [st.session_state.features[i] for i in indices]
        fig = px.bar(x=feature_names, y=importances[indices], title="Feature Importances (XGBoost)",
                     labels={'x': 'Feature', 'y': 'Importance'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Model Accuracy Comparison
        st.subheader("Model Accuracy Comparison")
        model_comparison = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost'],
            'Accuracy': [accuracy_score(st.session_state.y_test, st.session_state.y_pred_rf),
                         accuracy_score(st.session_state.y_test, st.session_state.y_pred_xgb)]
        })
        fig = px.bar(model_comparison, x='Model', y='Accuracy', title="Model Accuracy Comparison",
                     color='Model', text='Accuracy', text_auto='.3f')
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Dataset could not be loaded. Please check the error message in the sidebar.")

# Predict Page
elif page == "Predict":
    st.title("🕹️ Predict Air Quality")
    if st.session_state.data is not None and st.session_state.model_trained:
        st.subheader("Input Parameters for Prediction")
        
        # Create input sliders for features, organized in two columns
        col1, col2 = st.columns(2)
        with col1:
            co_gt = st.slider("CO(GT) (mg/m³)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
            no2_gt = st.slider("NO2(GT) (µg/m³)", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
            temperature = st.slider("Temperature (°C)", min_value=-10.0, max_value=40.0, value=20.0, step=0.1)
        with col2:
            humidity = st.slider("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
            hour = st.slider("Hour of Day", min_value=0, max_value=23, value=14, step=1)
            pollutant_ratio = co_gt / no2_gt if no2_gt != 0 else 0.0
            st.write("**Pollutant Ratio (CO/NO2):**", f"{pollutant_ratio:.3f}")
        
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'CO(GT)': [co_gt],
            'NO2(GT)': [no2_gt],
            'T': [temperature],
            'RH': [humidity],
            'Hour': [hour],
            'Pollutant_Ratio': [pollutant_ratio]
        })
        
        # Make predictions when button is clicked
        if st.button("Predict Air Quality"):
            rf_pred = st.session_state.rf.predict(input_data)
            xgb_pred = st.session_state.xgb.predict(input_data)
            
            rf_quality = st.session_state.le.inverse_transform(rf_pred)[0]
            xgb_quality = st.session_state.le.inverse_transform(xgb_pred)[0]
            
            # Display prediction results
            st.subheader("Prediction Results")
            st.markdown(f"**Random Forest Prediction:** {rf_quality}")
            st.markdown(f"**XGBoost Prediction:** {xgb_quality}")
            
            # Display prediction confidence (probabilities)
            rf_probs = st.session_state.rf.predict_proba(input_data)[0]
            xgb_probs = st.session_state.xgb.predict_proba(input_data)[0]
            
            prob_df = pd.DataFrame({
                'Quality Level': st.session_state.le.classes_,
                'Random Forest Probability': rf_probs,
                'XGBoost Probability': xgb_probs
            })
            
            st.subheader("Prediction Probabilities")
            fig = px.bar(prob_df, x='Quality Level', y=['Random Forest Probability', 'XGBoost Probability'],
                         barmode='group', title="Prediction Probabilities",
                         labels={'value': 'Probability', 'variable': 'Model'})
            fig.update_traces(texttemplate='%{y:.2f}', textposition='auto')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Dataset could not be loaded or models are not trained. Please check the error message in the sidebar.")
