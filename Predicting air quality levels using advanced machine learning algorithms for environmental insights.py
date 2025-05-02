# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import iqr
import warnings
warnings.filterwarnings('ignore')

# 1. Data Collection
def load_data():
    # Assuming CSV file from Kaggle/UCI
    # Replace with actual dataset path
    df = pd.read_csv('air_quality_data.csv')  # Update with actual file path
    return df

# 2. Data Preprocessing
def preprocess_data(df):
    # Handle missing values
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Remove duplicates based on timestamp and sensor_id
    df.drop_duplicates(subset=['timestamp', 'sensor_id'], keep='first', inplace=True)
    
    # Handle outliers using IQR (Winsorization)
    for col in ['PM2.5', 'NO2', 'CO', 'temperature', 'humidity']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # One-hot encoding for categorical variables
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    weather_encoded = encoder.fit_transform(df[['weather_condition']])
    weather_columns = encoder.get_feature_names_out(['weather_condition'])
    df[weather_columns] = weather_encoded
    df.drop('weather_condition', axis=1, inplace=True)
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_cols = ['PM2.5', 'NO2', 'CO', 'temperature', 'humidity']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

# 3. Exploratory Data Analysis
def perform_eda(df):
    # Univariate Analysis
    plt.figure(figsize=(12, 6))
    for col in ['PM2.5', 'NO2', 'CO']:
        plt.subplot(1, 3, ['PM2.5', 'NO2', 'CO'].index(col) + 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('univariate_distributions.png')
    plt.close()
    
    # Bivariate Analysis
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # Scatter plot
    fig = px.scatter(df, x='PM2.5', y='AQI', color='AQI_category')
    fig.write_layout(title='PM2.5 vs AQI')
    fig.write('scatter_plot.html')

# 4. Feature Engineering
def feature_engineering(df):
    # Create average pollutant index
    df['avg_pollutant_index'] = df[['PM2.5', 'NO2', 'CO']].mean(axis=1)
    
    # Extract temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    
    # Bin pollutants into risk levels
    df['PM2.5_risk'] = pd.cut(df['PM2.5'], bins=[0, 0.3, 0.6, 1.0], 
                             labels=['Low', 'Moderate', 'High'])
    
    return df

# 5. Model Building
def build_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'model': model
        }
        
        # Visualizations
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            feature_imp = pd.Series(model.feature_importances_, index=X_train.columns)
            feature_imp.nlargest(10).plot(kind='barh')
            plt.title(f'Feature Importance - {name}')
            plt.savefig(f'feature_importance_{name.lower().replace(" ", "_")}.png')
            plt.close()
    
    # Model comparison
    metrics_df = pd.DataFrame({
        'Model': results.keys(),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'F1-Score': [results[model]['f1_score'] for model in results]
    })
    
    fig = px.bar(metrics_df, x='Model', y=['Accuracy', 'F1-Score'], barmode='group')
    fig.update_layout(title='Model Performance Comparison')
    fig.write('model_comparison.html')
    
    return results

# Main execution
def main():
    # Load data
    df = load_data()
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Perform EDA
    perform_eda(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Prepare features and target
    features = ['PM2.5', 'NO2', 'CO', 'temperature', 'humidity', 
                'avg_pollutant_index', 'hour', 'day', 'month'] + \
               [col for col in df.columns if 'weather_condition' in col]
    X = df[features]
    y = df['AQI_category']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        stratify=y, 
                                                        random_state=42)
    
    # Build and evaluate models
    results = build_models(X_train, X_test, y_train, y_test)
    
    # Print results
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")

if _name_ == "_main_":
    main()
