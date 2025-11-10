# src/predictor.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime, timedelta

def load_model_and_scaler(model_path):
    """Load saved model package"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    package = joblib.load(model_path)
    return package['model'], package['scaler'], package['window_size'], package['forecast_days']

def prepare_prediction_features(df, window_size=30):
    """Prepare features for prediction using last 'window_size' days"""
    # Get last window_size days
    recent_data = df.tail(window_size).copy()
    
    if len(recent_data) < window_size:
        raise ValueError(f"Need at least {window_size} days of history, got {len(recent_data)}")
    
    # Apply same feature engineering as training
    recent_data = add_time_features(recent_data)
    recent_data = encode_categorical_features(recent_data)
    
    # Select features (same as training)
    feature_cols = [col for col in recent_data.columns 
                   if col not in ['datetime', 'temp', 'tempmin', 'tempmax', 'conditions']]
    
    # Flatten to match training format
    X = recent_data[feature_cols].values.flatten().reshape(1, -1)
    
    return X, feature_cols

def add_time_features(df):
    """Same as your pipeline"""
    df = df.copy()
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['weekday'] = df['datetime'].dt.weekday
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    df['season'] = df['month'].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 
                                   6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
    return df

def encode_categorical_features(df):
    """Handle categorical features for prediction"""
    df = df.copy()
    categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
    if 'datetime' in categorical_cols:
        categorical_cols.remove('datetime')
    
    # For prediction, we'll use simple encoding (or you can save label encoders from training)
    for col in categorical_cols:
        # Use string hash for consistent encoding
        df[col] = df[col].astype(str).apply(lambda x: hash(x) % 10000)
    
    return df

def predict_next_5_days(df, model_path):
    """Main prediction function for Streamlit"""
    try:
        model, scaler, window_size, forecast_days = load_model_and_scaler(model_path)
        X, _ = prepare_prediction_features(df, window_size)
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)[0]  # Get first (and only) prediction
        
        # Create result DataFrame
        today = datetime.now().date()
        future_dates = [today + timedelta(days=i+1) for i in range(forecast_days)]
        
        result_df = pd.DataFrame({
            'datetime': [pd.Timestamp(d) for d in future_dates],
            'date': future_dates,
            'temp': predictions,
            'tempmin': predictions - 2.0,  # Conservative estimate
            'tempmax': predictions + 2.0,
            'conditions': ['Partly Cloudy'] * forecast_days  # Replace with real condition predictor if available
        })
        
        return result_df
        
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")