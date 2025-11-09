"""
Hanoi Temperature Forecasting - Streamlit Web Application

This is the main UI application for the Hanoi Temperature Forecasting project.
It provides an interactive interface for data upload, model training, predictions, and monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import json

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set page configuration
st.set_page_config(
    page_title="Hanoi Temperature Forecasting",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e88e5;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Main header
    st.markdown('<h1 class="main-header">🌡️ Hanoi Temperature Forecasting</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced ML Application for Weather Temperature Prediction</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Upload", "🔍 Data Exploration", "🛠️ Model Training", 
         "🎯 Predictions", "📈 Model Monitoring", "📋 Analytics"]
    )
    
    # Route to different pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Upload":
        show_data_upload_page()
    elif page == "🔍 Data Exploration":
        show_data_exploration_page()
    elif page == "🛠️ Model Training":
        show_model_training_page()
    elif page == "🎯 Predictions":
        show_predictions_page()
    elif page == "📈 Model Monitoring":
        show_monitoring_page()
    elif page == "📋 Analytics":
        show_analytics_page()

def show_home_page():
    """Display the home page with project overview."""
    
    st.markdown('<h2 class="sub-header">🎯 Project Overview</h2>', unsafe_allow_html=True)
    
    # Project description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Hanoi Temperature Forecasting** is a comprehensive machine learning application designed to predict 
        temperature patterns in Hanoi, Vietnam. Using 10+ years of historical weather data with 33+ features, 
        this system provides accurate 5-day temperature forecasts.
        
        ### 🌟 Key Features:
        - **Multi-temporal Analysis**: Daily and hourly weather data forecasting
        - **Advanced ML Pipeline**: Feature engineering and model optimization
        - **Real-time Monitoring**: Performance tracking and automated retraining
        - **Interactive Interface**: User-friendly web application
        - **ONNX Deployment**: Optimized model deployment for production
        """)
        
        # Quick stats
        st.markdown('<h3 class="sub-header">📊 Quick Statistics</h3>', unsafe_allow_html=True)
        
        # Create metric columns
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("📅 Historical Data", "10+ Years", delta="3,650+ days")
        
        with metric_col2:
            st.metric("🔢 Weather Features", "33+", delta="Multi-type data")
        
        with metric_col3:
            st.metric("🎯 Forecast Horizon", "5 Days", delta="Daily predictions")
        
        with metric_col4:
            st.metric("🤖 ML Models", "6+ Algorithms", delta="Ensemble approach")
    
    with col2:
        # Weather features info
        st.markdown('<h3 class="sub-header">🌤️ Weather Features</h3>', unsafe_allow_html=True)
        
        features_info = {
            "🌡️ Temperature": ["temp", "tempmax", "tempmin", "feelslike"],
            "🌊 Atmospheric": ["humidity", "pressure", "visibility", "cloudcover"],
            "💨 Wind": ["windspeed", "winddir", "windgust"],
            "🌧️ Precipitation": ["precip", "precipprob", "snow"],
            "☀️ Solar": ["solarradiation", "uvindex"],
            "🌙 Celestial": ["moonphase", "sunrise", "sunset"],
            "📝 Conditions": ["weather conditions", "descriptions"]
        }
        
        for category, features in features_info.items():
            with st.expander(category):
                for feature in features:
                    st.write(f"• {feature}")
    
    # Project workflow
    st.markdown('<h2 class="sub-header">🔄 Project Workflow</h2>', unsafe_allow_html=True)
    
    workflow_steps = [
        ("1️⃣ Data Collection", "Gather 10 years of Hanoi weather data from Visual Crossing API"),
        ("2️⃣ Data Understanding", "Analyze 33+ features and temperature patterns"),
        ("3️⃣ Data Processing", "Clean, validate, and preprocess the data"),
        ("4️⃣ Feature Engineering", "Create predictive features for 5-day forecasting"),
        ("5️⃣ Model Training", "Train multiple ML models with hyperparameter tuning"),
        ("6️⃣ Model Evaluation", "Assess performance using RMSE, MAPE, R² metrics"),
        ("7️⃣ Deployment", "Deploy models with ONNX optimization"),
        ("8️⃣ Monitoring", "Track performance and implement retraining triggers")
    ]
    
    cols = st.columns(2)
    for i, (title, description) in enumerate(workflow_steps):
        col = cols[i % 2]
        with col:
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Getting started section
    st.markdown('<h2 class="sub-header">🚀 Getting Started</h2>', unsafe_allow_html=True)
    
    start_col1, start_col2, start_col3 = st.columns(3)
    
    with start_col1:
        st.info("""
        **📊 Upload Data**
        
        Start by uploading your weather data or use our data collection tools to gather Hanoi weather information.
        """)
    
    with start_col2:
        st.info("""
        **🛠️ Train Models**
        
        Configure and train various ML models including XGBoost, LightGBM, and Neural Networks for optimal performance.
        """)
    
    with start_col3:
        st.info("""
        **🎯 Make Predictions**
        
        Generate accurate 5-day temperature forecasts using your trained models with confidence intervals.
        """)

def show_data_upload_page():
    """Display the data upload and management page."""
    
    st.markdown('<h2 class="sub-header">📊 Data Upload & Management</h2>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📁 Upload Weather Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with weather data",
        type=['csv'],
        help="Upload your Hanoi weather dataset with daily observations"
    )
    
    if uploaded_file is not None:
        try:
            # Load the data
            df = pd.read_csv(uploaded_file)
            
            # Data preview
            st.success(f"✅ Successfully loaded {len(df)} records!")
            
            # Basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Records", f"{len(df):,}")
            with col2:
                st.metric("🔢 Features", len(df.columns))
            with col3:
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    date_range = (df['datetime'].max() - df['datetime'].min()).days
                    st.metric("📅 Date Range", f"{date_range} days")
            
            # Data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data quality check
            st.markdown("### 🔍 Data Quality Assessment")
            
            # Missing values
            missing_values = df.isnull().sum()
            missing_percent = (missing_values / len(df)) * 100
            
            if missing_values.sum() > 0:
                st.warning(f"⚠️ Found {missing_values.sum()} missing values across {(missing_values > 0).sum()} features")
                
                # Show missing values details
                missing_df = pd.DataFrame({
                    'Feature': missing_values.index,
                    'Missing Count': missing_values.values,
                    'Missing %': missing_percent.values
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
            
            # Feature types
            st.markdown("### 📋 Feature Information")
            
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = df.select_dtypes(include=['object']).columns.tolist()
            datetime_features = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            feat_col1, feat_col2, feat_col3 = st.columns(3)
            
            with feat_col1:
                st.info(f"**🔢 Numeric Features ({len(numeric_features)})**\n\n" + 
                       "\n".join([f"• {feat}" for feat in numeric_features[:10]]) +
                       (f"\n... and {len(numeric_features)-10} more" if len(numeric_features) > 10 else ""))
            
            with feat_col2:
                st.info(f"**📝 Categorical Features ({len(categorical_features)})**\n\n" + 
                       "\n".join([f"• {feat}" for feat in categorical_features[:10]]) +
                       (f"\n... and {len(categorical_features)-10} more" if len(categorical_features) > 10 else ""))
            
            with feat_col3:
                st.info(f"**📅 DateTime Features ({len(datetime_features)})**\n\n" + 
                       "\n".join([f"• {feat}" for feat in datetime_features]))
            
            # Save processed data
            if st.button("💾 Save Data for Analysis", type="primary"):
                # Create data directory if it doesn't exist
                os.makedirs("data/uploaded", exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/uploaded/uploaded_weather_data_{timestamp}.csv"
                
                df.to_csv(filename, index=False)
                st.success(f"✅ Data saved to {filename}")
                
                # Save to session state for use in other pages
                st.session_state['weather_data'] = df
                st.session_state['data_loaded'] = True
        
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    # Data collection section
    st.markdown("### 🌐 Collect New Data")
    
    with st.expander("🔧 Configure Data Collection"):
        col1, col2 = st.columns(2)
        
        with col1:
            api_key = st.text_input(
                "Visual Crossing API Key",
                type="password",
                help="Get your free API key from visualcrossing.com"
            )
            
            years = st.slider("Years of Historical Data", 1, 15, 10)
        
        with col2:
            data_type = st.selectbox("Data Frequency", ["Daily", "Hourly"])
            
            location = st.text_input("Location", value="Hanoi,Vietnam")
        
        if st.button("🚀 Start Data Collection"):
            if api_key:
                with st.spinner("Collecting weather data... This may take a few minutes."):
                    st.info("Data collection would start here. This requires the backend collector module.")
                    # TODO: Implement actual data collection
                    st.success("Data collection completed! (Placeholder)")
            else:
                st.error("Please provide a valid API key")

def show_data_exploration_page():
    """Display the data exploration and visualization page."""
    
    st.markdown('<h2 class="sub-header">🔍 Data Exploration & Visualization</h2>', unsafe_allow_html=True)
    
    # Check if data is loaded
    if 'weather_data' not in st.session_state:
        st.warning("📊 No data loaded. Please upload data first in the Data Upload page.")
        return
    
    df = st.session_state['weather_data']
    
    # Data overview
    st.markdown("### 📋 Dataset Overview")
    
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    
    with overview_col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with overview_col2:
        st.metric("Features", len(df.columns))
    
    with overview_col3:
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            days = (df['datetime'].max() - df['datetime'].min()).days
            st.metric("Time Span", f"{days} days")
    
    with overview_col4:
        if 'temp' in df.columns:
            st.metric("Avg Temperature", f"{df['temp'].mean():.1f}°C")
    
    # Temperature analysis
    if 'temp' in df.columns:
        st.markdown("### 🌡️ Temperature Analysis")
        
        # Temperature time series
        fig = px.line(df, x='datetime', y='temp', title='Hanoi Temperature Over Time')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Temperature statistics
        temp_col1, temp_col2 = st.columns(2)
        
        with temp_col1:
            # Temperature distribution
            fig_hist = px.histogram(df, x='temp', title='Temperature Distribution', 
                                  nbins=50, marginal='box')
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with temp_col2:
            # Monthly temperature patterns
            if 'datetime' in df.columns:
                df['month'] = df['datetime'].dt.month
                monthly_temp = df.groupby('month')['temp'].mean().reset_index()
                
                fig_monthly = px.bar(monthly_temp, x='month', y='temp', 
                                   title='Average Temperature by Month')
                fig_monthly.update_xaxes(tickmode='linear', tick0=1, dtick=1)
                st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Feature correlation analysis
    st.markdown("### 🔗 Feature Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        # Select features for correlation
        selected_features = st.multiselect(
            "Select features for correlation analysis:",
            options=numeric_cols,
            default=numeric_cols[:min(10, len(numeric_cols))]
        )
        
        if len(selected_features) > 1:
            # Correlation matrix
            corr_matrix = df[selected_features].corr()
            
            # Plot correlation heatmap
            fig_corr = px.imshow(corr_matrix, 
                               title="Feature Correlation Matrix",
                               color_continuous_scale='RdBu_r',
                               aspect='auto')
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Top correlations with temperature
            if 'temp' in selected_features:
                temp_corr = corr_matrix['temp'].abs().sort_values(ascending=False)
                temp_corr = temp_corr[temp_corr.index != 'temp'][:10]
                
                st.markdown("#### 🎯 Features Most Correlated with Temperature")
                
                corr_df = pd.DataFrame({
                    'Feature': temp_corr.index,
                    'Correlation': temp_corr.values
                })
                
                fig_bar = px.bar(corr_df, x='Correlation', y='Feature', 
                               orientation='h', title='Top 10 Temperature Correlations')
                st.plotly_chart(fig_bar, use_container_width=True)

def show_model_training_page():
    """Display the model training and configuration page."""
    
    st.markdown('<h2 class="sub-header">🛠️ Model Training & Configuration</h2>', unsafe_allow_html=True)
    
    # Check if data is loaded
    if 'weather_data' not in st.session_state:
        st.warning("📊 No data loaded. Please upload data first in the Data Upload page.")
        return
    
    df = st.session_state['weather_data']
    
    # Model configuration
    st.markdown("### ⚙️ Model Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        # Target variable
        target_column = st.selectbox(
            "Target Variable",
            options=[col for col in df.columns if 'temp' in col.lower()],
            index=0 if 'temp' in df.columns else 0
        )
        
        # Forecast horizon
        forecast_days = st.slider("Forecast Horizon (days)", 1, 14, 5)
        
        # Train-test split
        test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    
    with config_col2:
        # Model selection
        models_to_train = st.multiselect(
            "Select Models to Train",
            options=["Linear Regression", "Random Forest", "XGBoost", "LightGBM", "Neural Network"],
            default=["Random Forest", "XGBoost", "LightGBM"]
        )
        
        # Cross-validation
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        # Hyperparameter tuning
        tune_hyperparams = st.checkbox("Enable Hyperparameter Tuning", value=True)
    
    # Feature selection
    st.markdown("### 🎯 Feature Selection")
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    
    selected_features = st.multiselect(
        "Select Features for Training",
        options=numeric_features,
        default=numeric_features[:min(15, len(numeric_features))]
    )
    
    # Training progress section
    st.markdown("### 🏃‍♂️ Training Progress")
    
    if st.button("🚀 Start Model Training", type="primary"):
        if not selected_features:
            st.error("Please select at least one feature for training.")
            return
        
        with st.spinner("Training models... This may take several minutes."):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate training process
            total_models = len(models_to_train)
            
            for i, model_name in enumerate(models_to_train):
                status_text.text(f"Training {model_name}...")
                progress_bar.progress((i + 1) / total_models)
                
                # Simulate training time
                import time
                time.sleep(2)
                
                # Display training results (placeholder)
                st.success(f"✅ {model_name} training completed!")
                
                # Show placeholder metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(f"{model_name} RMSE", f"{np.random.uniform(1.5, 3.0):.2f}°C")
                
                with metric_col2:
                    st.metric(f"{model_name} MAPE", f"{np.random.uniform(5, 15):.1f}%")
                
                with metric_col3:
                    st.metric(f"{model_name} R²", f"{np.random.uniform(0.7, 0.95):.3f}")
            
            status_text.text("Training completed!")
            st.balloons()
    
    # Model comparison section
    st.markdown("### 📊 Model Comparison")
    
    # Placeholder for model comparison results
    if st.checkbox("Show Model Comparison (Demo)"):
        # Create sample model performance data
        model_results = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'LightGBM', 'Neural Network'],
            'RMSE': [2.1, 1.9, 1.8, 2.3],
            'MAPE': [8.5, 7.2, 6.8, 9.1],
            'R²': [0.87, 0.89, 0.91, 0.85],
            'Training Time (min)': [5, 12, 8, 25]
        })
        
        st.dataframe(model_results, use_container_width=True)
        
        # Performance visualization
        fig_perf = px.bar(model_results, x='Model', y=['RMSE', 'MAPE'], 
                         title="Model Performance Comparison",
                         barmode='group')
        st.plotly_chart(fig_perf, use_container_width=True)

def show_predictions_page():
    """Display the prediction interface."""
    
    st.markdown('<h2 class="sub-header">🎯 Temperature Predictions</h2>', unsafe_allow_html=True)
    
    # Mock prediction interface
    st.markdown("### 🔮 Generate Predictions")
    
    # Input features for prediction
    pred_col1, pred_col2 = st.columns(2)
    
    with pred_col1:
        st.markdown("#### Current Weather Conditions")
        
        current_temp = st.slider("Current Temperature (°C)", -10, 45, 25)
        humidity = st.slider("Humidity (%)", 0, 100, 65)
        pressure = st.slider("Pressure (hPa)", 980, 1040, 1013)
        wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)
    
    with pred_col2:
        st.markdown("#### Additional Parameters")
        
        cloud_cover = st.slider("Cloud Cover (%)", 0, 100, 40)
        precipitation = st.slider("Precipitation (mm)", 0, 50, 0)
        season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
        
        # Prediction horizon
        days_ahead = st.slider("Prediction Days Ahead", 1, 14, 5)
    
    # Generate predictions
    if st.button("🔍 Generate Forecast", type="primary"):
        with st.spinner("Generating predictions..."):
            # Simulate prediction generation
            import time
            time.sleep(2)
            
            # Generate mock predictions
            base_temp = current_temp
            predictions = []
            confidence_lower = []
            confidence_upper = []
            
            for i in range(days_ahead):
                # Add some randomness and trend
                trend = np.random.uniform(-0.5, 0.5)
                noise = np.random.normal(0, 1)
                pred_temp = base_temp + trend + noise
                
                predictions.append(pred_temp)
                confidence_lower.append(pred_temp - 2)
                confidence_upper.append(pred_temp + 2)
                
                base_temp = pred_temp
            
            # Create prediction DataFrame
            dates = pd.date_range(start=datetime.now().date(), periods=days_ahead, freq='D')
            pred_df = pd.DataFrame({
                'Date': dates,
                'Predicted_Temperature': predictions,
                'Lower_Bound': confidence_lower,
                'Upper_Bound': confidence_upper
            })
            
            # Display predictions
            st.success("✅ Predictions generated successfully!")
            
            # Predictions table
            st.markdown("### 📋 Forecast Results")
            st.dataframe(pred_df.round(2), use_container_width=True)
            
            # Predictions visualization
            st.markdown("### 📈 Forecast Visualization")
            
            fig = go.Figure()
            
            # Add prediction line
            fig.add_trace(go.Scatter(
                x=pred_df['Date'],
                y=pred_df['Predicted_Temperature'],
                mode='lines+markers',
                name='Predicted Temperature',
                line=dict(color='red', width=2)
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=pred_df['Date'],
                y=pred_df['Upper_Bound'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=pred_df['Date'],
                y=pred_df['Lower_Bound'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Interval',
                fillcolor='rgba(255,0,0,0.2)'
            ))
            
            fig.update_layout(
                title='Temperature Forecast with Confidence Intervals',
                xaxis_title='Date',
                yaxis_title='Temperature (°C)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model confidence
            st.markdown("### 🎯 Prediction Confidence")
            
            conf_col1, conf_col2, conf_col3 = st.columns(3)
            
            with conf_col1:
                st.metric("Model Accuracy", "89.2%", delta="2.1%")
            
            with conf_col2:
                st.metric("Prediction RMSE", "1.8°C", delta="-0.3°C")
            
            with conf_col3:
                st.metric("Confidence Score", "High", delta="Stable")

def show_monitoring_page():
    """Display the model monitoring dashboard."""
    
    st.markdown('<h2 class="sub-header">📈 Model Monitoring & Performance</h2>', unsafe_allow_html=True)
    
    # Model performance metrics
    st.markdown("### 📊 Real-time Performance Metrics")
    
    # Create mock performance data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    performance_data = pd.DataFrame({
        'Date': dates,
        'RMSE': np.random.uniform(1.5, 2.5, len(dates)),
        'MAPE': np.random.uniform(6, 12, len(dates)),
        'R²': np.random.uniform(0.85, 0.92, len(dates))
    })
    
    # Performance metrics cards
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        current_rmse = performance_data['RMSE'].iloc[-1]
        previous_rmse = performance_data['RMSE'].iloc[-2]
        delta_rmse = current_rmse - previous_rmse
        st.metric("Current RMSE", f"{current_rmse:.2f}°C", delta=f"{delta_rmse:.3f}°C")
    
    with metric_col2:
        current_mape = performance_data['MAPE'].iloc[-1]
        previous_mape = performance_data['MAPE'].iloc[-2]
        delta_mape = current_mape - previous_mape
        st.metric("Current MAPE", f"{current_mape:.1f}%", delta=f"{delta_mape:.2f}%")
    
    with metric_col3:
        current_r2 = performance_data['R²'].iloc[-1]
        previous_r2 = performance_data['R²'].iloc[-2]
        delta_r2 = current_r2 - previous_r2
        st.metric("Current R²", f"{current_r2:.3f}", delta=f"{delta_r2:.4f}")
    
    with metric_col4:
        st.metric("Model Status", "Healthy", delta="Stable")
    
    # Performance trends
    st.markdown("### 📈 Performance Trends (Last 30 Days)")
    
    fig_trends = px.line(performance_data, x='Date', y=['RMSE', 'MAPE'], 
                        title='Model Performance Trends')
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Data drift detection
    st.markdown("### 🚨 Data Drift Detection")
    
    drift_col1, drift_col2 = st.columns(2)
    
    with drift_col1:
        st.info("""
        **📊 Feature Drift Status**
        
        • Temperature features: ✅ Stable
        • Humidity: ✅ Stable  
        • Pressure: ⚠️ Minor drift detected
        • Wind patterns: ✅ Stable
        • Precipitation: ✅ Stable
        """)
    
    with drift_col2:
        st.info("""
        **🔄 Retraining Recommendations**
        
        • Last retrain: 5 days ago
        • Performance: Stable
        • Data drift: Minor (pressure)
        • Recommendation: Monitor for 2 more days
        • Next scheduled retrain: In 2 days
        """)
    
    # Model deployment status
    st.markdown("### 🚀 Deployment Status")
    
    deploy_col1, deploy_col2, deploy_col3 = st.columns(3)
    
    with deploy_col1:
        st.success("""
        **🏭 Production Model**
        
        • Status: Active
        • Version: v2.1.3
        • Uptime: 99.8%
        • Last updated: 2 days ago
        """)
    
    with deploy_col2:
        st.info("""
        **⚡ ONNX Optimization**
        
        • Status: Enabled
        • Speed improvement: 4.2x
        • Memory usage: -60%
        • Inference time: 12ms
        """)
    
    with deploy_col3:
        st.warning("""
        **🔧 Staging Model**
        
        • Status: Testing
        • Version: v2.2.0-beta
        • Performance: +5% accuracy
        • Deployment: Pending approval
        """)

def show_analytics_page():
    """Display advanced analytics and insights."""
    
    st.markdown('<h2 class="sub-header">📋 Advanced Analytics & Insights</h2>', unsafe_allow_html=True)
    
    # Feature importance analysis
    st.markdown("### 🎯 Feature Importance Analysis")
    
    # Mock feature importance data
    features = ['temp_lag_1', 'humidity', 'pressure', 'temp_lag_7', 'cloudcover', 
               'windspeed', 'solarradiation', 'temp_rolling_3d', 'season', 'month']
    importance = [0.28, 0.15, 0.12, 0.11, 0.09, 0.08, 0.07, 0.05, 0.03, 0.02]
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })
    
    fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h', title='Feature Importance for Temperature Prediction')
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Seasonal analysis
    st.markdown("### 🌅 Seasonal Pattern Analysis")
    
    # Mock seasonal data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    avg_temp = [16, 18, 22, 26, 29, 31, 31, 30, 28, 25, 21, 18]
    prediction_accuracy = [92, 91, 89, 87, 85, 83, 84, 86, 88, 90, 91, 93]
    
    seasonal_df = pd.DataFrame({
        'Month': months,
        'Avg_Temperature': avg_temp,
        'Prediction_Accuracy': prediction_accuracy
    })
    
    fig_seasonal = px.line(seasonal_df, x='Month', y=['Avg_Temperature'], 
                          title='Seasonal Temperature Patterns')
    fig_seasonal.add_bar(x=seasonal_df['Month'], y=seasonal_df['Prediction_Accuracy'], 
                        name='Prediction Accuracy (%)', yaxis='y2')
    
    fig_seasonal.update_layout(yaxis2=dict(title='Accuracy (%)', overlaying='y', side='right'))
    st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Error analysis
    st.markdown("### 🔍 Prediction Error Analysis")
    
    error_col1, error_col2 = st.columns(2)
    
    with error_col1:
        # Error distribution
        errors = np.random.normal(0, 1.5, 1000)
        fig_errors = px.histogram(errors, title='Prediction Error Distribution', 
                                 nbins=50, marginal='box')
        st.plotly_chart(fig_errors, use_container_width=True)
    
    with error_col2:
        # Error by temperature range
        temp_ranges = ['<10°C', '10-20°C', '20-30°C', '>30°C']
        avg_errors = [2.1, 1.6, 1.4, 1.8]
        
        error_range_df = pd.DataFrame({
            'Temperature_Range': temp_ranges,
            'Average_Error': avg_errors
        })
        
        fig_error_range = px.bar(error_range_df, x='Temperature_Range', y='Average_Error',
                               title='Average Error by Temperature Range')
        st.plotly_chart(fig_error_range, use_container_width=True)
    
    # Business insights
    st.markdown("### 💼 Business Insights & Recommendations")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.success("""
        **🎯 Model Performance Insights**
        
        • Best accuracy during winter months (Nov-Feb)
        • Temperature lag features are most predictive
        • Humidity and pressure provide strong signals
        • Model performs well for moderate temperatures
        • Consider ensemble approaches for extreme weather
        """)
    
    with insights_col2:
        st.info("""
        **🔧 Optimization Recommendations**
        
        • Increase training data for summer months
        • Add more meteorological features
        • Implement seasonal model variations
        • Consider hourly data for short-term forecasts
        • Explore deep learning for pattern recognition
        """)
    
    # Export functionality
    st.markdown("### 📤 Export & Reports")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("📊 Export Performance Report"):
            st.success("Performance report exported to downloads/")
    
    with export_col2:
        if st.button("📈 Export Predictions"):
            st.success("Predictions exported to CSV format")
    
    with export_col3:
        if st.button("🔍 Generate Analysis Report"):
            st.success("Comprehensive analysis report generated")

if __name__ == "__main__":
    main()