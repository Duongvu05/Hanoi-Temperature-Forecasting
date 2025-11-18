# Hanoi Temperature Forecasting Project
## 🌡️ Comprehensive Machine Learning System for Weather Prediction

### *Dự án Dự báo Nhiệt độ Hà Nội*
**Using 10 Years of Weather Data & Advanced CatBoost ML Techniques**

---

## 📊 Project Overview

### Key Highlights
- **Objective**: 5-day temperature forecasting for Hanoi, Vietnam using end-to-end ML pipeline
- **Data Source**: Visual Crossing Weather API (2015-2025) with comprehensive weather variables
- **Model**: CatBoost Gradient Boosting with Optuna hyperparameter optimization
- **Performance**: R² = 0.828472 (82.85% accuracy) across multi-horizon predictions
- **Deployment**: Live Streamlit web application + ONNX production optimization

### Technical Specifications & Scale
- **10 years** of historical weather data (3,653+ daily records)
- **33 comprehensive** weather features from Visual Crossing API
- **87,672+** hourly observations for enhanced granularity
- **Multi-horizon** predictions (T+1 to T+5 days) with performance degradation analysis
- **95 selected features** from 150+ engineered variables after correlation-based selection
- **Production Ready**: 1.6ms inference time with ONNX optimization

---

## Step 1: Data Collection
## 📊 Thu thập Dữ liệu từ Visual Crossing API

### 🎯 **Data Acquisition Strategy**
- **API Source**: Visual Crossing Weather API with comprehensive weather variables
- **Timeline**: 10 years (2015-2025) for Hanoi, Vietnam (UTC+07:00)
- **Location**: Hanoi coordinates with 2m height temperature measurements
- **Granularity**: Both daily and hourly aggregated weather data
- **Validation**: Data quality checks and temporal consistency verification

### 📈 **Comprehensive Dataset Features (33 Variables)**
| **Feature Category** | **Variables** | **Analysis Insights** |
|---------------------|---------------|----------------------|
| **Temperature (4)** | tempmax, tempmin, temp, feelslike | Target: 15-38°C range, seasonal patterns |
| **Atmospheric (4)** | humidity, dew, sealevelpressure, visibility | High correlation with temperature |
| **Solar (3)** | solarradiation, solarenergy, uvindex | Strong temp correlation (r=0.65) |
| **Precipitation (4)** | precip, precipprob, precipcover, preciptype | Sparse data, mostly 0 values |
| **Wind (3)** | windspeed, winddir, windgust | Low signal for temp prediction |
| **Temporal (3)** | datetime, sunrise, sunset | Seasonal variation, day length |
| **Weather (12)** | cloudcover, conditions, moonphase, etc. | Weather system indicators |

### 📊 **Dataset Scale & Quality Analysis**
- **Total Records**: 3,653 daily + 87,672 hourly observations
- **Missing Values**: <5% for any column (excellent data quality)
- **Temperature Range**: 15-42°C with clear 4-season cycles
- **Data Validation**: Zero duplicate records, temporal consistency verified
- **API Limitations**: Free plan limited to 1000 records/day

---

## Step 2: Exploratory Data Analysis
## 🔍 Khám Phá Patterns và Correlations trong Dữ liệu

### 🎯 **Phát Hiện Chính**
- **Seasonal Patterns**: 4 mùa rõ ràng (Hè: 32-38°C, Đông: 16-22°C)
- **Weather Memory**: Autocorrelation mạnh (r=0.87 lag-1 day)
- **Solar Correlation**: Bức xạ mặt trời ảnh hướng nhiệt độ (r=0.65)
- **Feature Redundancy**: `temp` vs `feelslike` (r=0.98) cần xử lý

### 📊 **Statistical Analysis Results**
| **Aspect** | **Finding** | **ML Implication** |
|------------|-------------|---------------------|
| **Temperature Range** | 15-38°C, ổn định 10 năm | Good for forecasting |
| **Missing Values** | <5% mỗi feature | High data quality |
| **Outliers** | Extreme weather events | Keep for robustness |
| **Seasonality** | Strong 365-day cycles | Need cyclical encoding |
| **Persistence** | High day-to-day correlation | Lag features critical |

### � **Top Correlations với Temperature**
1. **feelslike** (r=0.98) - Multicollinearity issue
2. **dew point** (r=0.78) - Humidity relationship  
3. **solarradiation** (r=0.65) - Energy source
4. **humidity** (r=-0.45) - Inverse relationship

---

## Step 3: Data Processing
## 🛠️ Clean and Prepare Data for Machine Learning

### 🔍 **Feature Type Classification & Processing**
- **Numerical (23)**: Temperature metrics, humidity, pressure, wind, solar, precipitation amounts
- **Categorical (4)**: preciptype, conditions, description, icon (encoded/removed)
- **Temporal (3)**: datetime, sunrise, sunset (parsed to datetime objects)
- **Boolean/Binary (3)**: Derived flags for weather conditions

### ⚙️ **Comprehensive Preprocessing Pipeline**
```python
# Data Quality Improvements Implementation
preprocessing_steps = {
    'datetime_conversion': 'Parsed sunrise, sunset to datetime objects',
    'column_removal': 'Dropped icon, stations, conditions, description (low signal)',
    'missing_values': 'SimpleImputer with median (numerical), most_frequent (categorical)',
    'duplicate_detection': '0 duplicate records found',
    'outlier_analysis': 'Temperature extremes identified but retained (valid winter data)'
}
```

### 📊 **Data Processing Results**
| **Processing Step** | **Before** | **After** | **Impact** |
|---------------------|------------|-----------|------------|
| **Feature Count** | 33 raw features | 29 cleaned features | -12% reduction |
| **Missing Values** | <5% per column | 0% complete | ✅ Perfect quality |
| **Data Types** | Mixed types | Standardized | ✅ ML-ready |
| **Memory Usage** | Raw dataset | Optimized format | ✅ Efficient |
| **Correlation Issues** | High multicollinearity | Flagged for feature selection | ✅ Identified |

### 🎯 **Key Processing Achievements**
- **Outlier Analysis**: Rare cold values (<10°C) identified but retained as valid winter data
- **Skewed Distributions**: Most non-temperature variables are non-normal (binary/sparse)
- **Correlation-Based Filtering**: Identified high multicollinearity groups for feature selection
- **Pipeline Architecture**: ColumnTransformer with separate numerical/categorical preprocessing
- **Output Quality**: Preprocessed dataset with 29 cleaned features ready for engineering

---

## 🔧 Complete ML Pipeline Architecture

### 9-Step Implementation Process
1. **Data Collection** → API integration & validation
2. **Exploratory Analysis** → Pattern discovery & correlations
3. **Data Processing** → Cleaning & preprocessing
4. **Feature Engineering** → Temporal & lag features
5. **Model Training** → CatBoost optimization
6. **UI Development** → Streamlit web application
7. **Performance Monitoring** → Automated retraining
8. **Hourly Enhancement** → Extended granularity
9. **ONNX Deployment** → Production optimization

---

## 🛠️ Step 1-3: Data Foundation

### Data Collection & Understanding
- **API Integration**: Visual Crossing Weather API
- **Validation**: Quality checks and temporal consistency
- **Storage**: Organized raw/processed/realtime structure

### Key Discoveries from EDA
- **Seasonal Patterns**: Clear 4-season cycle
  - Summer: 32-38°C (June-August)
  - Winter: 16-22°C (December-February)
- **High Autocorrelation**: Weather "stickiness" effect
- **Feature Redundancy**: Multiple temperature variants

### Data Processing Results
- **Features Removed**: `icon`, `stations`, `conditions` (low signal)
- **Missing Values**: Handled with median/mode imputation
- **Outlier Analysis**: Cold extremes (<10°C) retained as valid winter data

---

---

## Step 4: Feature Engineering
## ⚙️ Transform Raw Data into ML-Ready Features for Temperature Forecasting

### 🎯 **Forecasting Context & Strategy**
- **Objective**: Using daily historical data to predict Hanoi temperature for next 5 days
- **Engineering Pipeline**: Transform 29 cleaned features into 150+ engineered features
- **Selection Process**: Correlation-based selection to identify optimal feature subset
- **Final Feature Set**: 95 selected features achieving best performance balance

### 🕒 **Temporal Features - Seasonal Pattern Encoding**
```python
# Cyclical encoding for seasonal patterns
temporal_features = {
    'day_of_year': 'Linear progression through year (1-365)',
    'month': 'Seasonal indicator with cyclical encoding', 
    'season': 'Categorical season indicators (Spring/Summer/Fall/Winter)',
    'cyclical_encoding': 'sin/cos transformation for seasonal patterns'
}
```

### 📊 **Lag Features - Historical Temperature Values**
- **Lag Periods**: 1-30 day historical temperature values
- **Critical Importance**: Yesterday's temperature (lag_1) contributes 25-30% model importance
- **Extended Context**: Weekly (7-day), bi-weekly (14-day), monthly (30-day) patterns
- **Multi-variable Lags**: Temperature, humidity, solar radiation, pressure historical values

### 🌊 **Rolling Statistics - Moving Patterns**
| **Window Size** | **Metrics** | **Purpose** |
|-----------------|-------------|-------------|
| **3-day** | Moving avg, std dev | Short-term trend detection |
| **7-day** | Moving avg, std dev | Weekly weather stability |
| **14-day** | Moving avg, std dev | Bi-weekly pattern analysis |
| **30-day** | Moving avg, std dev | Monthly climate context |

### 🔬 **Advanced Derived Features**
- **Temperature Differences**: Rate of change, day-to-day variations
- **Weather Stability Indices**: Pressure trends, humidity persistence
- **Text Feature Processing**: Weather conditions encoding (One-hot, Label encoding)
- **NLP Processing**: Weather descriptions transformation to numerical features

---

## Step 5: Model Training & Optimization
## 🤖 CatBoost Multi-Output với 82.85% Accuracy

### 🏆 **Algorithm Comparison Results**
| **Algorithm** | **R² Score** | **MAE (°C)** | **RMSE (°C)** | **Rank** |
|---------------|--------------|--------------|---------------|----------|
| **🥇 CatBoost** | **0.8285** | **1.68** | **2.02** | **Winner** |
| 🥈 Ridge | 0.8109 | 1.69 | 2.21 | -2.1% |
| 🥉 Random Forest | 0.8078 | 1.76 | 2.23 | -2.5% |
| Lasso | 0.8063 | 1.73 | 2.24 | -2.7% |

### ⚙️ **Optimal Hyperparameters (Optuna - 50 trials)**
```python
best_params = {
    'learning_rate': 0.074,     # Stable convergence
    'depth': 7,                 # Complex interactions  
    'iterations': 1498,         # Early stopping at 261
    'l2_leaf_reg': 3.2,        # Regularization
    'loss_function': 'MultiRMSE'  # Multi-output
}
```

### 🎯 **Detailed Multi-Horizon Performance Analysis**
| **Horizon** | **R² Score** | **MAE (°C)** | **RMSE (°C)** | **Performance Quality** |
|-------------|--------------|--------------|---------------|------------------------|
| **T+1 Day** | **0.917359** (91.74%) | **1.14** | **1.46** | 🔥 Excellent |
| **T+2 Days** | **0.847739** (84.77%) | **1.55** | **1.98** | ✅ Very Good |
| **T+3 Days** | **0.812615** (81.26%) | **1.73** | **2.20** | ✅ Good |
| **T+4 Days** | **0.790585** (79.06%) | **1.85** | **2.33** | ✅ Acceptable |
| **T+5 Days** | **0.774062** (77.41%) | **1.92** | **2.42** | ⚠️ Acceptable |

**Performance Degradation**: -15.62% from T+1 to T+5 (natural forecast decay)

---

## Step 6: User Interface Development
## 🌐 Streamlit Interactive Web Application

### 🚀 **Live Production Deployment**
**🌐 [Access Live Demo](https://hanoi-temperature-forecasting.streamlit.app/)**
- **Framework**: Streamlit for interactive web application
- **Deployment**: Local and cloud deployment options (Live on Streamlit Cloud)
- **User Experience**: Intuitive interface for non-technical users
- **Performance**: Real-time predictions with minimal latency

### 🎯 **Comprehensive Application Features**
- **🌡️ Real-time Temperature Prediction**: 5-day forecast interface with confidence intervals
- **📈 Historical Data Visualization**: Interactive charts and time series analysis
- **🎯 Model Performance Metrics**: Live display of R² scores, MAE, RMSE across horizons
- **📈 Interactive Forecasting Results**: Dynamic charts with user-friendly weather displays
- **📱 Mobile Responsive**: Optimized for all device sizes and platforms

### 🛠️ **Production Technical Stack**
```python
# Web Application Framework
streamlit>=1.28.0          # Interactive web framework
plotly>=5.15.0             # Dynamic visualization charts
pandas>=2.0.0              # Data manipulation and analysis

# ML Model Integration
joblib>=1.3.0              # Trained model loading and persistence
catboost>=1.2.0            # CatBoost inference engine
onnxruntime>=1.15.0        # ONNX optimized model deployment
```

---

## Step 7: Model Performance Monitoring
## 📊 Production Monitoring & Automated Retraining System

### 🎯 **MLOps Production Results (30-Day Simulation)**
```python
# Automated retraining system performance
production_results = {
    'monitoring_period': '30 days (Oct 4 - Nov 2, 2023)',
    'training_samples': 2895,
    'automatic_retrains': 5,  # Triggered by performance alerts
    'performance_recovery': 'RMSE: 3.95°C → 1.49°C (T+1)',
    'model_versions': 'v1→v5 with systematic updates'
}
```

### 📈 **Retraining Strategy & Performance**
| **Trigger Condition** | **Threshold** | **Action** | **Results** |
|----------------------|---------------|------------|-------------|
| **T+1 RMSE Spike** | >30% increase | Emergency retrain | 165% spike → baseline |
| **T+2-5 Degradation** | >20% increase | Scheduled retrain | Performance recovered |
| **Seasonal Shifts** | Statistical drift | Model update | Adapted to weather changes |
| **Evaluation Window** | 7-day stability | Prevent overfit | Balanced responsiveness |

### 🔍 **Key Production Insights**
- **Weather Volatility**: October period showed high temperature variability requiring frequent updates
- **Alert System**: Real-time monitoring with 165% RMSE spike detection for critical failures  
- **Model Stability**: 7-day evaluation window prevents over-frequent retraining
- **Performance Patterns**: Systematic -16.37% degradation from T+1 to T+5 requires horizon-specific optimization

### 🚨 **Advanced Monitoring Features**
- **Performance Tracking**: Continuous monitoring of model accuracy over time
- **Degradation Detection**: Statistical methods to identify performance decline
- **Model Versioning**: Systematic model updates and rollback capabilities
- **Zero Downtime**: Seamless version transitions in production environment

---

## Step 8: Hourly Enhancement
## ⏰ Mở Rộng từ Daily → Hourly Forecasting

### 🚀 **System Enhancement Objectives**
- **Temporal Resolution**: Daily (1 pred/day) → Hourly (24 pred/day)
- **Data Volume**: 3,653 daily records → 87,672 hourly records
- **Forecast Granularity**: 5-day horizon → 120-hour horizon
- **Use Cases**: Detailed intraday planning, energy management, agriculture

### ⚡ **Performance Scaling Results**
| **Aspect** | **Daily Model** | **Hourly Model** | **Scale Factor** |
|------------|-----------------|------------------|------------------|
| **Training Time** | 42 seconds | 8.5 minutes | 12.1x |
| **Model Size** | 12.8 MB | 47.3 MB | 3.7x |
| **Inference** | 0.002s | 0.048s | 24x (per batch) |
| **Memory Usage** | 180 MB | 1.2 GB | 6.7x |
| **R² Score** | 82.85% | 79.21% | -4.4% (acceptable) |

### � **Hourly Model Performance**
```python
# Hour-specific performance patterns
peak_performance_hours = {
    'Best (12:00-15:00)': 'R² = 0.834 (high solar correlation)',
    'Good (06:00-09:00)': 'R² = 0.791 (morning stability)', 
    'Moderate (18:00-21:00)': 'R² = 0.757 (evening transitions)',
    'Challenging (00:00-05:00)': 'R² = 0.689 (night volatility)'
}
```

---

## Step 9: ONNX Deployment Optimization
## 🚀 Production-Ready Cross-Platform Model Optimization

### ⚙️ **ONNX Conversion Architecture**
```python
# Model conversion strategy
onnx_conversion = {
    'architecture': 'CatBoost multi-target → 5 single-target ONNX models',
    'conversion_status': '✅ All 5 models successfully converted and validated',
    'deployment_type': 'Hybrid: Single-target ONNX for flexible production',
    'model_format': 'T+1 to T+5 individual models for independent scaling'
}
```

### ⚡ **Performance Benchmarking Results**
| **Metric** | **CatBoost Native** | **ONNX Optimized** | **Improvement** |
|------------|--------------------|--------------------|----------------|
| **Inference Speed** | 0.002s | **0.0003s** | **1.51x faster** |
| **T+5 Specific** | Standard | **0.0006s ± 0.0032s** | Consistent performance |
| **Memory Usage** | Baseline | Optimized | Production efficient |
| **Cross-Platform** | Python only | **All platforms** | Universal deployment |

### 🎯 **Accuracy Validation (Spot Check)**
- **Prediction Comparison**: CatBoost vs ONNX for T+1 horizon
- **Maximum Difference**: **0.0175°C** between models
- **Mean Difference**: **0.0117°C** (excellent accuracy preservation)
- **Status**: ⚠️ Minor differences detected, acceptable for production

### 🌐 **Production Deployment Benefits**
- **Cross-Platform Compatibility**: Run on different hardware and operating systems
- **Performance Optimization**: 51% faster inference with maintained accuracy
- **Scalability**: Independent single-target models for flexible scaling
- **Production Ready**: Validated accuracy with <0.02°C maximum deviation
- **Universal Support**: Python, JavaScript, C++, Mobile (iOS/Android)

---

## 🏆 Final Results & Comprehensive Technical Achievements

### 🎯 **Project Success Metrics**
- **82.85% Accuracy**: R² = 0.828472 across 5-day multi-horizon forecasting
- **Production Ready**: Live Streamlit deployment with automated monitoring system
- **Cross-Platform**: ONNX optimization with 1.51x performance improvement
- **Open Source**: Complete end-to-end ML pipeline for community contribution

### 📊 **Detailed Technical Specifications**
- **Model Performance**: Mean R² = 0.828472, T+1 MAE = 1.14°C, T+5 RMSE = 2.42°C
- **Inference Optimization**: 0.0003s per prediction (ONNX), 1.6ms production latency
- **Data Engineering**: 10 years, 87,672+ hourly + 3,653 daily observations
- **Feature Engineering**: 95 selected features from 150+ engineered variables
- **Model Architecture**: Multi-output CatBoost with Optuna hyperparameter optimization

### 🌐 **Production Impact & Applications**
- **Live Web Application**: 🌐 **[hanoi-temperature-forecasting.streamlit.app](https://hanoi-temperature-forecasting.streamlit.app/)**
- **GitHub Repository**: Complete open-source implementation with comprehensive documentation
- **Educational Resource**: 9-step ML pipeline demonstration for learning and research
- **Scalable Framework**: Adaptable architecture for other cities and weather variables
- **Industry Applications**: Weather forecasting, urban planning, agriculture, tourism planning

---

## 👥 Team & Contributions

### Project Team
**Vu Ngoc Duong, Do Tuan Dat, Nguyen Thu Trang, Le Thi Anh Thu, Vu Tuan Dat**

### Individual Contributions
- **Data Engineering**: API integration, preprocessing pipeline
- **Model Development**: Feature engineering, hyperparameter tuning
- **Web Development**: Streamlit interface, visualization
- **Production**: ONNX optimization, monitoring system
- **Documentation**: Comprehensive project documentation

### GitHub Repository
🔗 **[Hanoi-Temperature-Forecasting](https://github.com/Duongvu05/Hanoi-Temperature-Forecasting)**

---

## 📈 Demonstration & Q&A

### Live Demo Features
- **Interactive Predictions**: Real-time 5-day forecasts
- **Historical Analysis**: 10-year trend visualization
- **Performance Metrics**: Model accuracy tracking
- **User-Friendly Interface**: Intuitive design for all users

### Key Questions Welcome
- Technical implementation details
- Model performance analysis
- Production deployment strategies
- Future enhancement possibilities
- Scalability and adaptation

---

## 🎉 Thank You!

### Project Success Highlights
- ✅ **82.85% accuracy** (R² = 0.828472) across 5-day multi-horizon temperature forecasting
- ✅ **Live web application** with real-time predictions and comprehensive monitoring
- ✅ **Production-ready system** with automated retraining and ONNX optimization
- ✅ **Complete ML pipeline** with 9-step implementation from data to deployment
- ✅ **Open source contribution** with comprehensive documentation for community learning

### Technical Achievements Summary
- **95 optimized features** from 150+ engineered variables
- **1.51x faster inference** with ONNX deployment optimization
- **Multi-granular forecasting**: Both daily and hourly prediction capabilities
- **Automated MLOps**: Performance monitoring with intelligent retraining system

### Contact & Resources
- **Team**: Vu Ngoc Duong, Do Tuan Dat, Nguyen Thu Trang, Le Thi Anh Thu, Vu Tuan Dat
- **GitHub**: [@Duongvu05](https://github.com/Duongvu05/Hanoi-Temperature-Forecasting)
- **Live Demo**: [Streamlit Application](https://hanoi-temperature-forecasting.streamlit.app/)
- **Documentation**: Complete project README with comprehensive technical specifications

**Questions & Discussion Welcome!** 🙋‍♂️🙋‍♀️