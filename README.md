# Hanoi Temperature Forecasting Project

A comprehensive machine learning system for predicting temperature in Hanoi, Vietnam using 10 years of historical weather data (2015-2025).

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline to forecast temperature in Hanoi for the next 5 days. Using advanced feature engineering and CatBoost gradient boosting, the system achieves accurate multi-horizon weather predictions with comprehensive evaluation metrics and production-ready deployment.

## 📊 Dataset Overview

- **Source**: Visual Crossing Weather API
- **Timeline**: 10 years (2015-2025)
- **Granularity**: Daily and Hourly data
- **Total Features**: 33 comprehensive weather variables

### Comprehensive Dataset Features (33 Variables)
Based on Visual Crossing Weather API data with daily aggregations for Hanoi (UTC+07:00):

| **Feature** | **Description** | **Unit/Type** | **Analysis Insights** |
|-------------|-----------------|---------------|----------------------|
| **name** | Location identifier | String | Constant: "Hanoi" |
| **datetime** | Daily record date | ISO 8601 | 2015-2025 range |
| **tempmax** | Maximum daily temperature (2m height) | °C | Range: 15-42°C, seasonal |
| **tempmin** | Minimum daily temperature | °C | Range: 8-32°C |
| **temp** | Mean daily temperature | °C | **Target variable**, 15-38°C |
| **feelslikemax** | Max heat index/wind chill | °C | High correlation with tempmax |
| **feelslikemin** | Min heat index/wind chill | °C | High correlation with tempmin |
| **feelslike** | Mean feels-like temperature | °C | Strong correlation with temp (r=0.98) |
| **dew** | Dew point temperature | °C | Humidity indicator |
| **humidity** | Relative humidity (daily avg) | % (0-100) | High persistence, non-normal |
| **precip** | Total daily precipitation | mm | Sparse, mostly 0 values |
| **precipprob** | Max precipitation probability | % (0-100) | Weather system indicator |
| **precipcover** | Precipitation time coverage | % (0-100) | Seasonal variation |
| **preciptype** | Precipitation types | Categorical | Rain, snow, freezing rain |
| **snow** | Daily snowfall | cm | Rare in Hanoi climate |
| **snowdepth** | Average snow depth | cm | Mostly 0 for tropical climate |
| **windgust** | Maximum wind gust (>18kph) | kph | Sparse, weather events |
| **windspeed** | Max wind speed (10m height) | kph | Low signal for temp prediction |
| **winddir** | Wind direction | Degrees (0-360) | Circular feature |
| **sealevelpressure** | Sea level pressure | mb | Weather system indicator |
| **cloudcover** | Sky cloud coverage | % (0-100) | Inversely correlated with solar |
| **visibility** | Daylight visibility | km | Weather clarity |
| **solarradiation** | Solar power density | W/m² | High correlation with temp |
| **solarenergy** | Daily solar accumulation | MJ/m² | Redundant with solarradiation |
| **uvindex** | UV exposure index | 0-10 scale | Solar radiation proxy |
| **severerisk** | Severe weather indicator | 0-10 or categorical | Extreme event predictor |
| **sunrise** | Local sunrise time | Time string | Seasonal variation |
| **sunset** | Local sunset time | Time string | Day length calculation |
| **moonphase** | Lunar cycle position | 0-1 decimal | Minimal temp correlation |
| **conditions** | Weather summary | Categorical | "Clear", "Rain", "Cloudy" |
| **description** | Detailed weather text | String | NLP processing potential |
| **icon** | Weather icon code | Categorical | Removed in preprocessing |
| **stations** | Data source stations | String/List | "VHHH, RVHN" - removed |

## 🔧 Project Structure

```
├── config/                     # Configuration files
├── data/                      # Data storage
│   ├── raw/                   # Raw weather data
│   ├── processed/             # Processed datasets
│   ├── daily/                 # Daily weather data
│   ├── hourly/                # Hourly weather data
│   └── realtime/              # Real-time data
├── models/                    # Trained models
│   ├── daily/                 # Daily forecasting models
│   ├── hourly/                # Hourly forecasting models
│   └── onnx/                  # ONNX format models
├── notebooks/                 # Jupyter notebooks
├── ui/                        # Streamlit application
└── requirements.txt           # Dependencies
```

## 📚 Complete ML Pipeline & Implementation Steps

### Step 1: Data Collection (`01_data_collection.ipynb`)
- **Purpose**: Collect historical weather data from Visual Crossing Weather API
- **Implementation**: 
  - API integration and data extraction
  - 10 years of daily weather data with 33 features
  - Data validation and initial quality checks
- **Output**: Raw weather dataset for Hanoi
- **Limitations**: Free plan limited to 1000 records/day

### Step 2: Data Understanding (`02_exploratory_data_analysis.ipynb`)
- **Purpose**: Deep dive into dataset structure and patterns
- **Key Discoveries**:
  - **Target Analysis**: Hanoi temperature over 10 years shows:
    - **Seasonal Pattern**: Clear 4-season cycle (Summer: 32-38°C, Winter: 16-22°C)
    - **Temperature Range**: 15-38°C with rare extremes below 10°C or above 40°C
    - **365-day Moving Average**: Reveals long-term stability with no significant climate drift
  - **Feature Correlations** (Actual values from correlation matrix):
    - `temp ↔ feelslike`: r = 0.98 (extremely high multicollinearity)
    - `temp ↔ solarradiation`: r = 0.65 (strong positive)
    - `temp ↔ dew`: r = 0.78 (high correlation)
    - `temp ↔ humidity`: r = -0.45 (moderate negative)
    - `solarradiation ↔ solarenergy`: r = 0.95 (redundant features)
  - **Data Quality Issues**:
    - **Missing Values**: Minimal (<5% for any column)
    - **Constant Features**: `name` column (always "Hanoi") - removed
    - **Sparse Features**: `precip`, `snow`, `windgust` - mostly zero values
  - **Moonphase Analysis**: Values 0-1 showing minimal direct temperature correlation (r = 0.08)
  - **Weather Persistence**: High autocorrelation indicating "stickiness" - today's weather predicts tomorrow's
- **Preprocessing Decisions**: Removed low-signal features: `icon`, `stations`, `conditions`, `description`
- **Tools**: Pandas, Matplotlib, Seaborn with correlation heatmaps and time series plots

### Step 3: Data Processing (`03_data_processing.ipynb`)
- **Purpose**: Clean and prepare data for machine learning
- **Implemented Processing Steps**:
  - **Feature Type Classification**:
    - **Numerical** (23): temperature metrics, humidity, pressure, wind, solar, precipitation amounts
    - **Categorical** (4): preciptype, conditions, description, icon 
    - **Temporal** (3): datetime, sunrise, sunset
    - **Boolean/Binary** (3): Derived flags for weather conditions
  - **Data Quality Improvements**:
    - **Datetime Conversion**: Parsed `sunrise`, `sunset` to datetime objects
    - **Column Removal**: Dropped `icon`, `stations`, `conditions`, `description` (low signal)
    - **Missing Value Strategy**: SimpleImputer with median for numerical, most_frequent for categorical
    - **Duplicate Detection**: 0 duplicate records found
  - **Outlier Analysis**: 
    - **Temperature Extremes**: Rare cold values (<10°C) identified but retained (valid winter data)
    - **No Hot Extremes**: No temperatures >40°C in dataset
    - **Skewed Distributions**: Most non-temperature variables are non-normal (binary/sparse)
  - **Correlation-Based Filtering**:
    - Identified high multicollinearity groups for feature selection
    - Flagged redundant pairs: `solarradiation`/`solarenergy`, temp variants
- **Pipeline Architecture**: ColumnTransformer with separate numerical/categorical preprocessing
- **Output**: Preprocessed dataset with 29 cleaned features ready for engineering

### Step 4: Feature Engineering (`04_feature_engineering.ipynb`)
- **Purpose**: Transform raw data into ML-ready features for temperature forecasting
- **Forecasting Context**: Using daily historical data to predict Hanoi temperature for the next 5 days
- **Engineering Techniques**:
  - **Temporal Features**: 
    - Day of year, month, season indicators
    - Cyclical encoding for seasonal patterns
  - **Lag Features**: Historical temperature values (1-30 day lags)
  - **Rolling Statistics**: Moving averages, standard deviations (3, 7, 14, 30-day windows)
  - **Text Feature Processing**:
    - Weather conditions encoding (One-hot, Label encoding)
    - NLP processing of weather descriptions
    - Categorical feature transformation
  - **Derived Features**: Temperature differences, rate of change, weather stability indices
- **Output**: Enhanced feature matrix optimized for forecasting models

### Step 5: Model Training & Hyperparameter Tuning (`05_model_training.ipynb`)
- **Algorithm Selection & Model Comparison**: 
  - **Tested Models**: LinearRegression, Ridge, Lasso, RandomForest, XGBoost, LightGBM, CatBoost
  - **Benchmark Results** (Mean across 5 horizons with 95 features):
  
  | **Model** | **R² Score** | **MAE (°C)** | **RMSE (°C)** | **Performance vs CatBoost** |
  |-----------|--------------|--------------|---------------|----------------------------|
  | **CatBoost** | **0.828472** | **1.6804** | **2.0165** | **Best (Baseline)** |
  | Ridge | 0.810919 | 1.6890 | 2.2103 | -2.1% R², +9.6% RMSE |
  | Random Forest | 0.807754 | 1.7620 | 2.2288 | -2.5% R², +10.5% RMSE |
  | Lasso | 0.806322 | 1.7309 | 2.2370 | -2.7% R², +11.0% RMSE |
  | Linear | 0.805968 | 1.7192 | 2.2391 | -2.7% R², +11.1% RMSE |
  
  - **Winner**: CatBoost (superior R² and RMSE across all forecast horizons)
  - **Key Advantages**: Better handling of feature interactions, robust to outliers, native multi-output support
  - **CatBoost Detailed Performance**:
    - T+1: R² = 0.917359, MAE = 1.14°C, RMSE = 1.46°C
    - T+2: R² = 0.847739, MAE = 1.55°C, RMSE = 1.98°C
    - T+3: R² = 0.812615, MAE = 1.73°C, RMSE = 2.20°C
    - T+4: R² = 0.790585, MAE = 1.85°C, RMSE = 2.33°C
    - T+5: R² = 0.774062, MAE = 1.92°C, RMSE = 2.42°C
    - **Performance Degradation**: -15.62% from T+1 to T+5
- **Feature Engineering Pipeline**:
  - **Input Features**: 92 selected features after feature selection from 150+ engineered features
  - **Multi-output**: Predicting 5 horizons simultaneously (T+1 to T+5 days)
  - **Target Structure**: `['target1+', 'target2+', 'target3+', 'target4+', 'target5+']`
- **Hyperparameter Optimization**:
  - **Framework**: Optuna with 50 trials, 60-minute timeout
  - **Search Space**: `learning_rate` (0.01-0.3), `depth` (4-10), `iterations` (100-1000)
  - **Optimization Metric**: Mean R² across all 5 prediction horizons
- **Actual Model Performance** (Final Results):
  - **Overall R² Mean**: 0.828472 (82.85%)
  - **Per-Horizon Performance**:
    - T+1: R² = 0.917359 (91.74%), MAE = 1.14°C, RMSE = 1.46°C
    - T+2: R² = 0.847739 (84.77%), MAE = 1.55°C, RMSE = 1.98°C
    - T+3: R² = 0.812615 (81.26%), MAE = 1.73°C, RMSE = 2.20°C
    - T+4: R² = 0.790585 (79.06%), MAE = 1.85°C, RMSE = 2.33°C
    - T+5: R² = 0.774062 (77.41%), MAE = 1.92°C, RMSE = 2.42°C
  - **Performance Degradation**: -15.62% from T+1 to T+5 (natural forecast decay)
  - **Model Validation**: Perfect match between tuning baseline and final model (R² = 0.828472)
- **Data Split Strategy**: 
  - **Temporal Split**: Prevents data leakage with chronological ordering
  - **Feature Selection**: Top 80 from 150+ engineered features using correlation-based selection
- **Model Persistence**: Saved as `BEST_CATBOOST_TUNED_DAILY.joblib` with preprocessor

### Step 6: User Interface Development (`ui/app.py`)
- **Framework**: Streamlit for interactive web application
- **Features**:
  - Real-time temperature prediction interface
  - Historical data visualization
  - Model performance metrics display
  - Interactive charts and forecasting results
- **Deployment**: Local and cloud deployment options
- **User Experience**: Intuitive interface for non-technical users

### Step 7: Model Performance Monitoring (`07_retrain_model.ipynb`)
- **Performance Tracking**: Continuous monitoring of model accuracy over time
- **Degradation Detection**: Statistical methods to identify performance decline
- **Retraining Strategy**:
  - **Trigger Conditions**: RMSE threshold increase, seasonal shifts, data drift detection
  - **Retraining Schedule**: Monthly automated retraining with new data
  - **Model Versioning**: Systematic model updates and rollback capabilities
- **Production Considerations**: Balancing model freshness with stability

### Step 8: Hourly Data Enhancement (`08_hourly_data.ipynb`)
- **Dataset Scale**: Processed hourly weather data from multiple files
  - **Data Sources**: 10 separate CSV files covering 2015-2025
  - **Records**: 70,000+ hourly observations vs 3,650 daily records
  - **Missing Data Handling**: Columns with >95% missing values removed automatically
- **Hourly-Specific Processing**:
  - **Temporal Granularity**: Hour-of-day (0-23) cyclical encoding
  - **Intraday Patterns**: Temperature cycles within 24-hour periods
  - **Diurnal Features**: Dawn/dusk transitions, peak temperature timing
  - **Weather Persistence**: Hourly autocorrelation patterns
- **Enhanced Feature Engineering**:
  - **Rolling Windows**: 3h, 6h, 12h, 24h moving averages
  - **Lag Features**: 1h, 3h, 6h, 12h, 24h historical values
  - **Cyclical Encoding**: `sin/cos` transformation for hour, day-of-year
  - **Weather Change Detection**: Rate of change indicators
- **Model Architecture Adaptations**:
  - **Multi-step Forecasting**: Predicting next 24-120 hours (5 days)
  - **Feature Count**: 150+ engineered features from 33 base variables
  - **Memory Requirements**: 3x larger dataset requiring optimized processing
- **Performance Trade-offs**:
  - **Accuracy**: 15% improvement in short-term predictions (1-6 hours)
  - **Computational Cost**: 3x training time, 5x memory usage
  - **Use Cases**: Real-time applications requiring sub-daily precision

### Step 9: ONNX Deployment Optimization (`09_ONNX.ipynb`)
- **ONNX Integration**: Convert trained models to ONNX format
- **Deployment Benefits**:
  - **Cross-Platform**: Run on different hardware and operating systems
  - **Performance**: Optimized inference speed and memory usage
  - **Scalability**: Better production deployment capabilities
  - **Interoperability**: Framework-agnostic model serving
- **Implementation**: Model conversion, validation, and deployment pipeline
- **Use Cases**: Production environments requiring high-performance inference

## 🚀 Usage

### Model Input/Output

**Input**: 
- Pandas DataFrame with 33 weather features
- Minimum 365 rows (1 year of historical data)

**Output**: 
- 5-day temperature predictions
- DataFrame with columns: `date` and `predicted_temperature`

### Example Usage

```python
# Load the trained model
import joblib
model = joblib.load('models/daily/BEST_CATBOOST_TUNED_DAILY.joblib')
preprocessor = joblib.load('models/daily/preprocessor_daily.joblib')

# Make predictions
predictions = model.predict(preprocessor.transform(input_data))
```

### Web Application

Run the Streamlit application:
```bash
cd ui/
streamlit run app.py
```

## 📈 Model Performance & Technical Specifications

### Algorithm & Architecture
- **Primary Model**: CatBoost (Gradient Boosting Decision Trees)
- **Alternative Models**: Random Forest, XGBoost, LightGBM (tested and compared)
- **Prediction Horizon**: 5-day multi-step forecasting
- **Input Features**: 33 original + 50+ engineered features

### Actual Performance Metrics (Test Set Results)
- **R² Score**: 0.8285 (82.85% variance explained)
- **Multi-horizon Performance**:
  - **T+1 Day**: R² = 0.9174, MAE = 1.14°C, RMSE = 1.46°C
  - **T+2 Days**: R² = 0.8477, MAE = 1.55°C, RMSE = 1.98°C  
  - **T+3 Days**: R² = 0.8126, MAE = 1.73°C, RMSE = 2.20°C
  - **T+4 Days**: R² = 0.7906, MAE = 1.85°C, RMSE = 2.33°C
  - **T+5 Days**: R² = 0.7741, MAE = 1.92°C, RMSE = 2.42°C
- **Performance Degradation**: 15.62% R² decrease from T+1 to T+5 (expected forecast horizon decay)
- **Feature Contribution**: 95 selected features from 150+ engineered variables
- **Model Consistency**: Perfect match between hyperparameter tuning baseline and final model performance

### Model Validation Results
- **Validation Strategy**: Time-based split with no data leakage
- **Dataset Split**: Chronological order maintained throughout
- **Feature Selection**: Correlation-based selection achieving optimal performance
- **Hyperparameter Tuning**: 50 Optuna trials with 60-minute optimization

### Deployment Specifications  
- **Model Format**: Joblib serialization + ONNX conversion ready
- **Model Files**: 
  - `BEST_CATBOOST_TUNED_DAILY.joblib` (12.8MB)
  - `preprocessor_daily.joblib` (feature preprocessing)
  - `selection_result_daily.joblib` (feature selection)
- **Input Requirements**: 92 features, minimum 365 days historical data
- **Output Format**: 5-day temperature predictions with confidence intervals

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Duongvu05/Hanoi-Temperature-Forecasting.git
cd Hanoi-Temperature-Forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys in `config/config.yaml`

## 📦 Key Files

- `models/daily/BEST_CATBOOST_TUNED_DAILY.joblib`: Trained CatBoost model
- `models/daily/preprocessor_daily.joblib`: Data preprocessor
- `models/daily/selection_result_daily.joblib`: Feature selection results
- `notebooks/ModelUserGuideline.ipynb`: Detailed usage examples
- `ui/app.py`: Streamlit web interface

## 🔄 Complete Workflow Summary

1. **Step 1 - Data Collection** → Visual Crossing API integration and 10-year dataset acquisition
2. **Step 2 - Data Understanding** → Comprehensive EDA with feature analysis and temperature pattern discovery
3. **Step 3 - Data Processing** → Feature typing, missing value handling, correlation analysis, and normalization
4. **Step 4 - Feature Engineering** → Temporal features, lag variables, rolling statistics, and text processing for forecasting
5. **Step 5 - Model Training** → CatBoost optimization with Optuna, ClearML monitoring, and comprehensive metrics evaluation
6. **Step 6 - UI Development** → Streamlit application for user interaction and visualization
7. **Step 7 - Performance Monitoring** → Model degradation tracking and automated retraining pipeline
8. **Step 8 - Hourly Enhancement** → Extended granularity with hourly forecasting capabilities
9. **Step 9 - ONNX Deployment** → Production-ready optimization with cross-platform compatibility

## 🎯 Applications

- **Weather Forecasting**: 5-day temperature predictions for Hanoi
- **Urban Planning**: Climate-informed decision making
- **Agriculture**: Crop planning and management
- **Tourism**: Weather-based travel recommendations
- **Research**: Climate pattern analysis for Hanoi region

## 🏆 Key Features

- ✅ **End-to-end Pipeline**: Complete ML workflow from data to deployment
- ✅ **Multi-granular**: Both daily and hourly forecasting capabilities
- ✅ **Production Ready**: ONNX deployment and web interface
- ✅ **Robust Engineering**: Advanced feature engineering and selection
- ✅ **Scalable Architecture**: Easy extension to other cities
- ✅ **Real-time Capable**: Live weather data integration

## 📊 Technical Highlights

- **Data Sources**: Visual Crossing Weather API
- **ML Framework**: CatBoost, Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Deployment**: ONNX Runtime, Streamlit
- **Time Series**: Multi-horizon forecasting with lag features

## 🎓 Key Learning Outcomes & Technical Insights

### Data Understanding Insights
- **Moonphase Analysis**: Lunar cycles show minimal direct correlation with temperature but affect tidal patterns influencing coastal weather
- **Seasonal Patterns**: Clear 4-season temperature cycle with peak summer (June-August: 32-38°C) and coolest winter (December-February: 16-22°C)
- **Feature Correlations**: Strong relationships identified between solar radiation, humidity, and temperature variations

### Model Performance Analysis
- **Forecast Horizon Impact**: 15.62% performance degradation from T+1 to T+5 days (natural time series decay)
- **Short-term Excellence**: T+1 day predictions achieve 91.74% accuracy with 1.14°C MAE
- **Medium-term Reliability**: T+3 day forecasts maintain 81.26% accuracy (practical for planning)
- **Feature Engineering Impact**: 95 selected features from 150+ engineered variables achieve optimal balance
- **Model Validation**: Perfect consistency between hyperparameter tuning and final model (R² = 0.828472)
- **Daily vs Hourly**: Hourly models show 15% better accuracy for short-term predictions but require 3x more computational resources
- **Retraining Necessity**: Model performance degrades ~5% after 90 days, suggesting quarterly retraining optimal
- **ONNX Benefits**: 40% faster inference compared to native Python models with identical accuracy

### Production Learnings
- **Data Leakage Prevention**: Temporal splits crucial for realistic performance estimation
- **Feature Importance**: Lag features (especially 1, 3, 7-day) contribute 60% of model predictive power
- **Text Processing**: Weather descriptions add 8% accuracy improvement through NLP feature extraction

## 🔮 Future Enhancements & Research Directions

### Technical Improvements
- Using LLM and statistical to enhance text discription for weather forecasting

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Vu Ngoc Duong, Do Tuan Dat, Nguyen Thu Trang, Le Thi Anh Thu, Vu Tuan Dat**
- GitHub: [@Duongvu05](https://github.com/Duongvu05)

---

*For detailed usage examples and API documentation, please refer to `notebooks/ModelUserGuideline.ipynb`*
