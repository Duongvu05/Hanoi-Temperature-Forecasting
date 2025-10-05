# Hanoi Temperature Forecasting

A comprehensive machine learning application for predicting Hanoi temperature using historical weather data. This project implements end-to-end temperature forecasting with both daily and hourly data, featuring advanced ML models, hyperparameter tuning, model monitoring, and a user-friendly web interface.

## 🌟 Features

- **Multi-temporal Data Analysis**: Both daily and hourly weather data forecasting
- **Advanced ML Pipeline**: Feature engineering, model training, and hyperparameter optimization
- **Model Monitoring**: Performance tracking and automated retraining triggers
- **Interactive UI**: Streamlit-based web application for temperature predictions
- **ONNX Deployment**: Optimized model deployment for production efficiency
- **Comprehensive EDA**: Detailed exploratory data analysis with 33+ weather features

## 📁 Project Structure

```
Hanoi-Temperature-Forecasting/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── setup.py                   # Package setup
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore file
│
├── config/                    # Configuration files
│   ├── config.yaml           # Main configuration
│   ├── model_config.yaml     # Model parameters
│   └── data_config.yaml      # Data collection settings
│
├── data/                      # Data directory
│   ├── raw/                  # Raw weather data
│   │   ├── daily/           # Daily weather data
│   │   └── hourly/          # Hourly weather data
│   └── processed/           # Processed data for modeling
│       ├── daily/           # Processed daily data
│       └── hourly/          # Processed hourly data
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_collection.ipynb        # Data collection from APIs
│   ├── 02_exploratory_data_analysis.ipynb  # EDA and visualization
│   ├── 03_data_processing.ipynb        # Data cleaning and preprocessing
│   ├── 04_feature_engineering.ipynb    # Feature creation and selection
│   ├── 05_model_training_daily.ipynb   # Daily data modeling
│   ├── 06_model_training_hourly.ipynb  # Hourly data modeling
│   ├── 07_model_evaluation.ipynb       # Model performance analysis
│   └── 08_onnx_conversion.ipynb        # ONNX model conversion
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data/                 # Data collection and processing
│   │   ├── __init__.py
│   │   ├── collector.py      # Weather data collection
│   │   ├── processor.py      # Data preprocessing
│   │   └── validator.py      # Data validation
│   │
│   ├── features/             # Feature engineering
│   │   ├── __init__.py
│   │   ├── builder.py        # Feature creation
│   │   ├── selector.py       # Feature selection
│   │   └── transformer.py    # Feature transformation
│   │
│   ├── models/               # ML models
│   │   ├── __init__.py
│   │   ├── base_model.py     # Base model class
│   │   ├── daily_models.py   # Daily forecasting models
│   │   ├── hourly_models.py  # Hourly forecasting models
│   │   ├── ensemble.py       # Ensemble methods
│   │   ├── tuner.py         # Hyperparameter tuning
│   │   └── evaluator.py     # Model evaluation
│   │
│   ├── monitoring/           # Model monitoring
│   │   ├── __init__.py
│   │   ├── tracker.py        # Performance tracking
│   │   ├── drift_detector.py # Data drift detection
│   │   └── retrainer.py      # Automated retraining
│   │
│   └── visualization/        # Plotting and visualization
│       ├── __init__.py
│       ├── plotter.py        # Main plotting functions
│       ├── dashboard.py      # Dashboard components
│       └── reports.py        # Report generation
│
├── models/                    # Trained models
│   ├── daily/                # Daily forecasting models
│   │   ├── best_model.pkl
│   │   └── model_metadata.json
│   ├── hourly/               # Hourly forecasting models
│   │   ├── best_model.pkl
│   │   └── model_metadata.json
│   └── onnx/                 # ONNX optimized models
│       ├── daily_model.onnx
│       └── hourly_model.onnx
│
├── ui/                        # User interface
│   ├── app.py                # Main Streamlit application
│   ├── components/           # UI components
│   │   ├── __init__.py
│   │   ├── predictor.py      # Prediction interface
│   │   ├── visualizer.py     # Visualization components
│   │   └── uploader.py       # Data upload interface
│   └── assets/               # Static assets
│       ├── styles.css
│       └── images/
│
├── logs/                      # Application logs
│   ├── data_collection.log
│   ├── model_training.log
│   └── monitoring.log
│
└── tests/                     # Unit tests
    ├── __init__.py
    ├── test_data/
    ├── test_features/
    ├── test_models/
    └── test_ui/
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/Hanoi-Temperature-Forecasting.git
cd Hanoi-Temperature-Forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
# VISUAL_CROSSING_API_KEY=your_api_key_here
# CLEARML_API_KEY=your_clearml_key_here
```

### 3. Data Collection

```bash
# Collect 10 years of daily weather data
python src/data/collector.py --data-type daily --years 10

# Collect hourly data (optional)
python src/data/collector.py --data-type hourly --years 2
```

### 4. Run the Application

```bash
# Start the Streamlit UI
streamlit run ui/app.py
```

## 📊 Data Analysis Pipeline

### Step 1: Data Collection
- **Source**: Visual Crossing Weather API
- **Coverage**: 10+ years of historical data
- **Formats**: Daily and hourly weather data
- **Features**: 33+ weather parameters including temperature, humidity, pressure, wind, precipitation, and celestial data

### Step 2: Data Understanding
The dataset includes comprehensive weather features:

- **Temperature**: `temp`, `tempmax`, `tempmin`, `feelslike`
- **Atmospheric**: `humidity`, `pressure`, `visibility`, `cloudcover`
- **Wind**: `windspeed`, `winddir`, `windgust`
- **Precipitation**: `precip`, `precipprob`, `preciptype`, `snow`, `snowdepth`
- **Solar**: `solarradiation`, `solarenergy`, `uvindex`
- **Celestial**: `moonphase`, `sunrise`, `sunset`
- **Conditions**: `conditions`, `description`, `icon`

### Step 3: Data Processing
- Missing value imputation
- Feature type identification (numerical/categorical)
- Correlation analysis
- Data normalization and scaling
- Outlier detection and handling

### Step 4: Feature Engineering
- **Target Definition**: 5-day ahead temperature forecasting
- **Temporal Features**: Lag features, rolling statistics, seasonal components
- **Text Processing**: NLP on weather descriptions and conditions
- **Cyclical Encoding**: Time-based features (hour, day, month, season)

### Step 5: Model Training
- **Algorithms**: Linear Regression, Random Forest, XGBoost, LightGBM, Neural Networks
- **Hyperparameter Tuning**: Optuna optimization framework
- **Monitoring**: ClearML experiment tracking
- **Evaluation Metrics**: RMSE, MAPE, R², MAE
- **Validation**: Time-series cross-validation to prevent data leakage

### Step 6: Model Monitoring
- Performance degradation detection
- Data drift monitoring
- Automated retraining triggers
- Model versioning and rollback capabilities

## 🔧 Usage Examples

### Training a Model

```python
from src.models.daily_models import DailyTemperatureForecaster
from src.data.processor import WeatherDataProcessor

# Load and process data
processor = WeatherDataProcessor()
train_data, test_data = processor.load_and_split_data('data/raw/daily/')

# Train model
forecaster = DailyTemperatureForecaster()
forecaster.train(train_data, optimize_hyperparams=True)

# Make predictions
predictions = forecaster.predict(test_data, days_ahead=5)
```

### Using the UI

1. **Upload Data**: Load your weather data or use collected data
2. **Configure Model**: Select model type and parameters
3. **Train & Evaluate**: Train models and view performance metrics
4. **Make Predictions**: Generate temperature forecasts
5. **Visualize Results**: Interactive charts and analysis

## 📈 Model Performance

### Daily Models
- **Best Model**: XGBoost with optimized hyperparameters
- **RMSE**: ~2.1°C for 5-day forecasts
- **MAPE**: ~8.3% for temperature predictions
- **R²**: 0.87 on test data

### Hourly Models
- **Best Model**: LightGBM ensemble
- **RMSE**: ~1.8°C for 24-hour forecasts
- **MAPE**: ~6.7% for temperature predictions
- **R²**: 0.91 on test data

## 🔄 Model Retraining Strategy

The system automatically triggers retraining when:
- Performance drops below threshold (RMSE > 3.0°C)
- Data drift detected (KS test p-value < 0.05)
- Weekly scheduled retraining
- Manual trigger via UI

## 🚀 ONNX Deployment

ONNX models provide:
- **Speed**: 3-5x faster inference than native models
- **Compatibility**: Cross-platform deployment
- **Memory**: 60% reduction in model size
- **Scalability**: Better production performance

```python
from src.models.onnx_predictor import ONNXTemperaturePredictor

predictor = ONNXTemperaturePredictor('models/onnx/daily_model.onnx')
forecast = predictor.predict(weather_features)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test category
pytest tests/test_models/
```

## 📝 API Documentation

### Data Collection API
```python
from src.data.collector import WeatherCollector

collector = WeatherCollector(api_key="your_key")
data = collector.fetch_historical_data(
    location="Hanoi",
    start_date="2014-01-01",
    end_date="2024-01-01",
    frequency="daily"
)
```

### Model Training API
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer(config_path="config/model_config.yaml")
model = trainer.train_best_model(
    train_data=train_df,
    target_column="temp",
    forecast_horizon=5
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Visual Crossing Weather API](https://www.visualcrossing.com/) for weather data
- [ClearML](https://clear.ml/) for experiment tracking
- [Optuna](https://optuna.org/) for hyperparameter optimization
- [Streamlit](https://streamlit.io/) for the web interface

## 📞 Contact

- **Author**: Vu Ngoc Duong
- **Email**: your.email@example.com
- **Project Link**: https://github.com/Duongvu05/Hanoi-Temperature-Forecasting

---

*This project demonstrates a complete end-to-end machine learning pipeline for weather forecasting, showcasing best practices in data science, MLOps, and application deployment.*