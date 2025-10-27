# GitHub Copilot Instructions for Hanoi Temperature Forecasting Project

## 🎯 Project Overview

This is a comprehensive **Hanoi Temperature Forecasting** machine learning application that predicts weather temperatures using 10+ years of historical data from the Visual Crossing API. The project follows a complete ML pipeline from data collection to production deployment with ONNX optimization.

## 🏗️ Project Architecture

### Core Technology Stack
- **Language**: Python 3.8+
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow/Keras
- **Data Processing**: pandas, numpy, matplotlib, seaborn, plotly
- **Web UI**: Streamlit with multi-page architecture
- **Optimization**: Optuna for hyperparameter tuning
- **Monitoring**: ClearML for experiment tracking
- **Deployment**: ONNX for production inference
- **Configuration**: YAML-based config management

### Directory Structure
```
Hanoi-Temperature-Forecasting/
├── src/                      # Core source code
│   ├── data/                 # Data collection & processing
│   ├── features/             # Feature engineering
│   ├── models/               # ML model implementations
│   ├── monitoring/           # Performance tracking
│   └── visualization/        # Plotting utilities
├── ui/                       # Streamlit web application
├── config/                   # YAML configuration files
├── notebooks/                # Jupyter analysis notebooks
├── data/                     # Raw and processed datasets
├── models/                   # Trained model artifacts
└── tests/                    # Unit and integration tests
```

## 🎯 Key Implementation Patterns

### 1. Configuration Management
- All settings centralized in `config/*.yaml` files
- Environment variables via `.env` for sensitive data
- Modular config loading in each component

### 2. Data Pipeline Architecture
- **Collection**: `src/data/collector.py` - Visual Crossing API integration with rate limiting
- **Processing**: `src/data/processor.py` - Validation, cleaning, feature extraction
- **Validation**: Comprehensive data quality checks and outlier detection

### 3. Feature Engineering Strategy
- **Temporal Features**: Lag variables, rolling averages, seasonal components
- **Weather Categories**: Temperature, atmospheric, wind, precipitation, solar, celestial
- **Text Processing**: Convert weather descriptions to numerical features
- **Target Engineering**: Multi-day temperature forecasting (1-5 days ahead)

### 4. Model Training Framework
- **Multi-Algorithm Support**: Random Forest, XGBoost, LightGBM, CatBoost, Neural Networks
- **Hyperparameter Optimization**: Optuna-based tuning with cross-validation
- **Ensemble Methods**: Weighted averaging and stacking approaches
- **Evaluation Metrics**: RMSE, MAPE, R², custom weather-specific metrics

### 5. UI Design Principles
- **Multi-Page Structure**: Data upload, exploration, training, predictions, monitoring
- **Interactive Visualizations**: Plotly charts with user controls
- **Real-time Feedback**: Progress indicators and status updates
- **Data Quality Dashboard**: Missing values, outliers, distribution analysis

## 🔧 Coding Guidelines

### Code Style & Organization
```python
# Always use descriptive imports with proper error handling
try:
    import pandas as pd
    import numpy as np
    from src.data.processor import WeatherDataProcessor
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

# Follow consistent class structure
class WeatherForecaster:
    """
    Temperature forecasting model with comprehensive error handling.
    
    Attributes:
        config (dict): Model configuration parameters
        model: Trained ML model instance
        scaler: Feature scaling transformer
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Train the forecasting model with validation."""
        # Implementation with proper logging and error handling
```

### Error Handling Patterns
```python
# Always implement comprehensive error handling
def collect_weather_data(location: str, start_date: str, end_date: str):
    """Collect weather data with robust error handling."""
    try:
        # Data collection logic
        response = api_client.fetch_data(location, start_date, end_date)
        
        if not response or response.status_code != 200:
            raise ValueError(f"API request failed: {response.status_code}")
            
        return self._process_response(response)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during data collection: {e}")
        raise
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in data collection: {e}")
        raise
```

### Configuration Loading
```python
# Standard configuration loading pattern
def load_config(config_path: str) -> dict:
    """Load YAML configuration with validation."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Validate required fields
        required_fields = ['data_source', 'model_params', 'training']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        
        return config
        
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise
```

## 🎯 Feature Development Guidelines

### 1. Data Processing Components
```python
# Always validate data before processing
def validate_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate weather data with comprehensive checks."""
    
    # Required columns check
    required_cols = ['datetime', 'temp', 'humidity', 'pressure']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Date range validation
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        date_range = (df['datetime'].max() - df['datetime'].min()).days
        if date_range < 30:
            logger.warning(f"Limited date range: {date_range} days")
    
    # Temperature range validation
    temp_cols = [col for col in df.columns if 'temp' in col.lower()]
    for col in temp_cols:
        if df[col].min() < -50 or df[col].max() > 60:
            logger.warning(f"Unusual temperature values in {col}")
    
    return df
```

### 2. Model Training Components
```python
# Standardized model training with hyperparameter tuning
def train_with_optimization(X_train: pd.DataFrame, y_train: pd.Series, 
                          model_type: str = 'xgboost') -> dict:
    """Train model with Optuna hyperparameter optimization."""
    
    def objective(trial):
        # Define hyperparameter search space based on model type
        if model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.8, 1.0)
            }
            model = XGBRegressor(**params, random_state=42)
        
        # Cross-validation evaluation
        scores = cross_val_score(model, X_train, y_train, 
                               cv=5, scoring='neg_mean_squared_error')
        return -scores.mean()
    
    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'model': model_type
    }
```

### 3. UI Component Development
```python
# Streamlit component patterns with proper state management
def create_data_upload_interface():
    """Create data upload interface with validation."""
    
    st.markdown("### 📁 Upload Weather Data")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file", 
        type=['csv'],
        help="Upload Hanoi weather data with required columns"
    )
    
    if uploaded_file is not None:
        try:
            # Load and validate data
            df = pd.read_csv(uploaded_file)
            df = validate_weather_data(df)
            
            # Display data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Records", f"{len(df):,}")
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                if 'datetime' in df.columns:
                    days = (df['datetime'].max() - df['datetime'].min()).days
                    st.metric("Date Range", f"{days} days")
            
            # Store in session state
            st.session_state['weather_data'] = df
            st.success("✅ Data loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
```

## 🔍 Testing & Quality Assurance

### Unit Testing Patterns
```python
# Comprehensive testing for data processing
class TestWeatherDataProcessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = WeatherDataProcessor()
        self.sample_data = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100),
            'temp': np.random.normal(20, 5, 100),
            'humidity': np.random.uniform(30, 90, 100)
        })
    
    def test_data_validation(self):
        """Test data validation functionality."""
        # Valid data should pass
        result = self.processor.validate_data(self.sample_data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # Invalid data should raise exception
        invalid_data = self.sample_data.drop('datetime', axis=1)
        with self.assertRaises(ValueError):
            self.processor.validate_data(invalid_data)
    
    def test_feature_engineering(self):
        """Test feature engineering pipeline."""
        features = self.processor.engineer_features(self.sample_data)
        
        # Check for required feature types
        self.assertIn('temp_lag1', features.columns)
        self.assertIn('temp_rolling_7d', features.columns)
        self.assertIn('month', features.columns)
```

## 🚀 Development Workflow

### 1. Adding New Features
1. **Configuration First**: Add settings to appropriate `config/*.yaml`
2. **Interface Design**: Define clear class interfaces and method signatures
3. **Implementation**: Follow error handling and logging patterns
4. **Testing**: Write comprehensive unit tests
5. **Integration**: Update UI components if needed
6. **Documentation**: Update docstrings and README

### 2. Model Development
1. **Experiment Setup**: Use Jupyter notebooks for initial exploration
2. **Implementation**: Move proven code to `src/models/`
3. **Hyperparameter Tuning**: Integrate with Optuna framework
4. **Evaluation**: Use consistent metrics (RMSE, MAPE, R²)
5. **Integration**: Add to model selection pipeline

### 3. UI Enhancement
1. **Component Design**: Create reusable Streamlit components
2. **State Management**: Use `st.session_state` consistently
3. **Error Handling**: Provide clear user feedback
4. **Performance**: Implement caching for expensive operations
5. **Testing**: Manual testing with various data scenarios

## 🎯 Current Development Status

### ✅ Completed Components
- Project structure and configuration system
- Data collection from Visual Crossing API
- Data processing and validation pipeline
- Comprehensive Streamlit UI application
- Feature distribution analysis notebooks
- Documentation and setup scripts

### 🚧 In Progress
- Feature engineering for time series forecasting
- Model training pipeline implementation
- Performance monitoring system
- ONNX model optimization

### 📋 Pending Tasks
- Unit test implementation
- Hourly data analysis extension
- Production deployment configuration
- Advanced ensemble methods
- Real-time data pipeline

## 🤖 AI Assistant Guidelines

When working on this project:

1. **Always check existing patterns** before implementing new features
2. **Follow the established architecture** for consistency
3. **Implement comprehensive error handling** for robustness
4. **Use the configuration system** for all parameters
5. **Add appropriate logging** for debugging and monitoring
6. **Create tests** for new functionality
7. **Update documentation** when adding features
8. **Consider UI integration** for user-facing features

### Common File Locations
- **Config**: `config/config.yaml`, `config/model_config.yaml`
- **Data Processing**: `src/data/processor.py`, `src/data/collector.py`
- **Models**: `src/models/daily_models.py`, `src/models/trainer.py`
- **UI Components**: `ui/app.py` (multi-page Streamlit app)
- **Tests**: `tests/` directory with organized test modules
- **Notebooks**: `notebooks/` for analysis and experimentation

This project prioritizes **code quality**, **user experience**, and **production readiness** while maintaining **scientific rigor** in the ML implementation.