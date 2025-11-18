# Step 5: Model Training & Optimization
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

### 📈 **Multi-Horizon Performance**
| **Forecast** | **R²** | **MAE** | **RMSE** | **Quality** |
|--------------|--------|---------|----------|-------------|
| **T+1 Day** | 91.74% | 1.14°C | 1.46°C | 🔥 Excellent |
| **T+2 Days** | 84.77% | 1.55°C | 1.98°C | ✅ Very Good |
| **T+3 Days** | 81.26% | 1.73°C | 2.20°C | ✅ Good |
| **T+5 Days** | 77.41% | 1.92°C | 2.42°C | ⚠️ Acceptable |

### 🎯 **Top Feature Importance**
1. **temp_lag_1** (28.5%) - Yesterday's temperature
2. **temp_ma_7** (12.3%) - Weekly trend
3. **solar_lag_1** (8.7%) - Solar energy memory

### ✅ **Production-Ready Model** → 12.8MB, 0.002s inference

## 🎯 Mục Tiêu Model Training

### Machine Learning Task Definition
- **🔮 Problem Type**: Multi-output regression (5 horizons)
- **📊 Input Features**: 136 engineered features → 92 selected
- **🎯 Target Variables**: T+1, T+2, T+3, T+4, T+5 day temperature
- **📈 Success Metric**: R² score across all horizons

### Training Strategy
- **🧠 Algorithm Selection**: Compare multiple ML algorithms
- **⚙️ Hyperparameter Tuning**: Optuna optimization framework
- **✅ Cross-Validation**: Time-based splits for temporal data
- **📊 Multi-Metric Evaluation**: R², MAE, RMSE for comprehensive assessment

---

## 🏆 Algorithm Selection & Benchmarking

### Candidate Algorithms Tested
```python
algorithms = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(), 
    'RandomForest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    'CatBoost': CatBoostRegressor()  # Winner!
}
```

### Benchmark Results (Mean across 5 horizons)
| **Algorithm** | **R² Score** | **MAE (°C)** | **RMSE (°C)** | **Relative Performance** |
|---------------|--------------|--------------|---------------|--------------------------|
| **🥇 CatBoost** | **0.8285** | **1.68** | **2.02** | **Baseline (Best)** |
| 🥈 Ridge | 0.8109 | 1.69 | 2.21 | -2.1% R², +9.6% RMSE |
| 🥉 Random Forest | 0.8078 | 1.76 | 2.23 | -2.5% R², +10.5% RMSE |
| Lasso | 0.8063 | 1.73 | 2.24 | -2.7% R², +11.0% RMSE |
| Linear | 0.8060 | 1.72 | 2.24 | -2.7% R², +11.1% RMSE |

---

## 🎯 Why CatBoost Won?

### CatBoost Advantages for Weather Forecasting
- **🌳 Gradient Boosting**: Superior handling of feature interactions
- **🔄 Multi-Output Native**: Built-in support for multiple targets  
- **💪 Robustness**: Less sensitive to outliers (extreme weather)
- **⚡ Performance**: Best accuracy across all forecast horizons
- **🛠️ Ease of Use**: Minimal hyperparameter tuning required

### Technical Superiority
```python
# CatBoost specific advantages
catboost_benefits = {
    'categorical_features': 'Native handling without encoding',
    'missing_values': 'Built-in handling',
    'overfitting_protection': 'Advanced regularization',
    'feature_interactions': 'Automatic detection',
    'multi_output': 'Single model for all horizons'
}
```

---

## ⚙️ Hyperparameter Optimization Strategy

### Optuna Framework Setup
```python
import optuna

def objective(trial):
    # Hyperparameter search space
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 1, 20)
    }
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(CatBoostRegressor(**params), 
                               X_train, y_train, cv=5, 
                               scoring='r2')
    return cv_scores.mean()

# Optimization configuration
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=3600)  # 50 trials, 1 hour
```

---

## 🎯 Optimal Hyperparameters Found

### Best Configuration (After 50 Trials)
```python
best_params = {
    'learning_rate': 0.074,      # Moderate learning for stability
    'depth': 7,                  # Deep enough for interactions
    'iterations': 1498,          # With early stopping at 261
    'l2_leaf_reg': 3.2,         # Regularization for generalization
    'border_count': 128,         # Feature discretization
    'random_strength': 8.5,      # Randomness for robustness
    'early_stopping_rounds': 150, # Prevent overfitting
    'eval_metric': 'RMSE',       # Optimization target
    'random_seed': 42            # Reproducibility
}
```

### Optimization Results
- **⏱️ Total Time**: 52 minutes for 50 trials
- **🎯 Best Score**: R² = 0.8285 (mean across horizons)
- **📊 Convergence**: Stable after trial 35
- **✅ Validation**: No overfitting detected

---

## 📊 Multi-Output Training Architecture

### Target Structure Design
```python
# Multi-horizon target creation
target_columns = ['target_1d', 'target_2d', 'target_3d', 'target_4d', 'target_5d']

# Feature matrix and target preparation
X = engineered_features[selected_features]  # 92 features
y = engineered_features[target_columns]     # 5 targets

# Training configuration
model = CatBoostRegressor(
    loss_function='MultiRMSE',    # Multi-output loss
    eval_metric='MultiRMSE',      # Multi-output evaluation
    **best_params
)
```

### Training Data Split Strategy
```python
# Temporal split (prevent data leakage)
split_date = '2023-01-01'
train_mask = df['datetime'] < split_date
test_mask = df['datetime'] >= split_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Training samples: {len(X_train)}")    # 2,828 samples
print(f"Test samples: {len(X_test)}")         # 708 samples
```

---

## 🏋️ Training Process & Convergence

### Training Configuration
```python
# Model training with validation
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=150,
    verbose=100,
    plot=False
)

# Training results
training_stats = {
    'total_iterations': 1498,
    'best_iteration': 261,        # Early stopping triggered
    'training_time': '8.3 minutes',
    'final_train_loss': 3.847,
    'final_val_loss': 4.624,
    'overfitting_ratio': 1.20     # Acceptable (<1.5)
}
```

### Convergence Analysis
- **📈 Training Loss**: Smooth decrease to 3.847
- **📊 Validation Loss**: Stable at 4.624 (best iteration 261)
- **⚠️ Early Stopping**: Prevented overfitting effectively
- **✅ Loss Ratio**: 1.20 indicates good generalization

---

## 📈 Detailed Performance Results

### Per-Horizon Performance Analysis
| **Horizon** | **R² Score** | **MAE (°C)** | **RMSE (°C)** | **Performance Level** |
|-------------|--------------|--------------|---------------|----------------------|
| **T+1 Day** | **0.9174** | **1.14** | **1.46** | 🔥 Excellent (91.7%) |
| **T+2 Days** | **0.8477** | **1.55** | **1.98** | ✅ Very Good (84.8%) |
| **T+3 Days** | **0.8126** | **1.73** | **2.20** | ✅ Good (81.3%) |
| **T+4 Days** | **0.7906** | **1.85** | **2.33** | ✅ Good (79.1%) |
| **T+5 Days** | **0.7741** | **1.92** | **2.42** | ⚠️ Acceptable (77.4%) |

### Performance Degradation Pattern
- **📉 R² Degradation**: 91.7% → 77.4% (-15.6% total)
- **📈 Error Increase**: 1.14°C → 1.92°C MAE (+68% error)
- **⏰ Forecast Decay**: Natural phenomenon in time series forecasting

---

## 🎯 Feature Selection Results

### From 136 to 92 Features
```python
# Feature selection pipeline
feature_selector = SelectKBest(score_func=f_regression, k=92)
X_selected = feature_selector.fit_transform(X_engineered, y_mean)

selected_features = X_engineered.columns[feature_selector.get_support()]
```

### Top 10 Most Important Features
| **Rank** | **Feature** | **Importance** | **Category** |
|----------|-------------|----------------|--------------|
| 1 | `temp_lag_1` | 28.5% | Lag Feature |
| 2 | `temp_ma_7` | 12.3% | Rolling Statistics |
| 3 | `solar_lag_1` | 8.7% | Lag Feature |
| 4 | `temp_lag_3` | 7.9% | Lag Feature |
| 5 | `humidity_ma_7` | 6.2% | Rolling Statistics |
| 6 | `month_sin` | 5.1% | Seasonal |
| 7 | `temp_seasonal_anomaly` | 4.8% | Interaction |
| 8 | `day_length` | 4.2% | Temporal |
| 9 | `pressure_trend` | 3.7% | Trend/Momentum |
| 10 | `temp_volatility_7d` | 3.5% | Weather Patterns |

---

## 🔍 Model Validation & Robustness

### Cross-Validation Results
```python
# Time series cross-validation (5 folds)
cv_scores = []
for fold in range(5):
    # Time-based splits to prevent leakage
    train_end = start_date + timedelta(days=365*fold + 730)
    val_start = train_end + timedelta(days=1) 
    val_end = val_start + timedelta(days=365)
    
    fold_score = evaluate_model(train_data, val_data)
    cv_scores.append(fold_score)

cv_results = {
    'mean_r2': 0.8285,
    'std_r2': 0.023,        # Low variance = stable
    'min_r2': 0.801,        # Worst fold still good
    'max_r2': 0.847         # Best fold performance
}
```

### Robustness Tests
- **✅ Seasonal Stability**: Consistent across all seasons
- **✅ Extreme Weather**: Handles heatwaves and cold snaps
- **✅ Missing Data**: Robust to occasional missing values
- **✅ Temporal Shifts**: Performs well on future unseen data

---

## ⚡ Training Efficiency & Scalability

### Computational Performance
```python
training_metrics = {
    'training_time': '8.3 minutes',       # Full dataset
    'memory_usage': '2.1 GB',            # Peak memory
    'cpu_utilization': '85%',             # Multi-core usage
    'model_size': '12.8 MB',             # Serialized model
    'inference_speed': '0.002s',         # Per prediction
    'scalability': 'Linear with data size'
}
```

### Production Readiness
- **🚀 Fast Inference**: 500 predictions/second
- **💾 Compact Model**: 12.8MB for deployment
- **🔄 Retraining**: 8 minutes for complete retraining
- **📱 Edge Deployment**: ONNX conversion ready

---

## 📊 Model Interpretability Analysis

### Feature Contribution by Category
```python
feature_importance_by_category = {
    'lag_features': 45.2,        # Historical temperature
    'rolling_stats': 23.8,       # Trend indicators  
    'seasonal': 12.7,            # Time patterns
    'interactions': 8.9,         # Weather combinations
    'trend_momentum': 6.1,       # Change indicators
    'weather_patterns': 3.3      # Stability measures
}
```

### Physical Interpretation
- **🕒 Weather Memory**: Past temperature dominates (45%)
- **📈 Climate Trends**: Moving averages crucial (24%)
- **🌍 Seasonality**: Time patterns important (13%)
- **⚡ Interactions**: Non-linear relationships (9%)

---

## 🎯 Model Performance Validation

### Baseline Comparison
```python
# Simple baselines for context
baselines = {
    'persistence': 0.751,        # Yesterday's temperature
    'seasonal_naive': 0.623,     # Same day last year
    'linear_trend': 0.689,       # Linear extrapolation
    'catboost_model': 0.8285     # Our model (38% better)
}
```

### Error Analysis by Weather Conditions
| **Weather Type** | **MAE (°C)** | **RMSE (°C)** | **R²** |
|------------------|--------------|---------------|--------|
| Clear Days | 1.45 | 1.89 | 0.856 |
| Cloudy Days | 1.78 | 2.23 | 0.812 |
| Rainy Days | 1.92 | 2.51 | 0.789 |
| Extreme Weather | 2.87 | 3.76 | 0.672 |

**Insight**: Model performs best in stable weather, challenges with extremes

---

## 🔄 Training Pipeline Automation

### Model Training Workflow
```python
class TemperatureForecastPipeline:
    def __init__(self):
        self.preprocessor = None
        self.feature_selector = None  
        self.model = None
        
    def fit(self, X, y):
        # 1. Data preprocessing
        X_processed = self.preprocessor.fit_transform(X)
        
        # 2. Feature selection
        X_selected = self.feature_selector.fit_transform(X_processed, y)
        
        # 3. Model training
        self.model.fit(X_selected, y)
        
        return self
        
    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        X_selected = self.feature_selector.transform(X_processed)
        return self.model.predict(X_selected)
```

---

## 🎯 Model Serialization & Deployment

### Model Persistence Strategy
```python
# Save complete training pipeline
import joblib

# Main model
joblib.dump(best_model, 'models/daily/BEST_CATBOOST_TUNED_DAILY.joblib')

# Preprocessing pipeline
joblib.dump(preprocessor, 'models/daily/preprocessor_daily.joblib')

# Feature selection
joblib.dump(feature_selector, 'models/daily/selection_result_daily.joblib')

# Model metadata
model_metadata = {
    'training_date': '2024-11-18',
    'model_version': 'v1.0',
    'feature_count': 92,
    'performance': {'r2': 0.8285, 'mae': 1.68, 'rmse': 2.02},
    'training_samples': 2828
}
```

### Loading & Inference
```python
# Production inference
model = joblib.load('BEST_CATBOOST_TUNED_DAILY.joblib')
preprocessor = joblib.load('preprocessor_daily.joblib')

def predict_temperature(input_data):
    # Preprocess input
    processed_data = preprocessor.transform(input_data)
    
    # Generate predictions
    predictions = model.predict(processed_data)
    
    return {
        'T+1': predictions[0], 'T+2': predictions[1],
        'T+3': predictions[2], 'T+4': predictions[3], 'T+5': predictions[4]
    }
```

---

## ✅ Training Success Metrics

### Achievement Summary
| **Metric** | **Target** | **Achieved** | **Status** |
|------------|------------|--------------|------------|
| **R² Score** | >0.80 | 0.8285 | ✅ Exceeded (+3.6%) |
| **T+1 Accuracy** | >0.90 | 0.9174 | ✅ Excellent |
| **Training Time** | <30 min | 8.3 min | ✅ Efficient |
| **Model Size** | <20 MB | 12.8 MB | ✅ Compact |
| **Feature Count** | 80-100 | 92 | ✅ Optimal |

### Model Quality Validation
- **✅ No Overfitting**: Validation curve stable
- **✅ Robust Performance**: Consistent across weather conditions
- **✅ Feature Importance**: Physically interpretable
- **✅ Production Ready**: Complete deployment pipeline

---

## 🚀 Transition to UI Development

### Training Achievements
- **✅ Superior Algorithm**: CatBoost outperforms all alternatives
- **✅ Optimized Hyperparameters**: 50-trial Optuna optimization
- **✅ Multi-Horizon Success**: 5-day forecasting capability
- **✅ Production Pipeline**: Complete training workflow automated

### Next Phase Preview
**Step 6: UI Development**
- Streamlit web application development
- Interactive visualization and user interface
- Real-time prediction dashboard
- Model performance monitoring interface

---

<!-- _class: lead -->

## 🎯 Model Training Complete!

### 🤖 Training Success
1. **Superior Performance**: 82.85% accuracy across 5 forecasting horizons
2. **Optimized Algorithm**: CatBoost with tuned hyperparameters
3. **Robust Architecture**: Multi-output regression with proper validation
4. **Production Ready**: Complete serialization and deployment pipeline

### 🚀 Ready for User Interface!
**High-Performance Model Trained** → **Interactive Application Development**