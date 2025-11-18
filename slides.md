# Hanoi Temperature Forecasting Project
## 🌡️ Comprehensive Machine Learning System for Weather Prediction

### *Dự án Dự báo Nhiệt độ Hà Nội*
**Using 10 Years of Weather Data & Advanced ML Techniques**

---

## 📊 Project Overview

### Key Highlights
- **Objective**: 5-day temperature forecasting for Hanoi, Vietnam
- **Data Source**: Visual Crossing Weather API (2015-2025)
- **Model**: CatBoost Gradient Boosting
- **Performance**: R² = 0.8285 (82.85% accuracy)
- **Deployment**: Live web application + ONNX optimization

### Timeline & Scale
- **10 years** of historical weather data
- **33 comprehensive** weather features
- **70,000+** hourly observations
- **Multi-horizon** predictions (T+1 to T+5 days)

---

## 🎯 Problem Statement & Motivation

### Why Temperature Forecasting Matters
- **Weather Planning**: Daily decision making for citizens
- **Agriculture**: Crop planning and management
- **Tourism**: Travel recommendations
- **Urban Planning**: Climate-informed decisions
- **Research**: Climate pattern analysis

### Technical Challenges
- **Multi-horizon forecasting** with decreasing accuracy
- **Seasonal patterns** and weather volatility
- **Feature engineering** from 33 raw variables
- **Real-time deployment** requirements

---

## 📈 Dataset Deep Dive

### Comprehensive Weather Features (33 Variables)
| Category | Features | Key Insights |
|----------|----------|--------------|
| **Temperature** | tempmax, tempmin, temp, feelslike* | Target variable range: 15-38°C |
| **Humidity & Pressure** | humidity, dew, sealevelpressure | High correlation with temperature |
| **Solar & UV** | solarradiation, solarenergy, uvindex | Strong predictor (r=0.65) |
| **Precipitation** | precip, precipprob, precipcover | Sparse data, mostly zero values |
| **Wind** | windspeed, winddir, windgust | Low signal for temperature prediction |
| **Atmospheric** | cloudcover, visibility, conditions | Weather system indicators |

### Data Quality Analysis
- **Missing Values**: < 5% for any column
- **Temperature Range**: 15-42°C with clear seasonal cycles
- **Correlation Insights**: `temp ↔ feelslike` (r=0.98), `temp ↔ solarradiation` (r=0.65)

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

## ⚙️ Step 4-5: Advanced Feature Engineering & Model Training

### Feature Engineering Techniques
- **Temporal Features**: Day of year, month, season indicators
- **Lag Features**: Historical temperature values (1-30 day lags)
- **Rolling Statistics**: Moving averages (3, 7, 14, 30-day windows)
- **Cyclical Encoding**: Sin/cos transformation for seasonal patterns
- **Derived Features**: Temperature differences, rate of change

### Model Selection & Performance
| Model | R² Score | MAE (°C) | RMSE (°C) | Performance vs CatBoost |
|-------|----------|----------|-----------|------------------------|
| **CatBoost** | **0.8285** | **1.68** | **2.02** | **Best (Baseline)** |
| Ridge | 0.8109 | 1.69 | 2.21 | -2.1% R², +9.6% RMSE |
| Random Forest | 0.8078 | 1.76 | 2.23 | -2.5% R², +10.5% RMSE |

---

## 🎯 Model Performance Deep Dive

### Multi-Horizon Performance Results
| Horizon | R² Score | MAE (°C) | RMSE (°C) | Performance Level |
|---------|----------|----------|-----------|-------------------|
| **T+1** | **0.9174** | **1.14** | **1.46** | Excellent |
| **T+2** | **0.8477** | **1.55** | **1.98** | Very Good |
| **T+3** | **0.8126** | **1.73** | **2.20** | Good |
| **T+4** | **0.7906** | **1.85** | **2.33** | Good |
| **T+5** | **0.7741** | **1.92** | **2.42** | Acceptable |

### Performance Insights
- **Performance Degradation**: -15.62% from T+1 to T+5 (expected)
- **Feature Contribution**: 92 selected from 150+ engineered features
- **Hyperparameter Optimization**: 50 Optuna trials, 60-minute optimization

---

## 📱 Step 6-7: Deployment & Monitoring

### User Interface Development
- **Framework**: Streamlit web application
- **Live Demo**: 🌐 [Access Live Application](https://hanoi-temperature-forecasting.streamlit.app/)
- **Features**: 
  - Real-time predictions
  - Historical data visualization
  - Interactive charts
  - Model performance metrics

### Performance Monitoring System
- **Automated Retraining**: Triggered by performance degradation
- **Alert Thresholds**: 30% RMSE increase (T+1), 20% (T+2-5)
- **30-Day Results**: 5 automatic retrains, performance recovery achieved
- **Model Versioning**: v1→v5 with systematic updates

---

## ⏰ Step 8-9: Advanced Features & Production

### Hourly Data Enhancement
- **Dataset Scale**: 70,000+ hourly observations
- **Granular Predictions**: Hour-by-hour forecasting
- **Enhanced Features**: 3h, 6h, 12h rolling windows
- **Performance**: R² = 0.9328 (T+1 hour), 0.7801 (T+5 days)

### ONNX Production Optimization
- **Conversion Success**: 5 single-target models from multi-target CatBoost
- **Performance Boost**: 1.51x faster inference (0.0003s per prediction)
- **Accuracy Preservation**: <0.02°C maximum deviation
- **Deployment Benefits**: Cross-platform compatibility

---

## 🏆 Technical Achievements & Results

### Key Performance Metrics
- **Overall Accuracy**: 82.85% variance explained
- **Best Short-term**: 91.74% accuracy for next-day prediction
- **Production Speed**: 1.51x faster with ONNX optimization
- **Model Stability**: Consistent performance across weather conditions

### Production System Results
- **Zero Downtime**: Seamless model updates
- **Automated Monitoring**: Real-time performance tracking  
- **Scalable Architecture**: Easy extension to other cities
- **Cross-Platform**: ONNX deployment flexibility

---

## 🔮 Key Learning Outcomes

### Data Science Insights
- **Feature Engineering Impact**: Lag features contribute 60% of predictive power
- **Temporal Patterns**: Strong seasonal cycles with 365-day stability
- **Model Selection**: CatBoost superior for multi-output regression
- **Performance Trade-offs**: Accuracy vs computational efficiency

### Production Learnings
- **Data Leakage Prevention**: Temporal splits crucial for realistic performance
- **Monitoring Strategy**: Proactive alerts prevent performance degradation
- **Deployment Optimization**: ONNX provides significant speed improvements
- **User Experience**: Streamlit enables rapid prototype-to-production

---

## 🚀 Applications & Impact

### Real-World Applications
- **Daily Weather Planning**: Citizens and businesses
- **Agricultural Decision Making**: Crop planning optimization
- **Tourism Industry**: Weather-based recommendations
- **Urban Planning**: Climate-informed infrastructure decisions
- **Research**: Climate pattern analysis for Hanoi region

### Technical Impact
- **End-to-End ML Pipeline**: Complete workflow demonstration
- **Production-Ready System**: Live deployment with monitoring
- **Scalable Architecture**: Framework for other cities/regions
- **Open Source Contribution**: Community learning resource

---

## 📊 System Architecture Overview

```
├── Data Layer (Visual Crossing API)
├── Processing Pipeline (33 → 92 features)
├── ML Models (CatBoost + ONNX)
├── Monitoring System (Automated retraining)
├── Web Interface (Streamlit)
└── Production Deployment (Cross-platform)
```

### Technical Stack
- **ML**: CatBoost, Scikit-learn, Optuna
- **Data**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn  
- **Deployment**: ONNX Runtime, Streamlit
- **Monitoring**: Custom performance tracking

---

## 🎯 Future Enhancements

### Technical Roadmap
- **LLM Integration**: Enhanced weather description processing
- **Multi-City Expansion**: Extend to other Vietnamese cities
- **Advanced Models**: Transformer architectures for sequence modeling
- **Real-time Data**: Integration with IoT weather stations

### Research Opportunities
- **Climate Change Impact**: Long-term trend analysis
- **Extreme Weather**: Enhanced prediction for weather events
- **Ensemble Methods**: Combining multiple model approaches
- **Feature Importance**: Deep analysis of weather predictors

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
- ✅ **82.85% accuracy** for temperature forecasting
- ✅ **Live web application** with real-time predictions
- ✅ **Production-ready system** with automated monitoring
- ✅ **Open source contribution** to ML community
- ✅ **Comprehensive documentation** and learning resource

### Contact & Resources
- **GitHub**: [@Duongvu05](https://github.com/Duongvu05)
- **Live Demo**: [Streamlit Application](https://hanoi-temperature-forecasting.streamlit.app/)
- **Documentation**: Complete project README with technical details

**Questions & Discussion Welcome!** 🙋‍♂️🙋‍♀️