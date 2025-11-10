import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def process_data(df: pd.DataFrame):
    """Process the weather data for feature engineering.
        return X and y for model training.
    """
    
    df['datetime'] = pd.to_datetime(df['datetime']) 
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['day_of_year'] = df['datetime'].dt.dayofyear

    # Create season mapping
    season_mapping = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}

    df['season'] = df['month'].map(season_mapping)

    # Temperature range
    df['temp_range'] = df['tempmax'] - df['tempmin']

    #Remove unused columns
    df.drop(columns=['name', 'description', 'icon', 'preciptype', 'snow', 'snowdepth', 'stations', 'severerisk', 'conditions'], inplace=True)

    df.set_index('datetime', inplace=True)
    df.index = pd.to_datetime(df.index)
    df['sunrise'] = pd.to_datetime(df['sunrise'])
    df['sunset'] = pd.to_datetime(df['sunset'])
    df['day_length_hours'] = df['sunset'] - df['sunrise']
    df = df.drop(columns=['sunrise', 'sunset'])
    df['day_length_hours'] = df['day_length_hours'].dt.total_seconds() / 3600.0

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    df.drop(columns=['day', 'month', 'day_of_year'], inplace=True)

    df["temp_solar_interaction"] = df["temp"] * df["solarradiation"]
    df["uv_temp_interaction"] = df["uvindex"] * df["temp"]
    df['temp_cloudcover_interaction'] = df['temp'] * df['cloudcover']
    df['temp_sealevelpressure_interaction'] = df['temp'] * df['sealevelpressure']
    df['weighted_precip'] = df['precipprob'] * df['precip']
    df['effective_solar'] = df['solarradiation'] * (1 - df['cloudcover']/100)
    df['precip_impact'] = df['precipprob'] * df['precip']

    df['wind_u'] = df['windspeed'] * np.cos(2 * np.pi * df['winddir'] / 360)  # gió đông-tây
    df['wind_v'] = df['windspeed'] * np.sin(2 * np.pi * df['winddir'] / 360)  # gió nam-bắc
    df = df.drop('winddir', axis=1)

    temp_minus_dew = df['temp'] - df['dew']

    # Create feature moonphase_sin
    df['moonphase_sin'] = np.sin(2 * np.pi * df['moonphase'] / 1)

    # Create feature moonphase_cos
    df['moonphase_cos'] = np.cos(2 * np.pi * df['moonphase'] / 1)

    # Remove original moonphase
    df = df.drop('moonphase', axis=1)

    # Create lagging features
    def create_lag_features(df, cols, lags):
        for col in cols:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    # Specify columns and lags
    # Get all numerical columns
    computing_columns = df.drop(columns=['year', 'season', 'month_sin',
                                        'month_cos', 'dayofyear_sin', 'dayofyear_cos']).columns

    lag_steps = [1, 2, 3, 5, 7, 10, 14, 21, 30]  # Example lag steps

    # Apply lagging features before handling rolling horizons
    df = create_lag_features(df, computing_columns, lag_steps)

    # Function to compute rolling mean and percentage change
    def compute_rolling(df, horizon, col):
        label = f"rolling_{horizon}_{col}"
        df[label] = df[col].rolling(horizon, min_periods=horizon).mean()  # Ensure full horizon is used
        df[f"{label}_change"] = df[col] - df[label]
        return df

    # Compute rolling features for specified horizons
    rolling_horizons = [3, 7, 14, 21, 30]  # Rolling windows of 3, 7, 14 days
    for horizon in rolling_horizons:
        for col in computing_columns:
            df = compute_rolling(df, horizon, col)
    
    #Months and days average
    def expand_mean(df):
        return df.expanding(1).mean()

    for col in computing_columns:
        df[f"month_avg_{col}"] = df[col].groupby(df.index.month, group_keys=False).apply(expand_mean)
        df[f"day_avg_{col}"] = df[col].groupby(df.index.day_of_year, group_keys=False).apply(expand_mean)
        df[f"year_avg_{col}"] = df[col].groupby(df.index.year, group_keys=False).apply(expand_mean)
        df[f"season_avg_{col}"] = df[col].groupby(df['season'], group_keys=False).apply(expand_mean)
        df["month_max_temp"] = df['temp'].groupby(df.index.month, group_keys=False).cummax()
        df["month_min_temp"] = df['temp'].groupby(df.index.month, group_keys=False).cummin()

    df["temp_volatility_7"] = df["temp"].rolling(7).std()
    df["temp_volatility_14"] = df["temp"].rolling(14).std()
    df["temp_volatility_21"] = df["temp"].rolling(21).std()
    df["temp_volatility_30"] = df["temp"].rolling(30).std()
    df["temp_spike_flag"] = (df["temp"] - df["temp"].shift(1)).abs() > 5
    df["temp_anomaly_vs_month_avg"] = df["temp"] - df["month_avg_temp"]
    df["temp_anomaly_vs_season_avg"] = df["temp"] - df["season_avg_temp"]

    df["pressure_trend_3d"] = df["sealevelpressure"] - df["sealevelpressure"].shift(3)
    df["pressure_trend_7d"] = df["sealevelpressure"] - df["sealevelpressure"].shift(30)
    df = df.iloc[30:]

    X = df

    return X

def select_top_k(X, k=93):
    selection_result = joblib.load('selection_result.joblib')

    mean_importance = selection_result['importance_mean']
    feature_names = selection_result['feature_names']
    k_features = k

    sorted_idx = mean_importance.argsort()[::-1]
    top_k_idx = sorted_idx[:k_features]
    top_k_features = [feature_names[i] for i in top_k_idx]
    X = X[top_k_features]

    return X

def build_preprocessing_pipeline_catboost(X):
    # 1. Slpiting categorical and numerical features
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
    cat_cols.extend(bool_cols)
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # removing bool type features
    num_cols = [col for col in num_cols if col not in bool_cols]

    print(f"Numerical Features ({len(num_cols)}): {num_cols}")
    print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")
    print("-" * 50)

    # 2. Pipeline cho số: chỉ impute, KHÔNG scale
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', 'passthrough')  # CatBoost not need to use StandardScaler
    ])

    # 3. ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, num_cols),
            ('cat', 'passthrough', cat_cols)  # CatBoost will automatically handle
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    preprocessor.set_output(transform="pandas")
    
    return preprocessor

def predict_future(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    df_raw: DataFrame chứa dữ liệu mới (có thể là 1 dòng hoặc nhiều dòng)
            PHẢI có đủ 33 cột gốc trước khi feature engineering
            PHẢI có tối thiểu 365 dòng (1 năm)
    """
    checking_columns = ['name', 'datetime', 'tempmax', 'tempmin', 'temp', 'feelslikemax',
       'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob',
       'precipcover', 'preciptype', 'snow', 'snowdepth', 'windgust',
       'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
       'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'sunrise',
       'sunset', 'moonphase', 'conditions', 'description', 'icon', 'stations']
    
    if df_raw.shape[1] != 33:
        print('The dataframe is not matching! Please recheck columns in the dataframe')
        return 0

    if df_raw.columns.to_list() != checking_columns:
        print('The dataframe is not matching! Please recheck columns in the dataframe')
        return 0
    
    if df_raw.shape[0] < 365:
        print('There are not enough rows to predict! Please recheck the dataframe to ensure it has at least 30 rows')
        return 0

    best = joblib.load('BEST_CATBOOST_TIMESERIES.joblib')

    model       = best['model']
    preprocessor = best['preprocessor']
    cols        = best['feature_names']   
    # 1. Preprocess data
    df = df_raw.copy()
    last_date = pd.to_datetime(df_raw['datetime'].iloc[-1])
    X_processed = process_data(df)
    # 2. Apply trained preprocessor
    X_final = preprocessor.transform(X_processed)
    
    X_final = select_top_k(X_final)
    # 4. Prediction
    y_pred_5days = model.predict(X_final)[-1]  # shape (5,)
    y_pred_5days = y_pred_5days[::-1]

    # 3. Create DataFrame for the next 5 days
    
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=5, freq='D')
    
    result = pd.DataFrame({
        'date': future_dates,
        'y_pred': y_pred_5days
    })
    
    return result