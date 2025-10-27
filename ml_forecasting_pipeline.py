import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TemperatureForecastPipeline:
    """Pipeline đơn giản cho dự đoán nhiệt độ 5 ngày tiếp theo"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_prepare_data(self):
        """Load và chuẩn bị dữ liệu"""
        print("Loading data...")
        
        data_path = '/home/vungocduong/Hanoi-Temperature-Forecasting/data/raw/daily/Daily_Data.csv'
        df = pd.read_csv(data_path)
        
        print(f"Original data shape: {df.shape}")
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Remove features with high missing values
        missing_threshold = 0.05
        missing_pct = df.isnull().sum() / len(df)
        valid_features = missing_pct[missing_pct <= missing_threshold].index.tolist()
        
        # Remove non-feature columns
        exclude_cols = ['datetime', 'name', 'resolvedAddress', 'tzoffset']
        features = [col for col in valid_features if col not in exclude_cols]
        
        print(f"Selected features: {len(features)}")
        print(f"Features: {features}")
        
        # Keep only valid features
        df = df[['datetime'] + features].copy()
        
        # Remove any remaining missing values
        df = df.dropna()
        
        print(f"Final data shape: {df.shape}")
        
        return df, features
    
    def encode_categorical_features(self, df, features):
        """Encode categorical features"""
        for feature in features:
            if df[feature].dtype == 'object':
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le
                print(f"Encoded categorical feature: {feature}")
        
        return df
    
    def create_sliding_windows(self, df, features):
        """Tạo sliding windows: 30 ngày history -> 5 ngày forecast"""
        print("Creating sliding windows...")
        
        window_size = 30
        forecast_days = 5
        
        X, y = [], []
        
        for i in range(window_size, len(df) - forecast_days + 1):
            # Features: 30 ngày trước
            X_window = df[features].iloc[i-window_size:i].values.flatten()
            
            # Target: nhiệt độ 5 ngày tiếp theo
            y_window = df['temp'].iloc[i:i+forecast_days].values
            
            X.append(X_window)
            y.append(y_window)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Windows created - X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y):
        """Chia train/test theo thời gian"""
        split_idx = int(len(X) * 0.8)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"Train samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def initialize_models(self):
        """Khởi tạo các models"""
        models = {
            'Linear Regression': MultiOutputRegressor(LinearRegression()),
            'Ridge Regression': MultiOutputRegressor(Ridge(alpha=1.0)),
            'Lasso Regression': MultiOutputRegressor(Lasso(alpha=1.0)),
            'Random Forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
            'XGBoost': MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, random_state=42)),
            'LightGBM': MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)),
            'CatBoost': MultiOutputRegressor(cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=False)),
            'Decision Tree': MultiOutputRegressor(DecisionTreeRegressor(random_state=42)),
            'KNN': MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5)),
            'SVR': MultiOutputRegressor(SVR(kernel='rbf'))
        }
        
        return models
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train và evaluate models"""
        print("Training models for 5-day temperature forecasting...")
        
        models = self.initialize_models()
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                start_time = datetime.now()
                model.fit(X_train, y_train)
                train_time = (datetime.now() - start_time).total_seconds()
                
                # Predictions
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics for each day
                day_metrics = {}
                for day in range(5):
                    rmse = np.sqrt(mean_squared_error(y_test[:, day], y_pred_test[:, day]))
                    mae = mean_absolute_error(y_test[:, day], y_pred_test[:, day])
                    r2 = r2_score(y_test[:, day], y_pred_test[:, day])
                    
                    day_metrics[f'day_{day+1}'] = {
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2
                    }
                
                # Overall metrics
                overall_rmse = np.mean([day_metrics[f'day_{i+1}']['rmse'] for i in range(5)])
                overall_mae = np.mean([day_metrics[f'day_{i+1}']['mae'] for i in range(5)])
                overall_r2 = np.mean([day_metrics[f'day_{i+1}']['r2'] for i in range(5)])
                
                results[name] = {
                    'model': model,
                    'overall_rmse': overall_rmse,
                    'overall_mae': overall_mae,
                    'overall_r2': overall_r2,
                    'day_metrics': day_metrics,
                    'train_time': train_time
                }
                
                print(f"  {name}: Avg RMSE={overall_rmse:.3f}, Avg R²={overall_r2:.3f}")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                continue
        
        return results
    
    def save_results(self, results):
        """Save results"""
        summary_data = []
        
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'Overall_RMSE': metrics['overall_rmse'],
                'Overall_MAE': metrics['overall_mae'],
                'Overall_R2': metrics['overall_r2'],
                'Train_Time': metrics['train_time']
            }
            
            # Add daily metrics
            for day in range(1, 6):
                day_key = f'day_{day}'
                if day_key in metrics['day_metrics']:
                    day_data = metrics['day_metrics'][day_key]
                    row[f'Day{day}_RMSE'] = day_data['rmse']
                    row[f'Day{day}_MAE'] = day_data['mae']
                    row[f'Day{day}_R2'] = day_data['r2']
            
            summary_data.append(row)
        
        df_results = pd.DataFrame(summary_data)
        df_results.to_csv('model_comparison_results.csv', index=False)
        print("\nResults saved to model_comparison_results.csv")
        
        return df_results
    
    def print_summary(self, results):
        """Print summary"""
        print("\n" + "="*60)
        print("SUMMARY - 5-DAY TEMPERATURE FORECASTING")
        print("="*60)
        
        # Sort by overall R²
        sorted_results = sorted(results.items(), key=lambda x: x[1]['overall_r2'], reverse=True)
        
        print(f"{'Model':<20} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Time(s)':<8}")
        print("-" * 60)
        
        for name, metrics in sorted_results:
            print(f"{name:<20} {metrics['overall_rmse']:<8.3f} "
                  f"{metrics['overall_mae']:<8.3f} {metrics['overall_r2']:<8.3f} "
                  f"{metrics['train_time']:<8.1f}")
        
        # Best model details
        best_model_name, best_metrics = sorted_results[0]
        print(f"\nBest Model: {best_model_name}")
        print(f"Overall Performance: RMSE={best_metrics['overall_rmse']:.3f}, "
              f"MAE={best_metrics['overall_mae']:.3f}, R²={best_metrics['overall_r2']:.3f}")
        
        print("\nDaily Performance:")
        for day in range(1, 6):
            day_key = f'day_{day}'
            day_data = best_metrics['day_metrics'][day_key]
            print(f"  Day {day}: RMSE={day_data['rmse']:.3f}, "
                  f"MAE={day_data['mae']:.3f}, R²={day_data['r2']:.3f}")
    
    def run_pipeline(self):
        """Run complete pipeline"""
        print("=== Temperature Forecasting ML Pipeline (5 Days) ===\n")
        
        # 1. Load data
        df, features = self.load_and_prepare_data()
        if df is None:
            return None
        
        # 2. Encode categorical features
        df = self.encode_categorical_features(df, features)
        
        # 3. Create sliding windows
        X, y = self.create_sliding_windows(df, features)
        
        # 4. Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # 5. Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print(f"Final data shape:")
        print(f"  Features: {X_train_scaled.shape[1]}")
        print(f"  Training samples: {X_train_scaled.shape[0]}")
        print(f"  Test samples: {X_test_scaled.shape[0]}")
        print(f"  Target shape: {y_train.shape} (5-day forecasting)")
        
        # 6. Train and evaluate models
        results = self.train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # 7. Save and display results
        df_results = self.save_results(results)
        self.print_summary(results)
        
        return results, df_results

def main():
    """Main function"""
    pipeline = TemperatureForecastPipeline()
    results, df_results = pipeline.run_pipeline()
    return pipeline, results, df_results

if __name__ == "__main__":
    pipeline, results, df_results = main()