"""
Data Processing Module for Hanoi Weather Data
Handles data cleaning, preprocessing, and preparation for feature engineering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import os
from typing import Dict, Tuple, List, Optional

warnings.filterwarnings('ignore')


class WeatherDataProcessor:
    """
    Main class for processing Hanoi weather data.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.outlier_summary = {}
        self.quality_report = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load raw weather data from CSV file."""
        if self.verbose:
            print(f"Loading data from: {file_path}")
        
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        if self.verbose:
            print(f"Data loaded successfully!")
            print(f"Shape: {df.shape}")
            print(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df = df.copy()
        
        if self.verbose:
            print("\nHANDLING MISSING VALUES:")
            print("-" * 30)
        
        # Handle preciptype
        if 'preciptype' in df.columns:
            missing_preciptype = df['preciptype'].isnull().sum()
            if missing_preciptype > 0:
                df['preciptype'] = df['preciptype'].fillna('none')
                if self.verbose:
                    print(f"Preciptype: {missing_preciptype} NaN values filled with 'none'")
        
        # Handle severerisk
        if 'severerisk' in df.columns:
            missing_severerisk = df['severerisk'].isnull().sum()
            if missing_severerisk > 0:
                df['severerisk'] = df['severerisk'].fillna(0)
                if self.verbose:
                    print(f"Severerisk: {missing_severerisk} NaN values filled with 0")
        
        # Check remaining missing values
        remaining_missing = df.isnull().sum()
        remaining_missing = remaining_missing[remaining_missing > 0]
        
        if len(remaining_missing) > 0 and self.verbose:
            print(f"Remaining missing values:")
            for col, count in remaining_missing.items():
                pct = (count / len(df)) * 100
                print(f"  {col}: {count} ({pct:.2f}%)")
        elif self.verbose:
            print("All missing values handled!")
        
        return df
    
    def detect_outliers_iqr(self, data: pd.DataFrame, column: str, factor: float = 1.5) -> Tuple:
        """Detect outliers using IQR method."""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    def analyze_outliers(self, df: pd.DataFrame) -> Dict:
        """Analyze outliers in numerical columns."""
        numerical_cols = ['temp', 'tempmax', 'tempmin', 'humidity', 'precip', 
                         'windspeed', 'sealevelpressure', 'cloudcover', 'visibility']
        
        outlier_summary = {}
        
        if self.verbose:
            print("\nOUTLIER ANALYSIS:")
            print("-" * 30)
        
        for col in numerical_cols:
            if col in df.columns:
                outliers, lower, upper = self.detect_outliers_iqr(df, col)
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / len(df)) * 100
                
                outlier_summary[col] = {
                    'count': outlier_count,
                    'percentage': outlier_pct,
                    'lower_bound': lower,
                    'upper_bound': upper
                }
                
                if self.verbose:
                    print(f"{col:15}: {outlier_count:4d} outliers ({outlier_pct:5.1f}%)")
        
        self.outlier_summary = outlier_summary
        return outlier_summary
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic temporal features."""
        df = df.copy()
        
        if self.verbose:
            print("\nCREATING TEMPORAL FEATURES:")
            print("-" * 30)
        
        # Basic time components
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['week_of_year'] = df['datetime'].dt.isocalendar().week
        
        # Season mapping
        season_mapping = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                         3: 'Spring', 4: 'Spring', 5: 'Spring',
                         6: 'Summer', 7: 'Summer', 8: 'Summer',
                         9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
        
        df['season'] = df['month'].map(season_mapping)
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rainy'] = (df['precip'] > 0).astype(int)
        
        # Temperature range
        if 'tempmax' in df.columns and 'tempmin' in df.columns:
            df['temp_range'] = df['tempmax'] - df['tempmin']
        
        temporal_features = ['year', 'month', 'day', 'day_of_year', 'day_of_week', 
                           'week_of_year', 'season', 'is_weekend', 'is_rainy', 'temp_range']
        
        if self.verbose:
            created_features = [f for f in temporal_features if f in df.columns]
            print(f"Created {len(created_features)} temporal features")
            for feature in created_features:
                print(f"  - {feature}")
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Comprehensive data quality validation."""
        report = {}
        
        # Basic stats
        report['total_records'] = len(df)
        report['total_features'] = len(df.columns)
        report['missing_values'] = df.isnull().sum().sum()
        report['duplicate_records'] = df.duplicated().sum()
        
        # Date range validation
        report['date_range'] = {
            'start': df['datetime'].min(),
            'end': df['datetime'].max(),
            'total_days': (df['datetime'].max() - df['datetime'].min()).days
        }
        
        # Numerical ranges validation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        report['value_ranges'] = {}
        
        for col in numerical_cols:
            if col in df.columns:
                report['value_ranges'][col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
        
        self.quality_report = report
        return report
    
    
    def visualize_outliers(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Visualize outliers for key variables."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Outlier Detection - Key Weather Variables', fontsize=16)
        
        variables_to_plot = ['temp', 'humidity', 'precip', 'windspeed']
        for i, var in enumerate(variables_to_plot):
            if var in df.columns and var in self.outlier_summary:
                row, col = i // 2, i % 2
                
                # Box plot
                df.boxplot(column=var, ax=axes[row, col])
                axes[row, col].set_title(f'{var.title()} - {self.outlier_summary[var]["count"]} outliers')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Outlier visualization saved to: {save_path}")
        
        plt.show()
    
    def export_data(self, df: pd.DataFrame, output_path: str):
        """Export processed data to CSV."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export to CSV
        df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"\nData exported to: {output_path}")
            print(f"Final dataset shape: {df.shape}")
    
    def process_data(self, input_path: str, output_path: str) -> pd.DataFrame:
        """
        Complete data processing pipeline.
        
        Args:
            input_path: Path to raw data CSV file
            output_path: Path to save processed data
            
        Returns:
            Processed DataFrame
        """
        if self.verbose:
            print("="*60)
            print("HANOI WEATHER DATA PROCESSING PIPELINE")
            print("="*60)
        
        # Step 1: Load data
        df = self.load_data(input_path)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Analyze outliers
        self.analyze_outliers(df)
        
        # Step 4: Create temporal features
        df = self.create_temporal_features(df)
        
        # Step 5: Validate data quality
        self.validate_data_quality(df)
        
        # Step 6: Run quality checks
        checks_passed, total_checks = self.run_quality_checks(df)
        
        # Step 7: Export processed data
        self.export_data(df, output_path)
        
        if self.verbose:
            print(f"\n🎉 DATA PROCESSING COMPLETED!")
            print(f"Quality Score: {checks_passed}/{total_checks}")
            print(f"Ready for Feature Engineering phase")
        
        return df


# Utility functions
def quick_process(input_path: str, output_path: str, verbose: bool = True) -> pd.DataFrame:
    """Quick processing function for simple use cases."""
    processor = WeatherDataProcessor(verbose=verbose)
    return processor.process_data(input_path, output_path)


if __name__ == "__main__":
    # Example usage
    input_file = "../data/raw/daily_data.csv"
    output_file = "../data/processed/daily_data_cleaned.csv"
    
    # Process data
    processor = WeatherDataProcessor(verbose=True)
    df_processed = processor.process_data(input_file, output_file)
    
    # Optional: Visualize outliers
    processor.visualize_outliers(df_processed, "../reports/outliers_analysis.png")